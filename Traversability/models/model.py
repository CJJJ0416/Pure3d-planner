"""可通行性代价模型：PTv3 几何编码 -> 三维位置查询 -> FiLM 条件化代价。

整体流程：
1. 用预训练或随机初始化的 PTv3 编码器分别提取局部精细点云与全楼粗点云的几何特征；
2. 训练在 Tomogram 监督位置查询，推理在建筑 PCD 的全部原始位置查询；
3. 将机器人四维条件经 FiLM（特征线性调制）注入查询特征；
4. 由代价头输出 [查询点数, 条件数] 的归一化可通行代价。
"""
import point_transformer_v3 as ptv3
import torch
from torch import nn


class QueryPool(nn.Module):
    """对三维几何近邻做注意力聚合，并显式使用米制相对位置。

    输入中每个查询点带有 K 个近邻索引及其相对坐标。网络先将近邻特征与
    相对位置编码相加，再用标量注意力权重加权求和，得到该查询点的几何描述子。
    """
    def __init__(self, width):
        """初始化相对位置编码与近邻打分网络。"""
        super().__init__()
        # 将 (dx, dy, dz, 距离) 四维几何关系映射到特征宽度
        self.position = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        # 为每个近邻打分，softmax 后得到聚合权重
        self.score = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features, indices, relative, mask=None):
        """聚合 [Q,K] 近邻特征，返回 [Q,width] 的查询描述子。"""
        # relative: [Q, K, 3] 米制相对坐标；distance: [Q, K, 1]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        # 取出 K 个近邻特征，并叠加相对位置编码
        value = features[indices] + self.position(torch.cat((relative, distance), dim=-1))
        # 在近邻维度 K 上做 softmax，得到每个近邻的贡献权重
        scores = self.score(value)
        if mask is not None:
            scores = scores.masked_fill(~mask[..., None], -torch.inf)
        weights = scores.softmax(dim=1)
        return (weights * value).sum(dim=1)


class ConditionedCost(nn.Module):
    """机器人条件化的可通行代价网络。

    参数:
        config: 模型配置，需包含 encoder、weights、width、point_chunk_size 等字段。
        condition_scale: 学生条件的归一化尺度，当前顺序为 clearance_min、slope_max、step_max、stand_ratio。
        pretrained: 为 True 时加载配置中的预训练 PTv3 权重；否则按预设结构新建编码器。
    """
    def __init__(self, config, condition_scale, pretrained=True):
        """按配置构建 PTv3 编码器、查询聚合、FiLM 条件头与代价头。"""
        super().__init__()
        self.config = config
        if pretrained:
            # head=False：只使用编码器主干，代价头由本模块自行实现
            self.encoder = ptv3.load(config["encoder"], weights=config["weights"], device="cpu",
                                     head=False, precision="fp32", warmup=False)
        else:
            # 评估/推理时从预设结构新建编码器，随后加载 checkpoint 权重
            self.encoder = ptv3.PointTransformerV3.from_preset(config["encoder"])
            self.encoder.precision = "fp32"
        width = config["width"]
        # 将编码器输出投影到统一的特征宽度
        self.projection = nn.Sequential(nn.Linear(self.encoder.feature_channels, width),
                                        nn.LayerNorm(width), nn.GELU())
        # 局部精细几何与全楼粗几何各用一个查询聚合器
        self.local_query = QueryPool(width)
        self.global_query = QueryPool(width)
        # 融合两种尺度的查询特征
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.GELU())
        # 四维学生条件生成 FiLM 参数，stand_ratio 对应教师 standable_ratio。
        # interval_free 留在教师侧固定为 0.5 m。
        self.condition_head = nn.Sequential(nn.Linear(len(condition_scale), 64), nn.GELU(), nn.Linear(64, 2 * width))
        # 代价头输出单个标量，再经 sigmoid 归一化到 [0, 1]
        self.cost_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        # 条件尺度作为 buffer 随模型保存与迁移，不参与梯度更新
        self.register_buffer("condition_scale", torch.tensor(condition_scale, dtype=torch.float32))

    def encode(self, point):
        """编码几何，按多尺度 upcast 恢复到 prepare 后的体素行序。"""
        # 使用可训练的前向 + 多尺度上采样；embed() 会关闭梯度，不能用于训练。
        return self.projection(ptv3.upcast(self.encoder(point)).feat)

    def encode_geometry(self, batch):
        """编码一次几何，供全部 K 组机器人条件共享；不读取真实代价标签。"""
        local = self.encode(batch["point"])
        coarse = self.encode(batch["global_point"])
        chunks = []
        # 查询点较多时分块聚合，避免显存峰值过高
        for first in range(0, len(batch["query_coord"]), self.config["point_chunk_size"]):
            last = first + self.config["point_chunk_size"]
            fine_query = self.local_query(local, batch["local_index"][first:last], batch["local_relative"][first:last],
                                          batch["local_neighbor_mask"][first:last])
            coarse_query = self.global_query(coarse, batch["global_index"][first:last], batch["global_relative"][first:last])
            chunks.append(self.fusion(torch.cat((fine_query, coarse_query), dim=-1)))
        return torch.cat(chunks)

    def decode_cost(self, geometry, conditions, query_batch):
        """返回 [Q,K] 的归一化代价；切换条件时可复用已编码几何。"""
        if self.config.get("decoder_precision") == "fp32":
            with torch.autocast(geometry.device.type, enabled=False):
                return self._decode_cost(geometry.float(), conditions.float(), query_batch)
        return self._decode_cost(geometry, conditions, query_batch)

    def _decode_cost(self, geometry, conditions, query_batch):
        # FiLM：gamma 和 beta 各为 [B, K, width]，按查询所属样本索引取值
        gamma, beta = self.condition_head(conditions / self.condition_scale).chunk(2, dim=-1)
        result = []
        for first in range(0, len(geometry), self.config["point_chunk_size"]):
            last = first + self.config["point_chunk_size"]
            batch = query_batch[first:last]
            # 对查询特征做逐通道仿射调制，再送入代价头
            conditioned = (1 + gamma[batch]) * geometry[first:last, None] + beta[batch]
            result.append(self.cost_head(conditioned).squeeze(-1).sigmoid().float())
        return torch.cat(result)

    def forward(self, batch):
        """训练/推理统一入口：先编码几何，再按批次内条件解码代价。"""
        return self.decode_cost(self.encode_geometry(batch), batch["conditions"], batch["query_batch"])
