"""The official PTv3 and a chunked map decoder for full current-frame training."""
# 骨干直接来自官方 PointTransformerV3 @ 4e277e6。
# 本文件的地图查询与解码 MLP 是 Tomography 任务适配，见官方比对文档。
import torch
from torch import nn

from point_transformer_v3 import Config, PointTransformerV3, prepare
from ptv3_tomography.models.perception import SpatialMapDecoder


class ChunkedMapDecoder(SpatialMapDecoder):
    """Bound query-to-point distance allocation without dropping spatial tokens."""
    def __init__(self, channels, map_config, hidden_channels, query_chunk_size):
        super().__init__(channels, map_config, hidden_channels)
        self.query_chunk_size = query_chunk_size

    def forward(self, coordinates, features, up):
        queries = self.query_coordinates(up)
        output = []
        for start in range(0, len(queries), self.query_chunk_size):
            query = queries[start:start + self.query_chunk_size]
            height = self.query_height[start:start + self.query_chunk_size]
            # Distance geometry stays fp32 even when the backbone/MLP use bf16.
            with torch.autocast(device_type=coordinates.device.type, enabled=False):
                distances, indices = torch.cdist(query.float(), coordinates.float()).topk(min(3, len(coordinates)), largest=False)
                weights = distances.clamp_min(1e-4).reciprocal()
                weights = weights / weights.sum(-1, keepdim=True)
                local = (features[indices].float() * weights[..., None]).sum(1)
            output.append(self.mlp(torch.cat((local, query, height[:, None]), dim=-1)))
        raw = torch.cat(output).float()
        return {"cost": raw[:, 0].sigmoid().reshape(self.shape),
                "height": (self.query_height + self.height_half_width * raw[:, 1].tanh()).reshape(self.shape)}


    def forward_batch(self, coordinates, features, offset, up, max_tokens):
        """Interpolate and decode all clouds together, chunking only the query axis."""
        batch_size = up.shape[0]
        # 官方 packed backbone 返回输入体素层的特征，offset 表达独立帧边界。
        # 项目地图头把它临时组织成 [B,max_tokens,C]，以 B 维并行做插值。
        starts = torch.cat((offset.new_zeros(1), offset[:-1]))
        counts = offset - starts
        batch = torch.repeat_interleave(torch.arange(batch_size, device=offset.device), counts,
                                        output_size=coordinates.shape[0])
        local_index = torch.arange(len(coordinates), device=offset.device) - starts[batch]
        padded_coordinates = coordinates.new_zeros((batch_size, max_tokens, 3))
        padded_features = features.new_zeros((batch_size, max_tokens, features.shape[-1]))
        padded_coordinates[batch, local_index] = coordinates
        padded_features[batch, local_index] = features
        # 填充位只用于张量并行，后续距离设为 inf，近邻只来自同帧的真实 token。
        valid = torch.arange(max_tokens, device=offset.device)[None, :] < counts[:, None]
        # 固定 LiDAR-XY 加重力高度生成全部查询，满足 dot(up,query)=band_center。
        # GT 高度和监督掩码不参与前向坐标构造。
        z = (self.query_height[None, :] - (self.query_xy[None, :, :] * up[:, None, :2]).sum(-1)) / up[:, None, 2]
        queries = torch.cat((self.query_xy[None].expand(batch_size, -1, -1), z[..., None]), dim=-1)
        batch_index = torch.arange(batch_size, device=offset.device)[:, None, None]
        output = []
        # 分块只改变一次分配的查询工作区，最终仍预测全部 6*64*64 个格。
        for start in range(0, queries.shape[1], self.query_chunk_size):
            query = queries[:, start:start + self.query_chunk_size]
            height = self.query_height[start:start + self.query_chunk_size]
            with torch.autocast(device_type=coordinates.device.type, enabled=False):
                distance = torch.cdist(query.float(), padded_coordinates.float())
                distance = distance.masked_fill(~valid[:, None, :], float('inf'))
                distances, indices = distance.topk(min(3, max_tokens), largest=False)
                weights = distances.clamp_min(1e-4).reciprocal()
                weights = weights / weights.sum(-1, keepdim=True)
                # features 保持可微，插值结果把地图损失传回完整 PTv3。
                neighbours = padded_features[batch_index, indices].float()
                local = (neighbours * weights[..., None]).sum(2)
            inputs = torch.cat((local, query, height[None, :, None].expand(batch_size, -1, -1)), dim=-1)
            output.append(self.mlp(inputs))
        raw = torch.cat(output, dim=1).float()
        shape = (batch_size, *self.shape)
        return {"cost": raw[..., 0].sigmoid().reshape(shape),
                "height": (self.query_height + self.height_half_width * raw[..., 1].tanh()).reshape(shape)}


class SingleFramePerception(nn.Module):
    """Task wrapper. Config selects the unchanged official PTv3 model size."""
    def __init__(self, config, map_config):
        super().__init__()
        cfg = Config.from_dict(config["backbone"])
        self.voxel_size_m = config["voxel_size_m"]
        # 官方 Config 构造完整 5 级编码/4 级解码。这里没有冻结骨干参数。
        # precision="fp32" 作为独立推理默认值。训练器的外层 BF16 autocast 优先，
        # 对应官方 model.py::PointTransformerV3._autocast，不会强制训练为 FP32。
        self.backbone = PointTransformerV3(config=cfg, precision="fp32")
        self.decoder = ChunkedMapDecoder(cfg.dec_channels[0], map_config,
                                        config["decoder_hidden_channels"], config["query_chunk_size"])

    def forward(self, points, up):
        point, _ = prepare(points, feat=("coord",), grid_size=self.voxel_size_m, center=False, pick="first")
        tokens = self.backbone(point)
        return self.decoder(tokens.coord, tokens.feat, up), int(len(tokens.feat))

    def forward_batch(self, point, up, max_tokens):
        # 官方骨干一次处理全部 packed 点，输出 [sum(N_i),64]。
        # 本项目地图头继续一次处理 B 帧，输出 cost/height 各 [B,6,64,64]。
        tokens = self.backbone(point)
        return self.decoder.forward_batch(tokens.coord, tokens.feat, point["offset"], up, max_tokens)
