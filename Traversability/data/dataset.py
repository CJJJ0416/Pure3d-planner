"""为独立建筑输入与原生 Tomogram 查询构造 PTv3 批次。

这里把 blocks.py 给出的“块索引 + 原始点云”转换成模型可直接消费的批次：
- 局部几何按 5 cm 体素化，全楼粗几何按 0.5 m 体素化；
- 训练时随机绕世界 Z 轴旋转做数据增强；
- 查询点本身不参与体素化，而是通过预计算的三维近邻表访问几何特征。
"""
import numpy as np
import point_transformer_v3 as ptv3
import torch
from torch.utils.data import Sampler

from Traversability.data.blocks import TomogramBlocks, collate_tomogram_blocks
from Traversability.training.condition_response import single_parameter_pairs


def prepare_cloud(coord, model_config, grid_size, pick="first", seed=0):
    """在现有块坐标系内体素化几何，并保持米制尺度不变。"""
    coord = torch.as_tensor(np.asarray(coord, dtype=np.float32))
    # 输入只有 XYZ，颜色与法线用零向量占位，满足 PTv3 的接口要求
    missing = torch.zeros_like(coord)
    point, inverse = ptv3.prepare(
        coord, color=missing, normal=missing, preset=model_config["encoder"],
        grid_size=grid_size, scale=1.0, center=False, pick=pick,
        generator=torch.Generator().manual_seed(seed), device="cpu",
    )
    # inverse 记录每个原始点所属的体素索引，用于把近邻表映射到体素特征
    return dict(point), inverse


def prepare_geometry_sample(sample, model_config, seed=0, augment=False):
    """共享的几何体素化与查询索引准备，不读取任何 cost 或条件标签。"""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi) if augment and model_config.get("augment_yaw", True) else 0.
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    coord = sample["input_coord"] @ rotation.T
    coarse = sample["global_coord"] @ rotation.T
    query = sample["query_coord"] @ rotation.T
    pick = "random" if augment else "first"
    point, inverse = prepare_cloud(coord, model_config, model_config["voxel_size_m"], pick, seed)
    global_point, global_inverse = prepare_cloud(
        coarse, model_config, model_config["global_voxel_size_m"], pick, seed,
    )
    local_knn, global_knn = sample["query_neighbors"], sample["global_neighbors"]
    return dict(
        point=point, global_point=global_point,
        local_index=inverse[local_knn], global_index=global_inverse[global_knn],
        local_neighbor_mask=torch.from_numpy(sample["query_neighbor_mask"]),
        local_relative=torch.from_numpy(coord[local_knn] - query[:, None]),
        global_relative=torch.from_numpy(coarse[global_knn] - query[:, None]),
        query_coord=torch.from_numpy(query), query_index=torch.from_numpy(sample["query_index"]),
        origin=torch.from_numpy(sample["origin"]), rotation=torch.from_numpy(rotation),
        input_count=len(coord), global_count=len(coarse), scene=sample["scene"], block=sample["block"],
    )


class PointBlocks(TomogramBlocks):
    """在 TomogramBlocks 基础上完成体素化与数据增强的 PyTorch 数据集。"""

    def __init__(self, directory, split, model_config, seed, augment=False, condition_bank=None):
        """建立学生条件列映射和每个场景可训练的 profile 列。"""
        super().__init__(directory, split)
        self.model_config, self.seed, self.augment = model_config, seed, augment
        self.condition_bank = condition_bank
        self.condition_names = model_config["condition_names"]
        # 教师元数据保存的是完整条件顺序；这里建立“学生条件名 -> 教师列号”的映射
        columns = {name: i for i, name in enumerate(self.metadata["condition_names"])}
        self.condition_columns = [columns[name] for name in self.condition_names]
        # condition_scale 也按学生条件顺序重排，保证 FiLM 归一化与列号一致
        self.condition_scale = [self.metadata["condition_scale"][i] for i in self.condition_columns]
        fixed = model_config["fixed_teacher_conditions"]
        # 教师缓存保留所有对照组。学生学习固定净空下的四维条件，保留全部查询点。
        # profile_columns 记录每个场景中满足固定条件的 profile 列索引。
        self.profile_columns = {
            scene["id"]: [i for i, profile in enumerate(scene["profiles"])
                          if all(profile["condition"][columns[name]] == value
                                 for name, value in fixed.items())]
            for scene in self.metadata["scenes"]
        }
        self.profile_banks = {s["id"]:s.get("profile_banks", [self.profile_columns[s["id"]]])
                              for s in self.metadata["scenes"]}
        self.profile_columns = {name:banks[0] for name,banks in self.profile_banks.items()}
        # 取第一个场景的 profile ID 作为指标输出的条件名称
        first = self.scene_rows[self.rows[0]["scene"]]
        self.profile_ids = [first["profiles"][i]["id"] for i in self.profile_columns[first["id"]]]
        self.condition_pairs = single_parameter_pairs(
            [[first["profiles"][i]["condition"][column] for column in self.condition_columns]
             for i in self.profile_columns[first["id"]]])
        if "condition_pairs" in self.metadata:
            self.condition_pairs = self.metadata["condition_pairs"]

    def __getitem__(self, item):
        """读取一个块并完成旋转增强、体素化和近邻索引映射。"""
        # 支持 (epoch, index) 元组，使每个 epoch 的随机增强可复现且不回放旧顺序
        epoch, index, draw = (*item, 0) if isinstance(item, tuple) and len(item) == 2 else (
            item if isinstance(item, tuple) else (0, item, 0))
        sample = super().__getitem__(index)
        # 只取符合固定教师条件的 profile 列，并按学生条件顺序重排四维条件
        banks = self.profile_banks[sample["scene"]]
        bank = self.condition_bank if self.condition_bank is not None else (epoch % len(banks) if self.augment else 0)
        profiles = banks[bank]
        target = sample["target"][:, profiles]
        conditions = sample["conditions"][profiles][:, self.condition_columns]
        # 由全局种子、epoch 与块索引共同决定增强随机数，保证可复现
        draw_seed = self.seed + epoch * len(self) + index + draw * 100003
        result = prepare_geometry_sample(sample, self.model_config, draw_seed, self.augment)
        result.update(target=torch.from_numpy(target), conditions=torch.from_numpy(conditions))
        result["condition_bank"] = bank
        if "facility_mask" in self.scenes[sample["scene"]]:
            result["facility_mask"] = torch.from_numpy(np.array(
                self.scenes[sample["scene"]]["facility_mask"][sample["query_index"]]))
        return result


class EpochSampler(Sampler):
    """把 epoch 编号一并传给 persistent DataLoader worker，保证增强可复现。

    每个 epoch 用 (seed + epoch) 生成新的块访问顺序，并把 (epoch, index)
    作为样本键交给数据集，使增强随机数既随 epoch 变化又可重复。
    """
    def __init__(self, dataset, seed, samples_per_epoch=None, stair_sampling=None):
        """每轮无放回抽取指定数量的块，使不同训练集可以比较相同更新预算。"""
        self.dataset, self.seed, self.epoch = dataset, seed, 0
        self.samples_per_epoch = len(dataset) if samples_per_epoch is None else samples_per_epoch
        self.stair_sampling = stair_sampling

    def __iter__(self):
        """按 (epoch, index) 顺序产出本轮块索引。"""
        # 使用独立的 torch 生成器，避免影响全局随机状态
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.stair_sampling:
            rows = self.dataset.rows
            use_replay = self.stair_sampling["source"] == "replay"
            stairs = torch.tensor([i for i,r in enumerate(rows)
                                   if r["stair_block"] and r["replay"] == use_replay])
            replay = torch.tensor([i for i,r in enumerate(rows) if r["replay"]])
            count = self.stair_sampling["stair_blocks"]
            balance_by = self.stair_sampling.get('balance_by')
            if balance_by:
                # 各设施模板等额抽样，避免长梯或大井口因块数更多而主导训练。
                groups = sorted({rows[i][balance_by] for i in stairs.tolist()})
                chosen = []
                for group_index, group in enumerate(groups):
                    members = torch.tensor([i for i in stairs.tolist() if rows[i][balance_by] == group])
                    size = count // len(groups) + (group_index < count % len(groups))
                    chosen.append(members[torch.randint(len(members), (size,), generator=generator)])
                selected_stairs = torch.cat(chosen)
            else:
                selected_stairs = stairs[torch.randint(len(stairs), (count,), generator=generator)]
            order = torch.cat((selected_stairs,
                               replay[torch.randint(len(replay), (self.samples_per_epoch-count,), generator=generator)]))
            order = order[torch.randperm(len(order), generator=generator)]
            return iter((self.epoch, index, draw) for draw,index in enumerate(order.tolist()))
        order = torch.randperm(len(self.dataset), generator=generator)
        return iter((self.epoch, index) for index in order[:self.samples_per_epoch].tolist())

    def __len__(self):
        """返回本轮抽取的块数。"""
        return self.samples_per_epoch


def collate_queries(samples):
    """打包纯几何、XYZ 查询与机器人条件；不需要训练标签。"""
    # 每个块的点数不同，累加偏移量用于把近邻索引映射到拼接后的点云
    local_offsets = np.cumsum([0, *[len(s["point"]["coord"]) for s in samples[:-1]]])
    global_offsets = np.cumsum([0, *[len(s["global_point"]["coord"]) for s in samples[:-1]]])
    batch = dict(
        # ptv3.collate 负责拼接点云并生成 offset/batch 等结构信息
        point=dict(ptv3.collate([s["point"] for s in samples])),
        global_point=dict(ptv3.collate([s["global_point"] for s in samples])),
        local_index=torch.cat([s["local_index"] + int(o) for s,o in zip(samples,local_offsets)]),
        global_index=torch.cat([s["global_index"] + int(o) for s,o in zip(samples,global_offsets)]),
        # query_batch 标记每个查询点属于批次中的哪个块（用于 FiLM 条件索引）
        query_batch=torch.cat([torch.full((len(s["query_coord"]),), i, dtype=torch.long) for i,s in enumerate(samples)]),
        query_offset=torch.tensor([len(s["query_coord"]) for s in samples]).cumsum(0),
        input_counts=torch.tensor([s["input_count"] for s in samples]),
        global_counts=torch.tensor([s["global_count"] for s in samples]),
        scene=[s["scene"] for s in samples], block=[s["block"] for s in samples],
    )
    # 逐点/逐近邻数据直接沿样本维拼接
    for key in ("local_relative", "global_relative", "local_neighbor_mask", "query_coord", "query_index"):
        batch[key] = torch.cat([s[key] for s in samples])
    # 每个块一份的标量/向量数据堆叠成 [B, ...]
    for key in ("conditions", "origin", "rotation"):
        batch[key] = torch.stack([s[key] for s in samples])
    return batch


def collate_blocks(samples):
    """训练合批在同一几何接口上附加 cost 目标。"""
    batch = collate_queries(samples)
    batch["target"] = torch.cat([sample["target"] for sample in samples])
    batch["condition_bank"] = torch.tensor([s["condition_bank"] for s in samples])
    if "facility_mask" in samples[0]:
        batch["facility_mask"] = torch.cat([s["facility_mask"] for s in samples])
    return batch


def to_device(batch, device):
    """递归地把 batch 中的张量搬到目标设备，非张量字段原样保留。"""
    return {key:to_device(value,device) if isinstance(value,dict)
            else value.to(device,non_blocking=True) if isinstance(value,torch.Tensor)
            else value for key,value in batch.items()}
