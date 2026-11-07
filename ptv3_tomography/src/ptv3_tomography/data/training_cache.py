"""One-time preparation and memory-mapped training arrays for current-frame maps."""
# 官方来源：PointTransformerV3 @ 4e277e6 的 preprocess.prepare / collate。
# 本项目将静态预处理结果持久化为 CPU mmap，地图标签保持逐帧对齐。
# 来源与适配说明见 docs/OFFICIAL_TRAINING_COMPARISON.md。
from pathlib import Path
import time

import numpy as np
import torch
from point_transformer_v3 import collate, prepare
from torch.utils.data import Dataset

from ptv3_tomography.data.pretraining import FrameCollection, canonical_digest
from ptv3_tomography.io import atomic_json, read_json


def prepare_training_cache(registry, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # 掩码直接采用 VST 标签可用性，与输入点数预算无关。
    collection = FrameCollection(registry, config)
    metadata = {"version": "ptv3_training_cache_v2", "map_config": collection.map_config,
                "datasets": collection.identity, "registry": str(Path(registry).resolve()),
                "registry_digest": canonical_digest(read_json(registry)),
                "max_points": config["max_points"], "grid_size_xy": config["grid_size_xy"],
                "voxel_size_m": config["model"]["voxel_size_m"], "splits": {}}
    for split, references in collection.references.items():
        directory = output / split
        directory.mkdir()
        shape = (len(references), len(collection.map_config["height_bin_edges_m"]) - 1,
                 *config["grid_size_xy"])
        if not references:
            metadata["splits"][split] = {"frames": [], "tokens": 0}
            continue
        maps = {name: np.lib.format.open_memmap(directory / (name + ".npy"), mode="w+",
                 dtype=np.float32, shape=shape)
                for name in ("cost", "height")}
        coordinates, grids, ups, frames, offsets = [], [], [], [], [0]
        for index in range(len(references)):
            sample = collection.sample(split, index)
            # 直接调用官方 prepare，不自行改写体素化算法。
            # 项目选择 XYZ、0.1m、首点代表，并保留雷达原点以对齐固定地图查询。
            # 官方 ScanNet 配方使用 0.02m、随机代表点和中心平移，属于不同数据配方。
            point, _ = prepare(sample["points"], feat=("coord",), center=False, pick="first",
                               grid_size=metadata["voxel_size_m"], device="cpu")
            row = len(frames)
            for name, array in maps.items():
                array[row] = sample["targets"][name]
            coordinates.append(point.coord.numpy())
            # 非负且帧内唯一的 grid_coord 由官方 prepare 产生。
            grids.append(point.grid_coord.numpy().astype(np.int32))
            ups.append(sample["up"])
            offsets.append(offsets[-1] + len(point.coord))
            frames.append({"capture_id": sample["capture_id"], "frame_index": sample["frame_index"],
                           "points": len(sample["points"]), "tokens": len(point.coord)})
        for array in maps.values():
            array.flush()
        del maps
        if frames:
            np.save(directory / "coord.npy", np.concatenate(coordinates))
            np.save(directory / "grid_coord.npy", np.concatenate(grids))
            np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))
            np.save(directory / "up.npy", np.asarray(ups, dtype=np.float32))
        metadata["splits"][split] = {"frames": frames, "tokens": offsets[-1]}
        print(f"PTV3_CACHE split={split} frames={len(frames)} tokens={offsets[-1]}", flush=True)
    metadata["elapsed_s"] = time.perf_counter() - started
    atomic_json(output / "metadata.json", metadata)
    return metadata


class CachedFrames(Dataset):
    def __init__(self, directory, split):
        self.directory = Path(directory) / split
        metadata = read_json(Path(directory) / "metadata.json")
        self.frames = metadata["splits"][split]["frames"]
        self.grid_size = metadata["voxel_size_m"]
        self.arrays = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        # worker 在 CPU 上懒加载 mmap。文件映射与预取队列使用主机内存。
        if self.arrays is None:
            self.arrays = {name: np.load(self.directory / (name + ".npy"), mmap_mode="r")
                           for name in ("coord", "grid_coord", "offsets", "up", "cost", "height")}
        arrays = self.arrays
        start, end = arrays["offsets"][index:index + 2]
        coord = torch.from_numpy(arrays["coord"][start:end].copy())
        cost = torch.from_numpy(arrays['cost'][index].copy())
        height = torch.from_numpy(arrays['height'][index].copy())
        available = torch.isfinite(cost) & torch.isfinite(height)
        return {'point': {'coord': coord, 'feat': coord,
                          'grid_coord': torch.from_numpy(arrays['grid_coord'][start:end].copy()),
                          'offset': torch.tensor([end-start]), 'grid_size': self.grid_size},
                'up': torch.from_numpy(arrays['up'][index].copy()),
                'targets': dict(cost=cost, height=height, mask_cost=available, mask_height=available),
                'frame': self.frames[index]}


def collate_frames(samples):
    # 与官方 demo/3_batch.py 相同：拼接点，offset 保存各样本累计结束位置。
    # 例如 [280,300] 点变为 580 个 token 和 offset=[280,580]，骨干只前向一次。
    # 项目地图/up 按 B 维堆叠。保持帧边界，不执行 ScanNet 配方中的 Mix3D。
    return {"point": dict(collate([sample["point"] for sample in samples])),
            "up": torch.stack([sample["up"] for sample in samples]),
            "targets": {name: torch.stack([sample["targets"][name] for sample in samples])
                        for name in ("cost", "height", "mask_cost", "mask_height")},
            "max_tokens": max(sample["frame"]["tokens"] for sample in samples),
            "frames": [sample["frame"] for sample in samples]}
