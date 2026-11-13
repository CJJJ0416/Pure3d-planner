"""建筑 PCD -> 逐点代价推理的无标签三维分块。输出位置就是输入 PCD 的 XYZ。"""
import argparse
from pathlib import Path
import time

import numpy as np

from Traversability.data.blocks import query_blocks
from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import read_json, voxel_downsample, write_json


def prepare(pcd, output, settings):
    """用全部建筑点作为推理位置，按原 PCD 行号分块；不依赖 Tomogram。"""
    from scipy.spatial import cKDTree

    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    phase_started = time.perf_counter()
    inputs = read_pcd_xyz(pcd)
    timings = dict(pcd_read_s=time.perf_counter() - phase_started)
    coord = inputs
    phase_started = time.perf_counter()
    global_coord = voxel_downsample(inputs, settings["global_voxel_m"])
    _, global_neighbors = cKDTree(global_coord).query(coord, k=settings["query_neighbors"]["global_k"])
    timings['global_geometry_and_neighbors_s'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    arrays = dict(input_coord=inputs, global_coord=global_coord,
                  global_neighbors=global_neighbors)
    for name, value in arrays.items():
        np.save(output / f"{name}.npy", value)
    timings['global_arrays_save_s'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    for old in output.glob("block_*.npz"):
        old.unlink()
    rows = []
    for index, (ii, core, qi, lower, upper) in enumerate(query_blocks(
        inputs, coord, settings["core_size_m"], settings["context_m"],
    )):
        _, neighbors = cKDTree(inputs[ii]).query(coord[qi], k=settings["query_neighbors"]["local_k"])
        # 真实扫描的远端块可能不足 K 个点。无邻居位置只作填充，不参与 Attention。
        neighbor_mask = neighbors < len(ii)
        neighbors = np.where(neighbor_mask, neighbors, 0)
        name = f"block_{index:05d}"
        np.savez(output / f"{name}.npz", input_indices=ii, query_indices=qi,
                 query_neighbors=neighbors, query_neighbor_mask=neighbor_mask)
        rows.append(dict(id=name, scene="inference", file=f"{name}.npz",
                         core_lower_m=lower.tolist(), core_upper_m=upper.tolist(),
                         input_points=len(ii), query_points=len(qi)))
    scene = dict(id="inference", arrays={name:f"{name}.npy" for name in arrays},
                 input_points=len(inputs), query_points=len(coord), global_points=len(global_coord))
    # 输入几何和输出查询共享同一份世界系 XYZ，不复制坐标，也不生成新位置。
    scene["arrays"]["query_coord"] = scene["arrays"]["input_coord"]
    timings['local_blocks_neighbors_and_save_s'] = time.perf_counter() - phase_started
    report = dict(version="building_point_cost_inference_v1", source_pcd=str(Path(pcd).resolve()),
                  query_source="input_pcd_xyz", blocks_config=settings, scenes=[scene], blocks=rows,
                  preparation_s=time.perf_counter() - started,
                  timings=timings,
                  coordinate_frame="map", map_origin_preserved=True,
                  query_coordinates="original input PCD XYZ in metres, original row order",
                  teacher_xyz_used=False, teacher_cost_used=False, slices_computed=False)
    write_json(output / "blocks.json", report)
    print(f"XYZ preparation: {len(inputs):,} geometry points, {len(coord):,} queries, "
          f"{len(rows)} 3D blocks, {report['preparation_s']:.2f}s", flush=True)
    return report


class QueryBlocks:
    """逐个建筑原始点查询代价，不读取任何教师数据。"""
    def __init__(self, root, scene, model_config, seed, condition):
        self.root = Path(root)
        metadata = read_json(self.root / "blocks.json")
        self.scene = next(row for row in metadata["scenes"] if row["id"] == scene)
        self.rows = [row for row in metadata["blocks"] if row["scene"] == scene]
        self.model_config, self.seed = model_config, seed
        self.condition = np.asarray(condition, dtype=np.float32)[None]
        self.arrays = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from Traversability.data.dataset import prepare_geometry_sample

        if self.arrays is None:
            self.arrays = {name:np.load(self.root / self.scene["arrays"][name], mmap_mode="r")
                           for name in ("input_coord", "query_coord", "global_coord", "global_neighbors")}
        row = self.rows[index]
        with np.load(self.root / row["file"]) as archive:
            ii, qi, neighbors = archive["input_indices"], archive["query_indices"], archive["query_neighbors"]
            neighbor_mask = archive["query_neighbor_mask"]
        origin = (np.asarray(row["core_lower_m"], dtype=np.float32)
                  + np.asarray(row["core_upper_m"], dtype=np.float32)) / 2
        sample = dict(input_coord=self.arrays["input_coord"][ii] - origin,
                      query_coord=self.arrays["query_coord"][qi] - origin,
                      global_coord=self.arrays["global_coord"][:] - origin,
                      query_neighbors=neighbors, global_neighbors=self.arrays["global_neighbors"][qi],
                      query_neighbor_mask=neighbor_mask,
                      query_index=qi, origin=origin, scene=self.scene["id"], block=row["id"])
        result = prepare_geometry_sample(sample, self.model_config, self.seed + index)
        result["conditions"] = torch.from_numpy(self.condition)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="包含 core_size_m/context_m 等的几何分块配置")
    args = parser.parse_args()
    prepare(args.pcd, args.output, read_json(args.config))


if __name__ == "__main__":
    main()
