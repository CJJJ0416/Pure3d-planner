"""Native VST arrays, raw LiDAR capture I/O and world-coordinate lookup."""
from pathlib import Path
import pickle

import numpy as np

from ptv3_tomography.io import file_sha256, read_json


def points_to_local(points_world, world_from_local):
    return (np.asarray(points_world) - world_from_local[:3, 3]) @ world_from_local[:3, :3]


def grid_xy(cfg):
    size = np.asarray(cfg["grid_size_xy"])
    axes = [(np.arange(n) - n // 2) * cfg["resolution_m"] for n in size]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def load_tomography(scene_dir):
    scene_dir = Path(scene_dir)
    scene = read_json(scene_dir / 'scene.json')
    meta = read_json(scene_dir / 'vst.meta.json')
    with (scene_dir / 'vst.pickle').open('rb') as stream:
        payload = pickle.load(stream)
    return payload, np.asarray(payload['data'], np.float32), scene, meta


def query_world_point_costs(data, payload, points_world, cfg):
    """逐点按世界 XY 查原生 VST，选择与回波 Z 最接近的已知地面层。

    同高重复层采用最小 cost。保留所有输入点，未知返回 NaN。
    此函数不使用局部标签、训练掩码或模型预测。
    """
    points = np.asarray(points_world)
    size = np.asarray(data.shape[2:])
    grid = np.rint((points[:, :2] - payload['center']) / payload['resolution']).astype(np.int64) + size // 2
    inside = ((grid >= 0) & (grid < size)).all(axis=1)
    x, y = np.clip(grid, 0, size - 1).T
    cost, ground = data[0][:, x, y], data[3][:, x, y]
    eligible = np.isfinite(ground) & np.isfinite(cost) & (cost >= 0) & (cost <= cfg['cost_scale'] + 1e-4)
    distance = np.where(eligible, np.abs(ground - points[None, :, 2]), np.inf)
    nearest = eligible & (distance <= distance.min(axis=0)[None] + cfg['duplicate_height_tolerance_m'])
    layer = np.argmin(np.where(nearest, cost, np.inf), axis=0)
    point_index = np.arange(len(points))
    available = inside & eligible.any(axis=0)
    error = distance[layer, point_index]
    return dict(cost_raw=np.where(available, cost[layer, point_index], np.nan),
                ground_world_m=np.where(available, ground[layer, point_index], np.nan),
                source_layer=np.where(available, layer, -1), source_grid_xy=grid,
                target_available=available,
                surface_height_error_m=np.where(available, error, np.nan))


def load_capture(capture_dir, scene_dir):
    capture_dir = Path(capture_dir)
    profile = read_json(capture_dir / 'profile.json')
    report = read_json(capture_dir / 'report.json')
    scans, records = [], []
    for row in report['frames']:
        path = capture_dir / f"scan_{row['index']:05d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            scan = {key: archive[key].copy() for key in archive.files}
        scans.append(scan)
        records.append(dict(index=row['index'], file=str(path.resolve()), sha256=file_sha256(path),
                            time_start_s=float(scan['time_start_s']), time_end_s=float(scan['time_end_s'])))
    return scans, records, profile, report
