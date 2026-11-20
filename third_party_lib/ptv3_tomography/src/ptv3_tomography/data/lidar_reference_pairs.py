"""Raw LiDAR-frame local maps supervised directly by available VST labels."""
from pathlib import Path

import numpy as np

from ptv3_tomography.io import asset_path, read_json
from ptv3_tomography.paths import PROJECT_ROOT

DEFAULT_LIDAR_CONFIG = PROJECT_ROOT / 'configs/tomography_pairing_lidar_v2.json'


def load_lidar_config(path=DEFAULT_LIDAR_CONFIG):
    return read_json(path)


def gravity_up_sensor(odom_from_sensor, up_odom=(0., 0., 1.)):
    up = np.asarray(up_odom, float)
    return np.asarray(odom_from_sensor)[:3, :3].T @ (up / np.linalg.norm(up))


class LidarLabelIndex:
    """Cache the native VST cells with known cost and ground height."""
    def __init__(self, data, payload, cost_scale):
        cost, ground = data[0], data[3]
        self.finite = (np.isfinite(ground) & np.isfinite(cost)
                       & (cost >= 0) & (cost <= cost_scale + 1e-4))


def add_teacher_masks(labels):
    """Both losses use every available local Tomography target."""
    labels['mask_cost'] = labels['target_available'].copy()
    labels['mask_height'] = labels['target_available'].copy()
    return labels


def _local_native_bounds(data, payload, rotation, translation, cfg, inverse_projection):
    """Conservative source XY box for all local grid centres and gravity bands.

    A matched source cell must contain a sensor-Z column/ground intersection.
    Its world XY is affine in sensor XY and gravity-relative height, so the
    eight corners bound every intersection. Expand for the native half-cell
    and floating-point rounding. Retain every native height layer in this box.
    """
    size = np.asarray(cfg["grid_size_xy"])
    low = -(size // 2) * cfg["resolution_m"]
    high = (size - 1 - size // 2) * cfg["resolution_m"]
    xy = np.array([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])])
    heights = np.asarray(cfg["height_bin_edges_m"])[[0, -1]]
    world_xy = ((xy[None] - heights[:, None, None] * rotation[2, :2])
                @ inverse_projection + translation[:2]).reshape(-1, 2)
    native = (world_xy - payload["center"]) / payload["resolution"] + np.asarray(data.shape[2:]) // 2
    start = np.floor(native.min(axis=0) - .5).astype(np.int64) - 1
    stop = np.ceil(native.max(axis=0) + .5).astype(np.int64) + 2
    start = np.clip(start, 0, data.shape[2:])
    stop = np.clip(stop, 0, data.shape[2:])
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))


def _direct_label_candidates(data, payload, rotation, translation, cfg, inverse_projection, source_index):
    """Map local XY/known-ground-height queries to native indices, then gather labels.

    Source heights already exist in Tomography. Their local XY bounding boxes
    bound the required queries. Four projected corners per distinct height
    replace projecting every source cell and expanding its neighbours.
    """
    sx, sy = _local_native_bounds(data, payload, rotation, translation, cfg, inverse_projection)
    if source_index is None:
        cost, ground = data[0, :, sx, sy], data[3, :, sx, sy]
        finite = (np.isfinite(ground) & np.isfinite(cost)
                  & (cost >= 0) & (cost <= cfg["cost_scale"] + 1e-4))
    else:
        finite = source_index.finite[:, sx, sy]
    source_layer, nx, ny = np.nonzero(finite)
    nx, ny = nx + sx.start, ny + sy.start
    ground = data[3, source_layer, nx, ny]
    height = ground.astype(float) - translation[2]
    edges = np.asarray(cfg["height_bin_edges_m"])
    keep = (height >= edges[0]) & (height < edges[-1])
    nx, ny, ground = nx[keep], ny[keep], ground[keep]
    pieces = [[] for _ in range(8)]
    if len(ground):
        levels, group = np.unique(ground, return_inverse=True)
        lo = np.full((len(levels), 2), data.shape[2:], np.int64)
        hi = np.full((len(levels), 2), -1, np.int64)
        for axis, values in enumerate((nx, ny)):
            np.minimum.at(lo[:, axis], group, values)
            np.maximum.at(hi[:, axis], group, values)
        size = np.asarray(cfg["grid_size_xy"])
        # Bound intermediate arrays even for scenes with many distinct heights.
        for start in range(0, len(levels), 32):
            g = levels[start:start + 32]
            lower, upper = lo[start:start + 32] - .5, hi[start:start + 32] + .5
            corners = np.stack((lower, np.column_stack((lower[:, 0], upper[:, 1])),
                                np.column_stack((upper[:, 0], lower[:, 1])), upper), axis=1)
            world_xy = (corners - np.asarray(data.shape[2:]) // 2) * payload["resolution"] + payload["center"]
            world = np.concatenate((world_xy, np.broadcast_to(g[:, None, None], (*world_xy.shape[:2], 1))), axis=2)
            sensor_xy = ((world - translation) @ rotation)[..., :2]
            qlo = np.floor(sensor_xy.min(axis=1) / cfg["resolution_m"]).astype(np.int64) + size // 2 - 1
            qhi = np.ceil(sensor_xy.max(axis=1) / cfg["resolution_m"]).astype(np.int64) + size // 2 + 2
            qlo, qhi = np.clip(qlo, 0, size), np.clip(qhi, 0, size)
            width = qhi - qlo
            count = width.prod(axis=1)
            groups = np.repeat(np.arange(len(g)), count)
            offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
            qx = qlo[groups, 0] + offset // width[groups, 1]
            qy = qlo[groups, 1] + offset % width[groups, 1]
            h = g[groups].astype(float) - translation[2]
            query_xy = (np.column_stack((qx, qy)) - size // 2) * cfg["resolution_m"]
            world_xy = (query_xy - h[:, None] * rotation[2, :2]) @ inverse_projection + translation[:2]
            native = np.rint((world_xy - payload["center"]) / payload["resolution"]).astype(np.int64) + np.asarray(data.shape[2:]) // 2
            inside = ((native >= 0) & (native < data.shape[2:])).all(axis=1)
            qx, qy, h, groups, native, world_xy = [a[inside] for a in (qx, qy, h, groups, native, world_xy)]
            x, y = native.T
            costs, grounds = data[0][:, x, y], data[3][:, x, y]
            eligible = ((grounds == g[groups][None]) & np.isfinite(costs)
                        & (costs >= 0) & (costs <= cfg["cost_scale"] + 1e-4))
            present = eligible.any(axis=0)
            # Exact same-height duplicates share one native XY. Canonical cost,
            # then lowest layer, matches the original global stable tie break.
            layer = np.argmin(np.where(eligible, costs, np.inf), axis=0)
            cost = costs[layer, np.arange(len(layer))]
            band = np.searchsorted(edges, h, side="right") - 1
            flat = (band * size[0] + qx) * size[1] + qy
            values = (layer, x, y, g[groups], cost, h, flat, world_xy)
            for bucket, value in zip(pieces, values):
                bucket.append(value[present])
    if not pieces[0]:
        integer = np.empty(0, np.int64)
        scalar = np.empty(0, np.float64)
        return integer, integer, integer, scalar, scalar, scalar, integer, np.empty((0, 2))
    return tuple(np.concatenate(bucket) for bucket in pieces)


def query_lidar_labels(data, payload, world_from_sensor, cfg, *, source_index=None):
    """将原生 VST 的水平单元投影到雷达 XY 格心，保留物理高度与源单元。

    对每个原生单元视作一片常高水平面，沿雷达 Z 方向求与雷达 XY
    格心柱的交点，再检验交点仍查询回相同源单元。完整使用雷达姿态，
    无 yaw-only 替代；高度带沿重力定义，不使用雷达 Z 作为物理高度。
    按局部 XY 与原图已知地面高度构建查询，转换到世界坐标后直接查表。
    """
    tf = np.asarray(world_from_sensor, float)
    rotation, translation = tf[:3, :3], tf[:3, 3]
    up = gravity_up_sensor(tf)
    size = np.asarray(cfg["grid_size_xy"])
    shape = (len(cfg["height_bin_edges_m"]) - 1, *size)
    total = int(np.prod(shape))
    labels = dict(cost_raw=np.full(shape, np.nan, np.float32),
                  ground_height_gravity_m=np.full(shape, np.nan, np.float32),
                  surface_points_sensor_m=np.full((*shape, 3), np.nan, np.float32),
                  surface_points_world_m=np.full((*shape, 3), np.nan, np.float64),
                  source_grid_xy=np.full((*shape, 2), -1, np.int32),
                  source_layer=np.full(shape, -1, np.int16), target_available=np.zeros(shape, bool),
                  ambiguous=np.zeros(shape, bool),
                  world_from_sensor=tf.copy(), gravity_up_sensor=up.astype(np.float32),
                  support_target=np.full(shape, -1, np.int8), mask_support=np.zeros(shape, bool))
    inverse_projection = np.linalg.inv(rotation[:2, :2])
    layer, nx, ny, ground, cost, height, flat, world_xy = _direct_label_candidates(
        data, payload, rotation, translation, cfg, inverse_projection, source_index)
    ids = np.arange(len(flat))
    if len(ids):
        h_min, h_max = np.full(total, np.inf), np.full(total, -np.inf)
        np.minimum.at(h_min, flat, height[ids])
        np.maximum.at(h_max, flat, height[ids])
        # Stable tie break: target cell, cost, source layer, native X/Y.
        order = np.lexsort((ny[ids], nx[ids], layer[ids], cost[ids], flat))
        sorted_flat = flat[order]
        first = order[np.r_[True, np.diff(sorted_flat) != 0]]
        selected_flat, selected_ids = flat[first], ids[first]
        ambiguous = (h_max[selected_flat] - h_min[selected_flat]) > cfg["duplicate_height_tolerance_m"]
        labels["ambiguous"].ravel()[selected_flat] = ambiguous
        first, selected_flat, selected_ids = [a[~ambiguous] for a in (first, selected_flat, selected_ids)]
        points_world = np.column_stack((world_xy[first], ground[selected_ids]))
        points_sensor = (points_world - translation) @ rotation
        labels["target_available"].ravel()[selected_flat] = True
        labels["cost_raw"].ravel()[selected_flat] = cost[selected_ids]
        labels["ground_height_gravity_m"].ravel()[selected_flat] = height[selected_ids]
        labels["surface_points_sensor_m"].reshape(-1, 3)[selected_flat] = points_sensor
        labels["surface_points_world_m"].reshape(-1, 3)[selected_flat] = points_world
        labels["source_grid_xy"].reshape(-1, 2)[selected_flat] = np.column_stack((nx[selected_ids], ny[selected_ids]))
        labels["source_layer"].ravel()[selected_flat] = layer[selected_ids]
    labels["cost_normalized"] = labels["cost_raw"] / cfg["cost_scale"]
    return add_teacher_masks(labels)



class LidarStepDataset:
    """Current-frame LiDAR, odometry and local Tomography labels."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.spec = read_json(self.directory / 'dataset_spec.json')
        self.manifest = read_json(self.directory / 'manifest.json')
        self.frames = self.manifest['frames']
        with np.load(self.directory / self.manifest['lidar_odometry']['file'], allow_pickle=False) as a:
            self.poses = a['odom_from_sensor'].copy()
            self.times = a['time_end_s'].copy()
            self.up_odom = a['gravity_up_odom'].copy()

    def __len__(self):
        return len(self.frames)

    def asset(self, value):
        return asset_path(self.directory, value)

    def __getitem__(self, index):
        frame = self.frames[index]
        with np.load(self.asset(frame['scan_file']), allow_pickle=False) as a:
            valid, points = a['hit_mask'].copy(), a['points_sensor_m'].copy()
        with np.load(self.directory / frame['label_file'], allow_pickle=False) as a:
            label = add_teacher_masks({k: a[k].copy() for k in a.files})
        start = index == 0 or frame['episode_id'] != self.frames[index-1]['episode_id']
        delta = np.eye(4) if start else np.linalg.inv(self.poses[index]) @ self.poses[index-1]
        dt = 0. if start else self.times[index] - self.times[index-1]
        inputs = dict(points_sensor_m=np.where(valid[:, None], points, 0.).astype(np.float32), point_mask=valid,
                      previous_sensor_in_current=delta.astype(np.float32), dt_s=np.array(dt, np.float32),
                      gravity_up_sensor=gravity_up_sensor(self.poses[index], self.up_odom).astype(np.float32),
                      episode_start=np.array(start))
        available = label['target_available']
        targets = {k: label[k] for k in ('mask_cost', 'mask_height', 'target_available')}
        targets['cost'] = np.where(available, label['cost_normalized'], 0.)
        targets['ground_height_gravity_m'] = np.where(available, label['ground_height_gravity_m'], 0.)
        return dict(inputs=inputs, targets=targets, metadata=dict(episode_id=frame['episode_id'], frame_index=index))
