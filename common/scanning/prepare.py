"""Fixed-frame alignment, mesh supervision, GT-PCD distances and core indices."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from common.pointcloud_io import read_pcd_xyz, write_pcd
from Filter.data.boxes import export_boxes, collect_boxes
from Filter.data.groups import loss_groups
from utils.pointcloud import read_json, write_json


def describe(values):
    return dict(count=len(values), mean_m=float(np.mean(values)), rmse_m=float(np.sqrt(np.mean(values ** 2))),
        median_m=float(np.median(values)), p95_m=float(np.quantile(values, .95)),
        max_m=float(np.max(values)), within_1cm=float(np.mean(values <= .01)),
        within_3cm=float(np.mean(values <= .03)), within_5cm=float(np.mean(values <= .05)),
        within_10cm=float(np.mean(values <= .1)))


def truth_pose(truth, time):
    position = np.array([np.interp(time, truth['time'], truth['position'][:, axis]) for axis in range(3)]).T
    rotation = Slerp(truth['time'], Rotation.from_quat(truth['quaternion']))(time).as_matrix()
    return position, rotation


def prepare(root, profile):
    started = perf_counter()
    config = read_json(root / 'config.actual.json')
    folder = root / profile
    records = read_json(folder / 'registered.json')
    truth = dict(np.load(root / 'truth_trajectory.npz'))
    odometry = np.load(folder / 'lio_odometry.npy')
    origin = config['time_origin_s']
    first = records[0]['stamp_s']
    j = int(np.argmin(abs(odometry[:, 0] - first)))
    assert abs(odometry[j, 0] - first) < 1e-6
    true_p, true_r = truth_pose(truth, first-origin)
    estimated_r = Rotation.from_quat(odometry[j, 4:8]).as_matrix()
    align_r = true_r @ estimated_r.T
    align_p = true_p - align_r @ odometry[j, 1:4]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = align_r, align_p
    write_json(folder / 'map_from_lio.json', dict(matrix=transform.tolist(), anchor_stamp_s=first,
        method='one rigid transform from first registered IMU pose; no per-frame correction or ICP'))
    chunks = {key: [] for key in ['coord', 'truth_coord', 'frame_index', 'ray_id', 'offset_ns',
                                 'face_id', 'instance_id', 'semantic_id', 'source_role', 'edge_mixed']}
    offsets = [0]
    for row in records:
        frame = row['frame']
        registered = dict(np.load(folder / 'registered' / f'{frame:05d}.npz'))
        raw = dict(np.load(folder / 'scans' / f'{frame:05d}.npz'))
        np.testing.assert_array_equal(registered['ray_id'], raw['ray_id'])
        np.testing.assert_array_equal(registered['offset_ns'], raw['offset_ns'])
        count = len(raw['coord'])
        role = np.zeros(count, np.uint8)
        role[raw['semantic_id'] >= 8] = 5
        role[raw['edge_mixed']] = 2
        role[raw['dynamic_instance'] > 0] = 7
        values = dict(coord=(registered['coord'] @ align_r.T + align_p).astype(np.float32),
            truth_coord=raw['truth_coord'], frame_index=np.full(count, frame, np.int32), source_role=role,
            **{key: raw[key] for key in ['ray_id', 'offset_ns', 'face_id', 'instance_id', 'semantic_id', 'edge_mixed']})
        for key in chunks:
            chunks[key].append(values[key])
        offsets.append(offsets[-1] + count)
    attributes = {key: np.concatenate(value) for key, value in chunks.items()}
    del chunks
    xyz = attributes['coord']
    merge_s = perf_counter() - started
    teacher_started = perf_counter()
    mesh = trimesh.load(root / 'static_mesh.ply', process=False)
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                         o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    mesh_distance, truth_distance = [], []
    for a in range(0, len(xyz), 200000):
        mesh_distance.append(scene.compute_distance(o3d.core.Tensor(xyz[a:a+200000]), nthreads=8).numpy())
        truth_distance.append(scene.compute_distance(o3d.core.Tensor(attributes['truth_coord'][a:a+200000]), nthreads=8).numpy())
    mesh_distance = np.concatenate(mesh_distance)
    truth_distance = np.concatenate(truth_distance)
    target = np.exp(-.5 * (mesh_distance / config['target_sigma_m']) ** 2)
    target[attributes['source_role'] == 7] = 0.
    attributes.update(keep_target=target, surface_distance_m=mesh_distance,
                      is_disconnected_control=np.zeros(len(xyz), bool))
    attributes['loss_group'] = loss_groups(attributes)
    teacher_s = perf_counter() - teacher_started
    comparison_started = perf_counter()
    clean = read_pcd_xyz(root / 'clean.pcd')
    pcd_distance, nearest = cKDTree(clean).query(xyz, workers=8)
    # Reverse direction describes missing coverage, not bad observed points.
    reverse_distance, _ = cKDTree(attributes['truth_coord']).query(clean, workers=8)
    attributes['gt_pcd_distance_m'] = pcd_distance.astype(np.float32)
    attributes['gt_pcd_index'] = nearest.astype(np.int32)
    comparison_s = perf_counter() - comparison_started
    save_started = perf_counter()
    write_pcd(folder / 'rough.pcd', xyz)
    write_pcd(folder / 'truth_fused.pcd', attributes['truth_coord'])
    write_pcd(folder / 'gt_scored.pcd', xyz, target, field_name='keep_target')
    np.savez(folder / 'attributes.npz', **attributes)
    np.savez(folder / 'incremental_index.npz', frame_index=np.array([r['frame'] for r in records]),
             offsets=np.asarray(offsets, np.int64), stamp_s=np.array([r['stamp_s'] for r in records]))
    arrays = folder / 'arrays_000'
    arrays.mkdir(exist_ok=True)
    for key in ['coord', 'keep_target', 'source_role', 'semantic_id', 'loss_group']:
        np.save(arrays / f'{key}.npy', attributes[key])
    for name, source in [('rough_000.pcd', 'rough.pcd'), ('rough_attributes_000.npz', 'attributes.npz'),
                         ('clean.pcd', '../clean.pcd'), ('clean_attributes.npz', '../clean_attributes.npz'),
                         ('clean.obj', '../static_mesh.ply')]:
        path = folder / name
        if not path.is_symlink():
            path.symlink_to(source)
    save_s = perf_counter() - save_started
    block_started = perf_counter()
    metadata = read_json(root / 'scene.json')
    metadata.update(scene_id=f'{root.name}_{profile}', geometry_group_id=f"fasterlio_{config['seed']}",
                    split='example', seed=config['seed'])
    write_json(folder / 'scene.json', metadata)
    window = dict(core_size_m=config['core_size_m'], context_margin_m=config['context_margin_m'],
                  grid_shift_m=[0, 0, 0], export_individual_pcd_boxes=False)
    box = export_boxes(folder, 0, attributes, clean, {}, window, config['seed'])
    collect_boxes(folder)
    block_s = perf_counter() - block_started
    index = np.load(folder / 'windows_000_index.npy')
    core = np.load(folder / 'windows_000_core.npy')
    np.testing.assert_array_equal(np.sort(index[core]), np.arange(len(xyz)))
    gt_p, gt_r = truth_pose(truth, odometry[:, 0] - origin)
    estimated_p = odometry[:, 1:4] @ align_r.T + align_p
    error = np.linalg.norm(estimated_p - gt_p, axis=1)
    rotation_error = Rotation.from_matrix(np.einsum('nij,njk->nik', np.swapaxes(gt_r, 1, 2),
        align_r @ Rotation.from_quat(odometry[:, 4:8]).as_matrix())).magnitude()
    static = attributes['source_role'] != 7
    mapping_displacement = np.linalg.norm(xyz-attributes['truth_coord'], axis=1)
    report = dict(profile=profile, input_frames=len(read_json(folder / 'scans.json')),
        registered_frames=len(records), points=len(xyz), boxes=len(box['windows']),
        static_points=int(static.sum()), dynamic_points=int((~static).sum()),
        mixed_return_points=int(attributes['edge_mixed'].sum()),
        scan_to_mesh=describe(mesh_distance[static]), scan_to_gt_pcd=describe(pcd_distance[static]),
        truth_fused_to_mesh=describe(truth_distance[static]),
        gt_pcd_to_truth_fused=describe(reverse_distance),
        paired_same_ray_lio_displacement=describe(mapping_displacement),
        gt_sampling_excess_mean_m=float(np.mean(pcd_distance-mesh_distance)),
        keep_target=dict(mean=float(target.mean()), low_02=float(np.mean(target < .2)),
                         high_08=float(np.mean(target >= .8))),
        trajectory=dict(position_rmse_m=float(np.sqrt(np.mean(error**2))),
            position_max_m=float(error.max()), end_position_error_m=float(error[-1]),
            orientation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(rotation_error**2))))),
        alignment=transform.tolist(), direct_checks=dict(point_order=True, unique_core_ownership=True,
            raw_and_registered_ray_ids=True, no_icp=True),
        timings_s=dict(merge_alignment=merge_s, mesh_labels=teacher_s, gt_pcd_comparison=comparison_s,
                       save=save_s, boxes=block_s, total=perf_counter()-started))
    log = (folder / 'faster_lio.log').read_text(errors='replace')
    replay = read_json(folder / 'replay.json')
    first_frame = records[0]['frame']
    replay['skipped_frame_reasons'] = {str(frame): ('IMU initialization; native log: no point skip' if frame == 0
        else 'initial map seed; native Run returns before publishing' if frame < first_frame
        else 'processing skip, see native log') for frame in replay['missing_frames']}
    replay['native_skip_messages'] = [line for line in log.splitlines() if 'skip' in line.lower()]
    write_json(folder / 'replay.json', replay)
    write_json(folder / 'metrics.json', report)
    print('PREPARED', profile, 'points', len(xyz), 'mesh MAE m', report['scan_to_mesh']['mean_m'],
          'GT PCD MAE m', report['scan_to_gt_pcd']['mean_m'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.profile)


if __name__ == '__main__':
    main()
