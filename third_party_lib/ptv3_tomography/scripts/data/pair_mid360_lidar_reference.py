#!/usr/bin/env python3
"""Pair current-frame LiDAR with the available native VST cost/height targets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from ptv3_tomography.data.lidar_reference_pairs import (
    DEFAULT_LIDAR_CONFIG, LidarLabelIndex, load_lidar_config, query_lidar_labels,
)
from ptv3_tomography.data.tomography_pairs import load_capture, load_tomography
from ptv3_tomography.io import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_LIDAR_CONFIG)
    args = parser.parse_args()
    scene_path, capture, output = args.scene.resolve(), args.capture.resolve(), args.output.resolve()
    started = time.perf_counter()
    cfg = load_lidar_config(args.config)
    payload, data, scene, meta = load_tomography(scene_path)
    scans, records, profile, raw_report = load_capture(capture, scene_path)
    index = LidarLabelIndex(data, payload, cfg['cost_scale'])
    (output / 'labels').mkdir(parents=True, exist_ok=True)
    poses = np.stack([s['world_from_sensor'] for s in scans])
    times = np.array([float(s['time_end_s']) for s in scans])
    np.savez_compressed(output / 'lidar_odometry.npz', odom_from_sensor=poses, time_end_s=times,
                        gravity_up_odom=np.array([0., 0., 1.]), world_from_odom=np.eye(4))
    episode = capture.name + '_' + file_sha256(capture / 'report.json')[:12]
    frames, details = [], []
    timing = dict(source_setup_s=time.perf_counter()-started, query_s=0., compressed_save_s=0., label_hash_s=0.)
    for i, scan in enumerate(scans):
        tick = time.perf_counter()
        label = query_lidar_labels(data, payload, poses[i], cfg, source_index=index)
        timing['query_s'] += time.perf_counter()-tick
        label.update(frame_index=np.array(i), time_start_s=scan['time_start_s'], time_end_s=scan['time_end_s'])
        name = f'labels/frame_{i:05d}.npz'
        tick = time.perf_counter()
        np.savez_compressed(output / name, **label)
        timing['compressed_save_s'] += time.perf_counter()-tick
        tick = time.perf_counter()
        digest = file_sha256(output / name)
        timing['label_hash_s'] += time.perf_counter()-tick
        frames.append(dict(index=i, episode_id=episode, scan_file=records[i]['file'],
                           scan_sha256=records[i]['sha256'], label_file=name, label_sha256=digest))
        available = label['target_available']
        details.append(dict(frame_index=i, target_available=int(available.sum()),
                            ambiguous=int(label['ambiguous'].sum()), mask_cost=int(label['mask_cost'].sum()),
                            mask_height=int(label['mask_height'].sum()),
                            supervised_cost_ge50=int((available & (label['cost_raw'] >= 50)).sum()),
                            supervised_cost60=int((available & (label['cost_raw'] == 60)).sum())))
    paths = [scene_path / name for name in ('scene.json', 'terrain.obj', 'terrain.pcd', 'vst.pickle', 'vst.meta.json')]
    paths += [capture / 'profile.json', capture / 'report.json']
    spec = dict(version='mid360_lidar_reference_v2', created_utc=datetime.now(timezone.utc).isoformat(),
        config=cfg, role='single_frame_native_vst_supervision', scene_directory=str(scene_path),
        capture_directory=str(capture), episode_id=episode,
        source_fingerprints={str(p): file_sha256(p) for p in paths},
        coordinate_contract=dict(units='metres, radians, seconds',
            points_and_targets_frame='current raw LiDAR frame; full rotation and translation',
            odometry='odom_from_sensor is the LiDAR pose itself',
            grid_axes='[gravity_height_band, sensor_x, sensor_y]',
            physical_height='up_sensor dot ground_sensor = ground_world_z - lidar_world_z',
            resampling='sensor-XY column intersection with native horizontal ground cells',
            duplicate_policy='equal ground heights use minimum cost; distinct heights in one band remain unavailable'),
        supervision=dict(cost='mask_cost = target_available', height='mask_height = target_available',
            policy='native_vst_available_only', source='vst.pickle data[0] cost and data[3] ground',
            support='disabled', unknown='no native target; excluded from loss'),
        actor_input_boundary='raw current LiDAR and gravity reference; teacher data is loss-only',
        tomography_source_commit=meta['source']['commit'])
    manifest = dict(version=spec['version'], frames=frames,
                    lidar_odometry=dict(file='lidar_odometry.npz', sha256=file_sha256(output / 'lidar_odometry.npz')))
    atomic_json(output / 'dataset_spec.json', spec)
    atomic_json(output / 'manifest.json', manifest)
    totals = {k: sum(row[k] for row in details) for k in details[0] if k != 'frame_index'}
    elapsed = time.perf_counter()-started
    timing.update(total_s=elapsed, frames_per_s=len(frames)/elapsed)
    report = dict(status='passed', stage='pairing', frames=len(frames), timing=timing,
                  totals=totals, frames_detail=details, supervision_policy='native_vst_available_only')
    atomic_json(output / 'pairing_report.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'frames', 'timing', 'totals')}, indent=2))


if __name__ == '__main__':
    main()
