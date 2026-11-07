#!/usr/bin/env python3
"""Collect route LiDAR scans with batched CUDA Warp mesh queries, without Kit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'isaac_training/src')]
from isaac_nav.runtime.io import atomic_json, file_sha256
from isaac_nav.sensors.mid360_pattern import (
    DEFAULT_CONFIG, Mid360Pattern, load_config, sensor_extrinsics, transform,
)
from isaac_nav.sensors.warp_mid360 import WarpMid360


def route_poses(route, config, speed, count=None):
    path = np.atleast_2d(np.loadtxt(route, delimiter=',', skiprows=1))
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-8]
    path, distance = path[keep], distance[keep]
    frames = math.ceil(distance[-1] / speed * config['scan_hz']) + 1 if count is None else count
    position = np.minimum(np.arange(frames) / config['scan_hz'] * speed, distance[-1])
    xyz = np.column_stack([np.interp(position, distance, path[:, k]) for k in range(3)])
    xyz[:, 2] += config['body_height_above_reference_m']
    yaw = np.interp(position, distance, np.unwrap(path[:, 3]))
    return np.column_stack((xyz, np.zeros((frames, 2)), yaw))


def prepare_jobs(args, config, pattern):
    if args.jobs:
        source = args.jobs.resolve()
        jobs = json.loads(source.read_text())['jobs']
        for job in jobs:
            for key in ('route', 'output'):
                path = Path(job[key])
                job[key] = path if path.is_absolute() else source.parent / path
    else:
        jobs = [dict(id=args.output.name, route=args.route, output=args.output,
                     frames=args.frames, speed=args.speed)]
    scene_assets = [dict(base='scene', path=name, sha256=file_sha256(args.scene / name))
                    for name in ('scene.json', 'terrain.obj', 'terrain.pcd', 'vst.pickle', 'vst.meta.json')]
    for job in jobs:
        job['route'], job['output'] = Path(job['route']).resolve(), Path(job['output']).resolve()
        job['speed'] = job.get('speed', args.speed)
        job['poses'] = route_poses(job['route'], config, job['speed'], job.get('frames', args.frames))
        job['rows'] = []
        output = job['output']
        (output / 'inputs').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(job['route'], output / 'inputs/route.csv')
        atomic_json(output / 'inputs/sensor_config.json', config)
        job['assets'] = scene_assets + [dict(base='capture', path='inputs/' + name,
            sha256=file_sha256(output / 'inputs' / name)) for name in ('route.csv', 'sensor_config.json')]
        profile = pattern.describe()
        profile.update(backend=config['backend'], capture_contract_version='portable_mid360_v2',
                       renderer_started=False, runtime_executed=True,
                       motion='kinematic_native_reference_replay_not_ground_truth_support',
                       acquisition='held pose per scan, deterministic MID-360 angular table')
        atomic_json(output / 'profile.json', profile)
    return jobs


def scans(jobs, pattern, config):
    extrinsics = sensor_extrinsics(config)
    for job in jobs:
        for index, pose in enumerate(job['poses']):
            ids, azimuth, elevation, directions = pattern.samples(index * config['rays_per_scan'], config['rays_per_scan'])
            body = transform(pose[:3], pose[3:])
            scan = dict(directions_sensor=directions, world_from_body=body,
                        body_from_sensor=extrinsics, world_from_sensor=body @ extrinsics,
                        azimuth_rad=azimuth, elevation_rad=elevation, source_indices=ids,
                        frame_index=np.array(index), pose_xyz_rpy=pose,
                        time_start_s=np.array(index / config['scan_hz']),
                        time_end_s=np.array((index + 1) / config['scan_hz']))
            yield job, index, scan


def save_scan(job, index, scan):
    started = time.perf_counter()
    path = job['output'] / f'scan_{index:05d}.npz'
    np.savez_compressed(path, **scan)
    saved = time.perf_counter()
    digest = file_sha256(path)
    return job, dict(index=index, rays=len(scan['ranges_m']), hits=int(scan['hit_mask'].sum()),
                     sha256=digest, compressed_save_s=saved-started, scan_hash_s=time.perf_counter()-saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--route', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=Path, help='Routes and output directories sharing one scene and sensor config')
    parser.add_argument('--frames', type=int, help='Omit to cover each complete route')
    parser.add_argument('--speed', type=float, default=.6)
    parser.add_argument('--rays-per-scan', type=int)
    parser.add_argument('--batch-frames', type=int, default=64)
    parser.add_argument('--writers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.scene = args.scene.resolve()
    started = time.perf_counter()
    config = load_config(args.config, args.rays_per_scan)
    config['backend'] = 'nvidia_warp.mesh_query_ray'
    pattern = Mid360Pattern(config)
    jobs = prepare_jobs(args, config, pattern)
    mesh = trimesh.load(args.scene / 'terrain.obj', force='mesh', process=False)
    engine = WarpMid360(mesh.vertices, mesh.faces, config['rays_per_scan'], device=args.device)
    stream = iter(scans(jobs, pattern, config))
    timing = dict(initialization_s=time.perf_counter()-started, query_and_cpu_return_s=0., batches=0)
    capture_started = time.perf_counter()
    pending = []
    with ThreadPoolExecutor(max_workers=args.writers) as writers:
        while batch := list(islice(stream, args.batch_frames)):
            tick = time.perf_counter()
            records = engine.read_batch([row[2] for row in batch], config['min_range_m'], config['max_range_m'])
            timing['query_and_cpu_return_s'] += time.perf_counter()-tick
            timing['batches'] += 1
            # Batch n queries overlap the writes from batch n-1.
            for future in pending:
                job, row = future.result()
                job['rows'].append(row)
            pending = [writers.submit(save_scan, job, index, record)
                       for (job, index, _), record in zip(batch, records)]
        for future in pending:
            job, row = future.result()
            job['rows'].append(row)
    wall = time.perf_counter()-capture_started
    total_frames = sum(len(job['rows']) for job in jobs)
    timing.update(scan_loop_s=wall, frames=total_frames, frames_per_s=total_frames/wall)
    import warp as wp
    for job in jobs:
        report = dict(status='passed', runtime_executed=True, capture_contract_version='portable_mid360_v2',
            backend=config['backend'], source_assets=job['assets'], source_hashes={},
            frames=job['rows'], speed_mps=job['speed'], warp_version=wp.__version__,
            gpu=wp.get_device(args.device).name, renderer_started=False, batch_frames=args.batch_frames,
            writers=args.writers, parallel_captures=len(jobs), capture_batch_wall_s=wall, timing=timing,
            imported_runtime_modules={name: name in sys.modules for name in ('isaacsim', 'isaaclab', 'omni', 'torch')},
            assumptions=dict(held_pose_per_scan=True, legged_dynamics=False, sensor_noise=False,
                             rolling_scan=False, robot_self_occlusion=False))
        atomic_json(job['output'] / 'report.json', report)
    print('WARP_CAPTURE_COMPLETE ' + json.dumps(timing), flush=True)


if __name__ == '__main__':
    main()
