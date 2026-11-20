"""Causal saved-LIO replay through Entry's actual local map and GPU worker."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from common.cost.client import WorkerClient
from common.io import atomic_json
from common.mapping.local_map import ContextVoxelMap, load_local_config


def replay(args):
    root = args.root.resolve()
    folder = root/args.profile
    output = args.output.resolve()
    (output/'frames').mkdir(parents=True, exist_ok=True)
    cfg = load_local_config(args.config, args.model)
    config_path = output/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_cfg = yaml.safe_load((root/'config.actual.json').read_text())
    increments = dict(np.load(folder/'incremental_index.npz'))
    coord = np.load(folder/'arrays_000/coord.npy', mmap_mode='r')
    odometry = np.load(folder/'lio_odometry.npy')
    transform = np.asarray(yaml.safe_load((folder/'map_from_lio.json').read_text())['matrix'])
    poses = odometry[:,1:4]@transform[:3,:3].T+transform[:3,3]
    rotations = transform[:3,:3]@Rotation.from_quat(odometry[:,4:8]).as_matrix()
    np.savez(output/'trajectory.npz', stamp_s=odometry[:,0]-source_cfg['time_origin_s'],
             position=poses, rotation=rotations)
    scan_cfg = yaml.safe_load(args.scan_config.read_text())['scan_planner_node']['ros__parameters']
    resolution = scan_cfg['grid_map.resolution']
    size = np.array([scan_cfg['grid_map.sliding_map_size_'+axis] for axis in 'xyz'])
    cells = np.ceil(size/resolution).astype(int)
    shift_cells = int(np.ceil(scan_cfg['grid_map.map_sliding_thresh']/resolution))
    local = ContextVoxelMap(cfg['voxel_m'], cfg['max_voxels'], cfg['core_size_m'],
                            cfg['context_m'], cfg['context_mode'])
    worker = WorkerClient(cfg, config_path, output)
    origin = source_cfg['time_origin_s']
    last_frame = int(np.searchsorted(increments['stamp_s']-origin,args.end_s+1e-6,side='right'))-1
    records = []
    started = time.perf_counter()
    manifest = dict(status='initializing', source_root=str(root), profile=args.profile,
        frame='map, metres, Z up', model=cfg['model'], condition=cfg['condition'],
        box_size_m=size.tolist(), box_center='aligned estimated LiDAR/IMU pose',
        core_size_m=cfg['core_size_m'], context_m=cfg['context_m'], voxel_m=cfg['voxel_m'],
        context_mode=cfg['context_mode'], update_period_s=cfg['period_s'], records=records,
        start_s=args.start_s,end_s=float(increments['stamp_s'][last_frame]-origin),
        inference='Entry.ContextVoxelMap + Entry.WorkerClient, only observations up to each frame',
        playback='ordered causal replay, inference blocks source-time advancement; not a live ROS latency test')
    atomic_json(output/'index.json', manifest)
    centre_cell = None
    next_due = 0.
    ingest_s = 0.
    try:
        worker.initialize()
        for i, stamp in enumerate(increments['stamp_s'][:last_frame+1]):
            tick = time.perf_counter()
            j = int(np.argmin(abs(odometry[:,0]-stamp)))
            position = poses[j]
            candidate = np.floor(position/resolution).astype(int)
            if centre_cell is None or np.max(abs(candidate-centre_cell)) >= shift_cells:
                centre_cell = candidate
                lower = (centre_cell-cells//2)*resolution
                upper = lower+cells*resolution
                local.set_bounds(lower, upper)
            first, last = increments['offsets'][i:i+2]
            local.add(coord[first:last], float(stamp))
            ingest_s += time.perf_counter()-tick
            elapsed = float(stamp-origin)
            if elapsed+1e-6 < args.start_s:
                continue
            if elapsed+1e-6 < next_due and i != last_frame:
                continue
            next_due = elapsed+cfg['period_s']
            tick = time.perf_counter()
            snapshot = local.snapshot()
            valid = bool(len(snapshot.coord) and stamp-snapshot.stamp <= cfg['max_input_age_s'])
            if valid:
                _, cost, report = worker.predict(snapshot)
                xyz = snapshot.coord
            else:
                xyz, cost = np.empty((0,3),np.float32), np.empty(0,np.float32)
                report = dict(encoder_calls=0,cache_hits=0,timing=dict(total_s=0.))
            wall = time.perf_counter()-tick
            name = f'frames/{len(records):05d}.npz'
            np.savez(output/name, xyz=xyz, cost=cost, position=position, rotation=rotations[j],
                lower=lower, upper=upper, context_lower=local.native.lower,
                context_upper=local.native.upper, stamp_s=elapsed,
                observation_stamp_s=float(snapshot.stamp-origin) if snapshot.stamp is not None else np.nan,
                scan_count=i+1, raw_point_count=int(last))
            row = dict(path=name,stamp_s=elapsed,scan_count=i+1,raw_point_count=int(last),
                points=len(xyz),context_points=len(snapshot.context_coord),valid=valid,
                snapshot_to_cost_s=wall,encoder_calls=report['encoder_calls'],
                cache_hits=report['cache_hits'],cost_timing=report['timing'])
            records.append(row)
            manifest.update(status='running', source_frames_processed=i+1,
                source_frames_total=last_frame+1, wall_s=time.perf_counter()-started)
            atomic_json(output/'index.json', manifest)
            if len(records)%10==0 or len(records)==1:
                print('LOCAL_COST',len(records),f'{elapsed:.1f}s',len(xyz),'points',
                      f'{wall*1000:.1f}ms',flush=True)
        times = np.array([row['snapshot_to_cost_s'] for row in records])
        manifest.update(status='complete',wall_s=time.perf_counter()-started,ingest_s=ingest_s,
            cost_updates=len(records),snapshot_to_cost_median_s=float(np.median(times)),
            snapshot_to_cost_p95_s=float(np.percentile(times,95)),
            source_duration_s=float(increments['stamp_s'][-1]-origin),
            encoder_calls=sum(row['encoder_calls'] for row in records),
            cache_hits=sum(row['cache_hits'] for row in records))
        atomic_json(output/'index.json',manifest)
        print('COST_REPLAY_COMPLETE',manifest['wall_s'],flush=True)
    finally:
        worker.close()
        local.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', default='mixed')
    parser.add_argument('--config', type=Path, default=Path('Entrance/config/local_cost.yaml'))
    parser.add_argument('--scan-config', type=Path,
        default=Path('Entrance/config/scan.yaml'))
    parser.add_argument('--model', choices=['small','tiny'])
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--end-s', type=float, default=float('inf'))
    parser.add_argument('--output', type=Path, required=True)
    replay(parser.parse_args())


if __name__ == '__main__':
    main()
