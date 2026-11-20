"""Random whole-map endpoints, using the retained native VST A* and GPMP.

Run with the Python ABI matching the native extension. No ROS or CUDA is
required for this planning stage. Map indexing follows VST [layer, y, x].
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np


def plan(config, output):
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(config['native_dir']).resolve()))
    import a_star
    import ele_planner
    import traj_opt
    started = perf_counter()
    with Path(config['vst']).open('rb') as stream:
        payload = pickle.load(stream)
    cost, gx, gy, ground, ceiling = np.asarray(payload['data'], np.float64).copy()
    valid = np.isfinite(ground)
    cost[~valid] = config['cost_barrier']
    ceiling = np.nan_to_num(ceiling, nan=1e6)
    shared = valid[1:] & valid[:-1] & (np.abs(np.diff(ground, axis=0)) < .1)
    gateway = np.zeros_like(cost, np.int32)
    delta = np.diff(cost, axis=0)
    gateway[:-1][(delta < -8) & shared] = 2
    gateway[1:][(delta > 8) & shared] = -2
    # Same initialization and gradient-axis convention as VST's native wrapper.
    flat = lambda value: np.ascontiguousarray(value.reshape(-1, value.shape[-1]), dtype=np.float64)
    planner = ele_planner.OfflineElePlanner(max_heading_rate=10., use_quintic=True)
    planner.init_map(config['cost_threshold'], 15., payload['resolution'], cost.shape[0], 1.,
        flat(cost), flat(ground), flat(ceiling), flat(gateway), flat(gy), flat(-gx))
    cells = np.argwhere(valid & np.isfinite(cost) & (cost >= 0) & (cost <= config['cost_threshold']))
    xyz = np.column_stack(((cells[:,1:]-np.array(cost.shape[1:])//2)*payload['resolution']+payload['center'], ground[tuple(cells.T)]))
    # A support may be represented in several slices. Sample its XYZ only once.
    _, unique = np.unique(np.round(xyz, 4), axis=0, return_index=True)
    xyz, cells = xyz[unique], cells[unique]
    rng = np.random.default_rng(config['route_seed'])
    records, history = [], []

    def save_report():
        report = dict(vst=str(Path(config['vst']).resolve()), route_seed=config['route_seed'],
            backend='native VST A* + quintic GPMP', routes=records,
            requested=config['route_count'], successful=len(records),
            native_dir=str(Path(config['native_dir']).resolve()), wall_s=perf_counter()-started,
            automatic_retries=True, retry_policy='sample new start and goal, retain cost threshold',
            attempt_count=len(history), failed_attempt_count=sum(not r['success'] for r in history),
            attempt_history=history)
        (output/'routes.json').write_text(json.dumps(report,indent=2)+'\n')
        return report

    for index in range(config['route_count']):
        attempt = 0
        while True:
            attempt += 1
            first = int(rng.integers(len(xyz)))
            distance = np.linalg.norm(xyz-xyz[first], axis=1)
            candidates = np.flatnonzero((distance >= config['endpoint_distance_m'][0]) &
                (distance <= config['endpoint_distance_m'][1]))
            last = int(rng.choice(candidates))
            start, goal = cells[first][[0,2,1]].astype(np.int32), cells[last][[0,2,1]].astype(np.int32)
            row = dict(id=f'route_{index:03d}', start_support_xyz=xyz[first].tolist(),
                goal_support_xyz=xyz[last].tolist(), start_index=start.tolist(), goal_index=goal.tolist(),
                endpoint_sampling='uniform full-map unique traversable supports, distance-conditioned goal',
                cost_threshold=config['cost_threshold'], success=False)
            if not planner.plan(start, goal, True):
                path = np.asarray(planner.get_path_finder().get_result_matrix())
                row['failure'] = 'A* path not found' if not len(path) else 'GPMP optimization failed'
            else:
                optimizer = planner.get_trajectory_optimizer_wnoj()
                raw = np.asarray(optimizer.get_result_matrix())
                height = np.asarray(optimizer.get_heights()).reshape(-1)
                path = np.column_stack(((raw[:,raw.shape[1]//2]-cost.shape[1]//2)*payload['resolution']+payload['center'][0],
                    (raw[:,0]-cost.shape[2]//2)*payload['resolution']+payload['center'][1], height))
                # Preserve the GPMP-selected floor. Convert its +0.1 m reference
                # to support heights by reading the nearest ground at each XYZ.
                xy = np.rint((path[:,:2]-payload['center'])/payload['resolution']).astype(int)+np.array(cost.shape[1:])//2
                heights = ground[:,xy[:,0],xy[:,1]]
                error = np.where(np.isfinite(heights), np.abs(heights-path[:,2]), np.inf)
                layer = error.argmin(axis=0)
                support = np.column_stack((path[:,:2], heights[layer,np.arange(len(path))]))
                np.save(output/f"{row['id']}_gpmp.npy", path)
                np.save(output/f"{row['id']}_support.npy", support)
                np.savetxt(output/f"{row['id']}_gpmp.csv",path,delimiter=',',header='x,y,z',comments='')
                row.update(success=True, gpmp=str(output/f"{row['id']}_gpmp.npy"),
                    support_path=str(output/f"{row['id']}_support.npy"), points=len(path),
                    length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                    height_range_m=[float(support[:,2].min()),float(support[:,2].max())])
            row['attempt'] = attempt
            history.append(row.copy())
            if row['success']:
                row['failed_attempts'] = [r for r in history if r['id'] == row['id'] and not r['success']]
                records.append(row)
            save_report()
            if row['success']:
                break
    report = save_report()
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    for key in ['vst','native_dir']:
        config[key]=str(Path(config[key]).resolve())
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    os.chdir(output)  # Native optimizer logs belong to this route experiment.
    plan(config,output)


if __name__=='__main__':main()
