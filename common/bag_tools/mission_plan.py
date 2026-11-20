"""Native VST A*/GPMP worker (Python 3.10 ABI); writes continuous route artifacts."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from common.io import atomic_json, atomic_npy
from common.bag_tools.mission_core import sha256, check_support_path


def plan(scene, spawn_path, native, output, start_base, seed=821111, count=4, goal_index=None,
         distance_range=(6.,16.),max_path=None,heading=None,cone_deg=70.,visited=None):
    if count < 1 or (goal_index is not None and count != 1):
        raise ValueError('Expected positive segment count; a fixed goal requires exactly one segment')
    output.mkdir(parents=True, exist_ok=False)
    (output/'native_logs').mkdir()
    os.chdir(output/'native_logs')
    sys.path.insert(0, str(native))
    import a_star  # Load the existing native dependencies before OfflineElePlanner.
    import traj_opt
    import ele_planner
    begin = time.perf_counter()
    vst = scene/'vst_nominal.pickle'
    with vst.open('rb') as stream:
        payload = pickle.load(stream)
    spawn = json.loads(spawn_path.read_text())
    cost, gx, gy, ground, ceiling = np.asarray(payload['data'], np.float64).copy()
    floor_z = float(spawn['ground_z'])
    allowed = np.isfinite(ground) & (np.abs(ground-floor_z) <= .15)
    cost[~allowed] = 50.
    ceiling = np.nan_to_num(ceiling, nan=1e6)
    shared = allowed[1:] & allowed[:-1] & (np.abs(np.diff(ground, axis=0)) < .1)
    gateway = np.zeros_like(cost, np.int32)
    delta = np.diff(cost, axis=0)
    gateway[:-1][(delta < -8) & shared] = 2
    gateway[1:][(delta > 8) & shared] = -2
    flat = lambda value: np.ascontiguousarray(value.reshape(-1, value.shape[-1]), dtype=np.float64)
    planner = ele_planner.OfflineElePlanner(max_heading_rate=10., use_quintic=True)
    planner.init_map(15., 15., payload['resolution'], cost.shape[0], 1., flat(cost), flat(ground),
                     flat(ceiling), flat(gateway), flat(gy), flat(-gx))
    cells = np.argwhere(allowed & np.isfinite(cost) & (cost >= 0) & (cost <= 15))
    xyz = np.column_stack(((cells[:, 1:]-np.array(cost.shape[1:])//2)*payload['resolution']+payload['center'],
                           ground[tuple(cells.T)]))
    _, unique = np.unique(np.round(xyz, 4), axis=0, return_index=True)
    xyz, cells = xyz[unique], cells[unique]
    requested = np.array([start_base[0], start_base[1], floor_z])
    first = int(np.linalg.norm(xyz-requested, axis=1).argmin())
    if np.linalg.norm(xyz[first]-requested) > .3:
        raise ValueError('Current base has no traversable VST support within 0.3 m')
    pcd = scene/payload.get('pcd', 'terrain_downsampled.pcd')
    report = dict(state='planning', frame_id='world', source_vst_frame_id=payload.get('frame_id'),
        coordinate_contract='VST scene coordinates equal Gazebo world; LIO map is a separate frame',
        scene=str(scene), spawn=str(spawn_path), vst=str(vst), native_dir=str(native),
        source_sha256=dict(scene=sha256(scene/'scene.json'), vst=sha256(vst), pcd=sha256(pcd)),
        planner='native VST A* + quintic GPMP', teacher_parameters=payload.get('teacher_parameters'),
        settings=dict(seed=seed, segments=count, floor=spawn['floor'], floor_z=floor_z,
            floor_tolerance_m=.15, cost_threshold=15., cost_barrier=50., body_height_m=.4,
            endpoint_distance_m=list(distance_range),maximum_path_m=max_path,heading_yaw=heading,
            forward_half_cone_deg=cone_deg,maximum_attempts_per_segment=100,
            native_first_waypoint_tolerance_m=.2, goal_tolerance_m=.1),
        start_policy='current_robot_base', native_start_omitted=True,
        requested_start_body_xyz=list(map(float, start_base)), requested_start_support_xyz=requested.tolist(), start_snap_distance_m=float(np.linalg.norm(xyz[first]-requested)),
        chain=[dict(vst_index=cells[first].tolist(), support_xyz=xyz[first].tolist())], segments=[], attempts=[])
    rng = np.random.default_rng(seed)
    used = {first}
    previous = None
    try:
        for number in range(count):
            distance = np.linalg.norm(xyz-xyz[first], axis=1)
            valid_candidates=(distance>=distance_range[0])&(distance<=distance_range[1])
            if visited is not None:
                for point in visited:
                    valid_candidates &= np.linalg.norm(xyz[:,:2]-np.asarray(point)[:2],axis=1)>=1.
            if heading is not None:
                delta_xy=xyz[:,:2]-xyz[first,:2]
                cosine=(delta_xy@np.array([np.cos(heading),np.sin(heading)]))/np.maximum(distance,1e-9)
                forward=valid_candidates&(cosine>=np.cos(np.radians(cone_deg)))
                if np.any(forward):valid_candidates=forward
                else:report['settings']['heading_fallback_used']=True
            candidates = np.flatnonzero(valid_candidates)
            candidates = np.array([i for i in candidates if i not in used])
            if goal_index is not None:
                candidates = np.flatnonzero(np.all(cells == np.asarray(goal_index), axis=1))
            if not len(candidates):
                raise RuntimeError('No traversable short-goal candidate within the distance/heading constraints')
            succeeded = False
            for attempt in range(1, 101):
                last = int(rng.choice(candidates))
                si, gi = cells[first][[0, 2, 1]].astype(np.int32), cells[last][[0, 2, 1]].astype(np.int32)
                trial = dict(segment=number, attempt=attempt, start_vst_index=cells[first].tolist(),
                             goal_vst_index=cells[last].tolist(), success=False)
                try:
                    if not planner.plan(si, gi, True):
                        raise RuntimeError('Native A*/GPMP failed')
                    optimizer = planner.get_trajectory_optimizer_wnoj()
                    raw = np.asarray(optimizer.get_result_matrix())
                    height = np.asarray(optimizer.get_heights()).reshape(-1)
                    gpmp = np.column_stack(((raw[:, raw.shape[1]//2]-cost.shape[1]//2)*payload['resolution']+payload['center'][0],
                                            (raw[:, 0]-cost.shape[2]//2)*payload['resolution']+payload['center'][1], height))
                    xy = np.rint((gpmp[:, :2]-payload['center'])/payload['resolution']).astype(int)+np.array(cost.shape[1:])//2
                    if np.any(xy < 0) or np.any(xy >= np.array(cost.shape[1:])):
                        raise ValueError('GPMP left the map')
                    heights = ground[:, xy[:, 0], xy[:, 1]]
                    valid = allowed[:, xy[:, 0], xy[:, 1]]
                    errors = np.where(valid, np.abs(heights-gpmp[:, 2]), np.inf)
                    layer = errors.argmin(axis=0)
                    support = np.column_stack((gpmp[:, :2], heights[layer, np.arange(len(gpmp))]))
                    astar_path = np.asarray(planner.get_path_finder().get_result_matrix(), np.int32)
                    # GetResultMatrix returns [layer, world-x cell, world-y cell]
                    # (the planner input uses the reversed pair).
                    astar_ground = ground[tuple(astar_path.T)]
                    if not np.isfinite(astar_ground).all() or np.any(np.abs(astar_ground-floor_z) > .15):
                        raise ValueError('A* leaves the selected floor')
                    if not np.isfinite(gpmp).all() or np.any(np.abs(gpmp[:, 2]-support[:, 2]-.1) > .15):
                        raise ValueError('GPMP height leaves the selected support floor')
                    check_support_path(support, xyz[first], xyz[last], floor_z)
                    selected_cost = cost[layer, xy[:, 0], xy[:, 1]]
                    if np.any(selected_cost > 15) or not np.isfinite(selected_cost).all():
                        raise ValueError('GPMP support samples cross the VST cost threshold')
                    body = support + [0., 0., .4]
                    length=float(np.linalg.norm(np.diff(body,axis=0),axis=1).sum())
                    if max_path is not None and length>max_path:
                        raise ValueError('Optimized path exceeds the short-path limit')
                    stem = output/f'segment_{number:02d}'
                    for label, array in [('gpmp', gpmp), ('support', support), ('body', body)]:
                        atomic_npy(str(stem)+f'_{label}.npy', array)
                    atomic_npy(str(stem)+'_astar.npy', astar_path)
                    row = dict(id=number, start_vst_index=cells[first].tolist(), goal_vst_index=cells[last].tolist(),
                        start_support_xyz=xyz[first].tolist(), goal_support_xyz=xyz[last].tolist(),
                        start_body_xyz=(xyz[first]+[0, 0, .4]).tolist(), goal_body_xyz=(xyz[last]+[0, 0, .4]).tolist(),
                        support=str(stem)+'_support.npy', body=str(stem)+'_body.npy', gpmp=str(stem)+'_gpmp.npy',
                        length_m=length,
                        points=len(body), maximum_sampled_vst_cost=float(selected_cost.max()),
                        native_start_omitted=True,
                        seam_error_m=float(np.linalg.norm(previous[-1]-support[0])) if previous is not None else 0.)
                    report['segments'].append(row)
                    report['chain'].append(dict(vst_index=cells[last].tolist(), support_xyz=xyz[last].tolist()))
                    trial['success'] = succeeded = True
                    previous, first = support, last
                    used.add(last)
                except (ValueError, RuntimeError) as error:
                    trial['error'] = str(error)
                report['attempts'].append(trial)
                atomic_json(output/'mission.json', report)
                if succeeded:
                    break
            if not succeeded:
                raise RuntimeError(f'Segment {number} exhausted 100 endpoint attempts')
        report['state'] = 'complete'
    except Exception as error:
        report.update(state='failed', error=str(error))
        raise
    finally:
        report['wall_s'] = time.perf_counter()-begin
        atomic_json(output/'mission.json', report)
    print(output/'mission.json', flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--spawn', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-base', nargs=3, type=float, required=True)
    p.add_argument('--goal-index', nargs=3, type=int)
    p.add_argument('--seed', type=int, default=821111)
    p.add_argument('--segments', type=int, default=4)
    p.add_argument('--distance-min',type=float,default=6.)
    p.add_argument('--distance-max',type=float,default=16.)
    p.add_argument('--max-path',type=float)
    p.add_argument('--heading',type=float)
    p.add_argument('--cone-deg',type=float,default=70.)
    p.add_argument('--visited',type=Path)
    a = p.parse_args()
    plan(a.scene.resolve(), a.spawn.resolve(), a.native.resolve(), a.output.resolve(), a.start_base, a.seed, a.segments, a.goal_index,
         (a.distance_min,a.distance_max),a.max_path,a.heading,a.cone_deg,
         json.loads(a.visited.read_text()) if a.visited else None)


if __name__ == '__main__':
    main()
