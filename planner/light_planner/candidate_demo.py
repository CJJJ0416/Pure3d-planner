"""Offline local multi-candidate experiment on an existing XYZ+trav_cost PCD."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common.pointcloud_io import read_pcd_fields, write_pcd
from .config import PlannerConfig
from .graph.candidates import CandidateOptions, LocalCandidateSearch, backtrack, score_features
from .surface import SparseSurfacePlanner


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def summarize(samples):
    return dict(samples=len(samples), p50_ms=float(np.median(samples)*1000),
                p95_ms=float(np.percentile(samples, 95)*1000))


def plot_case(folder, planner, start, goal, result, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz, cost = planner.full_xyz, planner.full_cost
    rows = np.linspace(0, len(xyz)-1, min(len(xyz), 60000), dtype=int)
    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    axes = [fig.add_subplot(131), fig.add_subplot(132, projection='3d'), fig.add_subplot(133)]
    a, b, c = axes
    a.scatter(xyz[rows, 0], xyz[rows, 1], c=cost[rows], cmap='RdYlGn_r', vmin=0, vmax=1,
              s=2, alpha=.35, rasterized=True)
    b.scatter(xyz[rows, 0], xyz[rows, 1], xyz[rows, 2], c=cost[rows], cmap='RdYlGn_r',
              vmin=0, vmax=1, s=1, alpha=.12, rasterized=True)
    for candidate in result['candidates'][::-1]:
        path = planner.node_xyz[candidate['node_path']]
        selected = candidate['candidate_id'] == 0
        color, width = ('#d81b60', 3.5) if selected else ('#387cb8', 1.)
        label = f"Selected (Z weight {result['options']['z_weight']:g})" if selected else None
        a.plot(path[:, 0], path[:, 1], color=color, lw=width, alpha=1 if selected else .4, label=label)
        b.plot(*path.T, color=color, lw=width, alpha=1 if selected else .4)
    if baseline is not None:
        path = planner.node_xyz[baseline['node_path']]
        a.plot(path[:, 0], path[:, 1], '--', color='#ff9800', lw=2, label='Same tree, Z weight 0')
        b.plot(*path.T, '--', color='#ff9800', lw=2)
    a.scatter(*start[:2], c='black', marker='o', s=40, label='Start query')
    a.scatter(*goal[:2], c='black', marker='*', s=120, label='Goal query')
    b.scatter(*start, c='black', s=35)
    b.scatter(*goal, c='black', marker='*', s=100)
    a.set(xlabel='World X (m)', ylabel='World Y (m)', title='Local candidates on saved cost PCD')
    a.set_aspect('equal', adjustable='datalim')
    a.legend(fontsize=8, loc='best')
    b.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Existing surface graph paths')
    b.view_init(elev=25, azim=-55)
    if result['candidates']:
        values = result['candidates']
        opts = result['options']
        h = opts['horizon_m']
        ids = np.arange(len(values))
        c.bar(ids-.24, [opts['z_weight']*v['z_progress_m']/h for v in values], width=.24,
              label='Z progress term', color='#d81b60')
        c.bar(ids, [opts['xyz_weight']*v['xyz_progress_m']/h for v in values], width=.24,
              label='XYZ progress term', color='#387cb8')
        c.bar(ids+.24, [-opts['cost_weight']*v['integral_cost_m']/(h*planner.config.cost_threshold)
                        for v in values], width=.24, label='Path cost term', color='#8c8c8c')
        c.plot(ids, [v['score'] for v in values], 'k.-', label='Total score')
    c.set(xlabel='Retained candidate rank (0 = selected)', ylabel='Score', title='Analytic score contributions')
    c.axhline(0, c='black', lw=.5)
    c.legend(fontsize=8)
    fig.suptitle(folder.name + ' | offline simulation, no robot commands')
    fig.savefig(folder/'candidates.png', dpi=155)
    plt.close(fig)


def run_case(case, rows, args, folder, options):
    folder.mkdir(parents=True, exist_ok=True)
    start, goal = np.asarray(case['start'], dtype=float), np.asarray(case['goal'], dtype=float)
    center = np.asarray(case.get('roi_center', start), dtype=float)
    tick = time.perf_counter()
    mask = np.ones(len(rows), bool)
    for k, name in enumerate(('x', 'y', 'z')):
        mask &= np.abs(rows[name]-center[k]) <= args.box[k]/2
    xyz = np.column_stack([rows[k][mask] for k in ('x', 'y', 'z')])
    cost = rows['trav_cost'][mask]
    crop_s = time.perf_counter()-tick
    cfg = PlannerConfig(cost_threshold=args.cost_limit, dc_input_cost_max=args.cost_limit)
    planner = SparseSurfacePlanner(cfg, folder/'native')
    planner.load_arrays(xyz, cost)
    write_pcd(folder/'local_input.pcd', xyz, cost)
    write_pcd(folder/'surface.pcd', planner.node_xyz, planner.node_cost)
    tick = time.perf_counter()
    start_id, fallback = planner.match_endpoint(start, 1.)
    snap_s = time.perf_counter()-tick
    report = dict(case=case, frame=args.frame, coordinate_reference='support XYZ, not body XYZ',
                  roi=dict(center=center.tolist(), size_m=args.box, points=len(xyz)),
                  graph=planner.report, graph_setup=planner.setup,
                  crop_s=crop_s, start_snap_s=snap_s,
                  start_node=start_id, start_blind_spot_fallback=fallback,
                  actual_execution=False, options=asdict(options))
    if start_id is None:
        report.update(success=False, reason='start_snap_failed')
        save_json(folder/'result.json', report)
        planner.close()
        return report
    search = LocalCandidateSearch(planner.step_graph, planner.node_xyz, planner.node_cost,
                                  cost_limit=args.cost_limit, astar_cost_weight=cfg.cost_weight,
                                  gravity_up=cfg.gravity_up)
    repeats = [search.query(start_id, goal, options) for _ in range(args.repeat)]
    result = repeats[0]
    report.update(success=result['success'], reason=result['reason'], metrics=result['metrics'],
                  start_anchor=planner.node_xyz[start_id].tolist(),
                  start_snap_distance_m=float(np.linalg.norm(start-planner.node_xyz[start_id])),
                  weighted_graph_prepare_s=search.prepare_s,
                  timing={k: summarize([r['metrics'][k] for r in repeats])
                          for k in result['metrics'] if k.endswith('_s')},
                  deterministic=all([c['node_id'] for c in r['candidates']] ==
                                    [c['node_id'] for c in result['candidates']] for r in repeats),
                  validation_scope='exact existing graph edges, not robot collision or tracking')
    features = result['features']
    baseline = None
    if len(features['node_id']):
        baseline_scores = score_features(features['z_progress_m'], features['xyz_progress_m'],
                                         features['integral_cost_m'], options.horizon_m,
                                         args.cost_limit, z_weight=0., xyz_weight=options.xyz_weight,
                                         cost_weight=options.cost_weight)
        index = int(np.lexsort((features['node_id'], features['length_m'], -baseline_scores))[0])
        node = int(features['node_id'][index])
        baseline = {k: float(v[index]) for k, v in features.items() if k != 'node_id'}
        baseline.update(node_id=node, score_with_z=baseline['score'], score=float(baseline_scores[index]),
                        endpoint=planner.node_xyz[node].tolist(),
                        node_path=backtrack(result['parents'], start_id, node))
        np.savetxt(folder/'no_z_selected.csv', planner.node_xyz[baseline['node_path']],
                   delimiter=',', header='x,y,z', comments='')
        report['no_z_ablation'] = {k: v for k, v in baseline.items() if k != 'node_path'}
        report['no_z_ablation']['additional_search_calls'] = 0
    np.savez_compressed(folder/'search_tree.npz', parents=result['parents'],
                        distances=result['distances'], node_xyz=planner.node_xyz, node_cost=planner.node_cost,
                        start_node=start_id, **{'endpoint_'+k: v for k, v in features.items()})
    keys = list(features)
    np.savetxt(folder/'endpoints.csv', np.column_stack([features[k] for k in keys]),
               delimiter=',', header=','.join(keys), comments='')
    report['candidates'] = []
    for candidate in result['candidates']:
        path = candidate['node_path']
        np.savetxt(folder/f"candidate_{candidate['candidate_id']:02d}.csv",
                   np.column_stack((planner.node_xyz[path], planner.node_cost[path], path)),
                   delimiter=',', header='x,y,z,trav_cost,node_id', comments='')
        report['candidates'].append({k: v for k, v in candidate.items() if k != 'node_path'})
    if result['success']:
        winner = result['candidates'][0]
        path = winner['node_path']
        np.savetxt(folder/'selected.csv', planner.node_xyz[path], delimiter=',', header='x,y,z', comments='')
        # Reference search is separate from candidate generation and benchmark timings.
        metrics, reference_path = planner.search.search(planner.step_graph, planner.node_xyz,
            planner.node_cost, [start_id, winner['node_id']], cfg.cost_weight)
        objective = winner['length_m'] + cfg.cost_weight*winner['integral_cost_m']
        report['separate_astar_check'] = dict(metrics=metrics,
            objective_difference=abs(metrics['objective']-objective),
            purpose='verify chosen tree path shortest objective, excluded from query timing')
        report['selected_interpolated_check'] = planner.validate_trajectory(planner.node_xyz[path], check_cost=False)
    report['duplicates'] = result['duplicates']
    save_json(folder/'result.json', report)
    plot_case(folder, planner, start, goal, result, baseline)
    planner.close()
    return report


def rollout_case(case, rows, args, folder, options):
    """Ideal graph following on a fixed saved map, not dynamics or perception simulation."""
    current = np.asarray(case['start'], dtype=float)
    goal = np.asarray(case['goal'], dtype=float)
    trace, steps, visits = [], [], {}
    search_calls = 0
    reason = 'step_budget_exhausted'
    for step in range(args.rollout_steps):
        if np.linalg.norm(current-goal) <= options.goal_tolerance_m:
            reason = 'goal_reached'
            break
        tick = time.perf_counter()
        mask = np.ones(len(rows), bool)
        for k, name in enumerate(('x', 'y', 'z')):
            mask &= np.abs(rows[name]-current[k]) <= args.box[k]/2
        xyz = np.column_stack([rows[k][mask] for k in ('x', 'y', 'z')])
        cost = rows['trav_cost'][mask]
        crop_s = time.perf_counter()-tick
        cfg = PlannerConfig(cost_threshold=args.cost_limit, dc_input_cost_max=args.cost_limit)
        planner = SparseSurfacePlanner(cfg, folder/'native')
        try:
            planner.load_arrays(xyz, cost)
            start_id, fallback = planner.match_endpoint(current, 1.)
            if start_id is None:
                reason = 'start_snap_failed'
                break
            search = LocalCandidateSearch(planner.step_graph, planner.node_xyz, planner.node_cost,
                cost_limit=args.cost_limit, astar_cost_weight=cfg.cost_weight, gravity_up=cfg.gravity_up)
            result = search.query(start_id, goal, options)
            search_calls += 1
            if not result['success']:
                reason = result['reason']
                break
            path = planner.node_xyz[result['candidates'][0]['node_path']]
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            index = max(1, int(np.searchsorted(distance, args.advance, side='right')-1)) if len(path)>1 else 0
            index = min(index, len(path)-1)
            next_position = path[index].copy()
            record = dict(step=step, start=current.tolist(), anchor=path[0].tolist(),
                snap_distance_m=float(np.linalg.norm(current-path[0])), next=next_position.tolist(),
                selected_endpoint=path[-1].tolist(), advanced_graph_length_m=float(distance[index]),
                xyz_error_m=float(np.linalg.norm(next_position-goal)),
                z_error_m=abs(float(next_position[2]-goal[2])), crop_s=crop_s,
                graph_prepare_s=planner.setup['total_prepare_s'],
                weighted_graph_prepare_s=search.prepare_s, query=result['metrics'])
            steps.append(record)
            # Retain the whole followed prefix, not a chord between decisions.
            trace.extend(path[:index+1].tolist())
            key = tuple(np.round(next_position, 3))
            visits[key] = visits.get(key, 0)+1
            if visits[key] >= 3:
                reason = 'repeated_position_cycle'
                current = next_position
                break
            current = next_position
        finally:
            planner.close()
    if np.linalg.norm(current-goal) <= options.goal_tolerance_m:
        reason = 'goal_reached'
    report = dict(success=reason=='goal_reached', reason=reason, steps=steps,
        final_position=current.tolist(), final_xyz_error_m=float(np.linalg.norm(current-goal)),
        final_z_error_m=abs(float(current[2]-goal[2])), graph_search_calls=search_calls,
        semantics='ideal selected-graph-prefix following on fixed full saved PCD, no robot dynamics or new observations',
        actual_execution=False)
    save_json(folder/'rollout.json', report)
    np.savetxt(folder/'rollout.csv', np.asarray(trace).reshape(-1, 3), delimiter=',', header='x,y,z', comments='')
    plot_rollout(folder, rows, np.asarray(trace).reshape(-1, 3), goal, report)
    print(json.dumps(dict(case=case['name'], rollout=reason, steps=len(steps),
                          final_xyz_error_m=report['final_xyz_error_m'])), flush=True)
    return report


def plot_rollout(folder, rows, trace, goal, report):
    if not len(trace):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bounds = np.vstack((trace, goal))
    lower, upper = bounds.min(axis=0)-.7, bounds.max(axis=0)+.7
    mask = np.ones(len(rows), bool)
    for k, name in enumerate(('x', 'y', 'z')):
        mask &= (rows[name] >= lower[k]) & (rows[name] <= upper[k])
    ids = np.flatnonzero(mask)
    ids = ids[np.linspace(0, len(ids)-1, min(60000, len(ids)), dtype=int)]
    xyz = np.column_stack([rows[n][ids] for n in ('x', 'y', 'z')])
    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    a = fig.add_subplot(131)
    b = fig.add_subplot(132, projection='3d')
    c = fig.add_subplot(133)
    a.scatter(xyz[:, 0], xyz[:, 1], c=rows['trav_cost'][ids], cmap='RdYlGn_r',
              vmin=0, vmax=1, s=1, alpha=.3, rasterized=True)
    a.plot(*trace[:, :2].T, color='#d81b60', lw=2.5)
    a.scatter(*trace[0, :2], c='black', s=40, label='Start anchor')
    a.scatter(*goal[:2], c='black', marker='*', s=120, label='Goal')
    a.set(xlabel='World X (m)', ylabel='World Y (m)', title='Followed graph prefixes')
    a.set_aspect('equal', adjustable='datalim')
    a.legend(fontsize=8)
    b.scatter(*xyz.T, c=rows['trav_cost'][ids], cmap='RdYlGn_r', vmin=0, vmax=1,
              s=1, alpha=.12, rasterized=True)
    b.plot(*trace.T, color='#d81b60', lw=2.5)
    b.scatter(*goal, c='black', marker='*', s=100)
    b.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Fixed saved point cloud')
    b.view_init(elev=25, azim=-55)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(trace, axis=0), axis=1))]
    c.plot(distance, trace[:, 2], color='#d81b60', lw=2, label='Support Z')
    c.axhline(goal[2], color='black', ls='--', label='Goal support Z')
    c.set(xlabel='Virtual travelled arc length (m)', ylabel='World Z (m)', title='Height along followed path')
    c.legend(fontsize=8)
    fig.suptitle(f"{folder.name} | {report['reason']} | {len(report['steps'])} virtual decisions | "
                 f"final XYZ error {report['final_xyz_error_m']:.3f} m")
    fig.savefig(folder/'rollout.png', dpi=155)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pcd', type=Path, required=True)
    p.add_argument('--cases', type=Path, help='JSON list of name/start/goal in support coordinates')
    p.add_argument('--start', type=float, nargs=3)
    p.add_argument('--goal', type=float, nargs=3)
    p.add_argument('--frame', default='world')
    p.add_argument('--box', type=float, nargs=3, default=[6., 6., 5.])
    p.add_argument('--cost-limit', type=float, default=.8)
    p.add_argument('--horizon', type=float, default=3.)
    p.add_argument('--minimum-length', type=float, default=.8)
    p.add_argument('--z-weight', type=float, default=8.)
    p.add_argument('--xyz-weight', type=float, default=2.)
    p.add_argument('--path-cost-weight', type=float, default=1.)
    p.add_argument('--repeat', type=int, default=10)
    p.add_argument('--rollout-steps', type=int, default=0, help='optional ideal virtual following decisions')
    p.add_argument('--advance', type=float, default=.5, help='metres of selected graph path followed per decision')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.repeat < 1 or args.rollout_steps < 0 or not np.isfinite(args.advance) or args.advance <= 0 or
            not np.isfinite(args.box).all() or min(args.box) <= 0):
        p.error('repeat and box dimensions must be positive')
    if args.cases:
        cases = json.loads(args.cases.read_text())
    elif args.start is not None and args.goal is not None:
        cases = [dict(name='query', start=args.start, goal=args.goal)]
    else:
        p.error('provide --cases or both --start and --goal')
    names = [case['name'] for case in cases]
    if not cases or len(set(names)) != len(names) or any(Path(n).name != n or n in ('.', '..') for n in names):
        p.error('cases need unique, nonempty safe directory names')
    for case in cases:
        for key in ('start', 'goal', 'roi_center'):
            value = np.asarray(case.get(key, case['start']), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                p.error('case coordinates must be finite XYZ')
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output/n/'result.json').exists() for n in names):
        p.error('use a new output directory to preserve prior results')
    options = CandidateOptions(horizon_m=args.horizon, minimum_length_m=args.minimum_length,
                               z_weight=args.z_weight, xyz_weight=args.xyz_weight,
                               cost_weight=args.path_cost_weight)
    began = time.perf_counter()
    rows = read_pcd_fields(args.pcd)
    load_s = time.perf_counter()-began
    if not {'x', 'y', 'z', 'trav_cost'}.issubset(rows.dtype.names):
        p.error('PCD must contain x,y,z,trav_cost')
    digest = hashlib.sha256()
    with args.pcd.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    reports = []
    for case in cases:
        report = run_case(case, rows, args, args.output/case['name'], options)
        if args.rollout_steps:
            report['rollout'] = rollout_case(case, rows, args, args.output/case['name'], options)
        reports.append(report)
        print(json.dumps(dict(case=case['name'], success=report['success'],
                              timing=report.get('timing'), candidates=report.get('metrics', {}).get('candidates'))), flush=True)
    save_json(args.output/'summary.json', dict(pcd=str(args.pcd.resolve()), pcd_sha256=digest.hexdigest(),
        input_points=len(rows), pcd_load_s=load_s, frame=args.frame, options=asdict(options),
        cases=reports, actual_execution=False, graph_rules_modified=False))


if __name__ == '__main__':
    main()
