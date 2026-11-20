# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.evaluation.landing_revision --help
"""Audit revision_1994 landing patches and export independent preview data."""
import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np

from planner.learning_planner.neupan.data.expert import ROOT, native_log
from planner.learning_planner.neupan.geometry.landing import MIN_AREA, RADIUS, read_raw_map, support_patches
from planner.learning_planner.neupan.data.landing_gpmp import GPMPEngine, arc_length, checkpoints, dense_support, dump
from planner.learning_planner.neupan.geometry.tomogram import Tomogram


def requested(supply):
    xyz = [supply['routes'][0]['support'][0], *(row['support'][1] for row in supply['routes'][1:])]
    return np.asarray(xyz, dtype=float)


def prepare(scene, raw_file, output, native_dir):
    raw = read_raw_map(raw_file)
    with np.load(raw_file, allow_pickle=False) as arrays:
        ceiling = arrays['ceiling']
    snapshot = scene / 'vst_snapshot'
    report = json.loads((snapshot / 'summary.json').read_text())
    points = requested(report)
    with (snapshot / 'tomogram.pickle').open('rb') as stream:
        payload = pickle.load(stream)
    nominal = Tomogram.read(snapshot / 'tomogram.pickle')

    endpoints = []
    for point in points[:2]:
        x, y = np.rint((point[:2] - nominal.origin) / nominal.resolution).astype(int)
        error = np.abs(nominal.ground[:, x, y] - point[2])
        candidates = np.flatnonzero((error <= .15) & (nominal.cost[:, x, y] <= 15.))
        if not len(candidates):
            raise ValueError('Cannot recover the archived G1 GPMP anchors')
        layer = candidates[np.lexsort((candidates, error[candidates]))[0]]
        endpoints.append(np.array([layer, x, y], dtype=np.int32))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    astar = np.load(snapshot / 'tread_1_astar.npy', allow_pickle=False)
    os.chdir(output)
    with native_log(output / 'revision_gpmp.log'):
        engine = GPMPEngine(payload, (slice(None),) * 3, native_dir)
        engine.engine.set_reference_height(.1)
        xyz, layers = engine.route(*endpoints)
    if xyz is not None:
        xyz[:, :2] = raw.origin + xyz[:, :2] * raw.resolution
        dense, error = dense_support(raw, ceiling, xyz, layers)
        records = checkpoints(raw, dense) if dense is not None else []
    else:
        error, records, xyz = 'gpmp_failed', [], np.empty((0, 3))
    diagnostic = []
    if error and len(xyz):
        phase = arc_length(xyz)
        # These marks show what the nominal sampling would propose. They are
        # never consumed by the training dataset if the complete curve failed.
        for s in np.r_[np.arange(0., phase[-1], .4), phase[-1]]:
            index = int(np.argmin(np.abs(phase - s)))
            i, j = np.rint((xyz[index, :2] - raw.origin) / raw.resolution).astype(int)
            pos = xyz[index].copy()
            if 0 <= i < raw.cost.shape[1] and 0 <= j < raw.cost.shape[2]:
                z = raw.ground[layers[index], i, j]
                if np.isfinite(z):
                    pos[2] = z
            measure = support_patches(raw, [pos])
            diagnostic.append(dict(xyz=pos.tolist(), area_m2=float(measure.area[0]), teacher=False))
    dump(output / 'revision_checkpoint_report.json', dict(gpmp_result=error or 'valid', checkpoints=records,
                                                          diagnostic=diagnostic,
                                                          start=raw.xyz(endpoints[:1])[0].tolist()))
    np.savez_compressed(output / 'revision_curves.npz', astar=astar, gpmp=xyz)


def selection_rollout(scorer, tomo, start, goal, max_steps):
    """Replay one fixed observation as if each selected support point were reached."""
    from planner.learning_planner.neupan.inference.landing_route import LandingRoutes
    from planner.learning_planner.neupan.inference.reachability import goal_mask

    samples = [start.tolist()]
    approach_path = [start.tolist()]
    steps = []
    current = start
    status = 'step_limit'
    for _ in range(max_steps):
        choices, reason = scorer.candidates(tomo, current, goal)
        if not choices:
            status = reason
            break
        best = choices[0]
        current = best.xyz
        path = best.approach_xyz
        section = path[1:].tolist() if path is not None else [current.tolist()]
        approach_path.extend(section)
        steps.append(dict(xyz=current.tolist(), model_score=best.score, trav_cost=best.trav_cost,
                          approach_m=best.approach_m,
                          approach_cost=best.approach_cost, remaining_cost=best.remaining_cost,
                          observed_area_m2=best.observed_area_m2))
        revisited = any(np.linalg.norm(current - prior) < .075 for prior in samples)
        samples.append(current.tolist())
        if revisited:
            status = 'revisited'
            break
        if goal_mask(current, goal):
            status = 'goal_reached'
            break
    route = LandingRoutes(tomo).costs(start, goal, np.asarray(samples))
    values = [float(v) if np.isfinite(v) else None for v in route.remaining_cost] if route else []
    return dict(status=status, selected_xyz=samples, approach_xyz=approach_path, steps=steps,
                observed_goal=route.observed_goal if route else False, remaining_cost=values)


def audit(scene, raw_file, output, checkpoint=None, baseline=None, reference_dir=None, rollout_steps=0):
    snapshot = scene / 'vst_snapshot'
    source = json.loads((snapshot / 'summary.json').read_text())
    reference_dir = reference_dir or output
    reference = json.loads((reference_dir / 'revision_checkpoint_report.json').read_text())
    raw = read_raw_map(raw_file)
    requested_points = requested(source)
    start = np.asarray(reference['start'])
    with np.load(scene / 'frame.npz', allow_pickle=False) as arrays:
        cloud = np.c_[arrays['xyz'], arrays['prediction']]
    aligned = Tomogram.observed(cloud, start, grid_phase=raw.origin)
    default = Tomogram.observed(cloud, start)
    p = support_patches(aligned, requested_points)
    v = support_patches(raw, requested_points)
    initial = support_patches(default, requested_points)
    rows = [dict(name=f'G{i}', xyz=xyz.tolist(), native_planner_stage=source['routes'][i]['stage'],
                 native_preinflation_area_m2=float(v.area[i]), native_center_observed=bool(v.center_observed[i]),
                 predicted_area_m2=float(p.area[i]), predicted_center_observed=bool(p.center_observed[i]),
                 default_lattice_predicted_area_m2=float(initial.area[i]),
                 default_lattice_center_observed=bool(initial.center_observed[i]),
                 eligible=bool(p.center_observed[i] and p.area[i] >= MIN_AREA))
            for i, xyz in enumerate(requested_points)]
    scored = []
    status = 'area_audit'
    comparison = {}
    if checkpoint is not None:
        import torch
        from planner.learning_planner.neupan.data.landing_approach import approach_distances
        from planner.learning_planner.neupan.data.landing_dataset import HORIZON
        from planner.learning_planner.neupan.models.landing_model import LandingScorer
        torch.set_num_threads(2)
        # Approach G2 from the previous tread. Archive request points stay
        # exact for the area audit and are not snapped to adjacent cells.
        g1 = requested_points[1]
        x, y = np.rint((g1[:2] - aligned.origin) / aligned.resolution).astype(int)
        heights = aligned.ground[:, x, y]
        start = np.r_[aligned.origin + np.array([x, y]) * aligned.resolution,
                      heights[np.nanargmin(abs(heights - g1[2]))]]
        goal = requested_points[2]

        def replay(model, *, route_cost):
            goals, reason = LandingScorer(model, route_cost=route_cost).candidates(aligned, start, goal)
            points = np.asarray([g.xyz for g in goals]).reshape(-1, 3)
            lengths = approach_distances(aligned, start, points, limit=np.inf)
            records = [dict(xyz=g.xyz.tolist(), observed_area_m2=g.observed_area_m2,
                            estimated_area_m2=g.estimated_area_m2, score=g.score,
                            observed_approach_m=float(d) if np.isfinite(d) else None,
                            within_approach_budget=bool(d <= HORIZON + 1e-5))
                       for g, d in zip(goals, lengths)]
            return records, reason

        scored, status = replay(checkpoint, route_cost=True)
        if baseline is not None:
            before, before_status = replay(baseline, route_cost=False)
            comparison = dict(baseline_goals=before, baseline_status=before_status,
                              query_goal=goal.tolist(), approach_budget_m=HORIZON)
        if rollout_steps:
            before = LandingScorer(baseline or checkpoint, route_cost=False)
            after = LandingScorer(checkpoint)
            comparison.update(query_goal=goal.tolist(), approach_budget_m=HORIZON,
                              rollout=dict(before=selection_rollout(before, aligned, start, goal, rollout_steps),
                                           after=selection_rollout(after, aligned, start, goal, rollout_steps)))
    content = dict(revision=source['revision'], input='recorded predicted cost in archived VST grid phase',
                   min_area_m2=MIN_AREA, patch_radius_m=RADIUS, targets=rows,
                   gpmp_result=reference['gpmp_result'], checkpoints=reference['checkpoints'],
                   diagnostic=reference['diagnostic'], learned_goals=scored, learned_status=status,
                   query_start=start.tolist(), **comparison)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'revision_audit.json', content)
    with np.load(reference_dir / 'revision_curves.npz', allow_pickle=False) as curves:
        np.savez_compressed(output / 'revision_preview.npz', astar=curves['astar'], gpmp=curves['gpmp'],
                            cloud=aligned.cloud(), report=json.dumps(content, allow_nan=False))
    print(json.dumps({k: v for k, v in content.items() if k != 'checkpoints'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'audit'))
    for name in ('scene', 'raw-map', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--baseline', type=Path, help='Replay the old and new scorers on the identical query.')
    parser.add_argument('--reference', type=Path, help='Prepared directory when writing results elsewhere.')
    parser.add_argument('--rollout-steps', type=int, default=0,
                        help='Compare repeated selections with and without weighted route cost.')
    parser.add_argument('--native-dir', type=Path, default=ROOT / 'datasets/cache/vst_native_py310')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.scene.resolve(), args.raw_map.resolve(), args.output, args.native_dir.resolve())
    else:
        if (args.baseline or args.rollout_steps) and not args.checkpoint:
            parser.error('--baseline and --rollout-steps need --checkpoint')
        if args.rollout_steps < 0:
            parser.error('--rollout-steps must be nonnegative')
        audit(args.scene, args.raw_map, args.output, args.checkpoint, args.baseline, args.reference,
              args.rollout_steps)


if __name__ == '__main__':
    main()
