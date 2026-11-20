# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.evaluation.sequence_benchmark --help
"""Held-out and recorded-frame benchmark for support-sequence checkpoints."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from planner.learning_planner.neupan.data.expert import fingerprint, read_routes
from planner.learning_planner.neupan.geometry.landing import read_raw_map, support_patches
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.models.landing_model import LandingScorer
from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.data.sequence_bank import masked
from planner.learning_planner.neupan.data.sequence_expert import Teacher, match_points, observation
from planner.learning_planner.neupan.training.sequence_train import Corpus, expert_series, preview, step_series
from planner.learning_planner.neupan.data.supervision import read_cloud
from planner.learning_planner.neupan.models.support_sequence import COST_UNIT, HORIZON, MAX_LENGTH, MAX_POINTS, SequenceGraph, SequencePlanner, tensors
from planner.learning_planner.neupan.geometry.tomogram import Tomogram


def baseline(model, tomo, graph, start, goal):
    """Roll out the previous common interface under the same return budgets."""
    current, past, steps, budget = start, [], [], MAX_LENGTH
    for _ in range(MAX_POINTS):
        if goal_mask(current, goal):
            return steps, 'goal_reached'
        choices, status = model.candidates(tomo, current, goal, limit=3, visited=past)
        if not choices:
            return steps, status
        source = graph.source(current)
        if source < 0:
            return steps, 'no_start_support'
        routes = graph.actions(source, budget, [*past, current])
        selected = None
        for choice in choices:
            target = graph.source(choice.xyz)
            ids = np.flatnonzero(routes.ids == target)
            if len(ids):
                selected = choice, int(ids[0])
                break
        if selected is None:
            return steps, 'no_confirmed_progress'
        p, i = selected
        budget -= routes.lengths[i]
        steps.append(dict(xyz=p.xyz.tolist(), approach=graph.xyz[routes.chain(i)].tolist(),
                          area_m2=float(support_patches(tomo, [p.xyz]).support_area[0]),
                          cost=p.trav_cost, length_m=float(routes.lengths[i])))
        past.append(current)
        current = p.xyz
    return steps, 'goal_reached' if goal_mask(current, goal) else 'horizon_reached'


def audit(graph, teacher, start, steps):
    visited = [start]
    topology, repeat, gt_errors, cross_layer, unknown = 0, 0, 0, 0, 0
    delta = None
    verified_length = 0.
    for segment in steps:
        xyz = np.asarray(segment['xyz'])
        path = np.asarray(segment['approach'])
        ids = match_points(graph, path, tolerance=.002)
        if (ids < 0).any():
            topology += 1
            unknown += 1
        elif np.any(np.asarray(graph.routes.weights[ids[:-1], ids[1:]]).reshape(-1) <= 0):
            topology += 1
            if np.any(abs(np.diff(path[:, 2])) > .300001):
                cross_layer += 1
        elif teacher is not None and not np.isfinite(teacher.chain_cost(ids)):
            gt_errors += 1
        repeat += int(any(np.linalg.norm(q[:2] - xyz[:2]) < .10 and abs(q[2] - xyz[2]) < .08 for q in visited))
        verified_length += float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
        visited.append(xyz)
    if teacher is not None and graph.source(start) >= 0 and graph.source(visited[-1]) >= 0:
        a, b = teacher.remaining[graph.source(start)], teacher.remaining[graph.source(visited[-1])]
        if np.isfinite([a, b]).all():
            delta = float(a - b)
    return dict(illegal_edges=topology, missing_edges=unknown, gt_invalid_segments=gt_errors,
                cross_layer=cross_layer, revisits=repeat, gt_progress=delta, length_m=verified_length)


@torch.inference_mode()
def pair_metrics(planner, graph, teacher, source):
    if teacher is None:
        return dict(area_total=0, area_preferred=0, cost_total=0, cost_preferred=0)
    label = teacher.label(source, visited=[graph.xyz[source]])
    a = label.actions
    x, _, _, _ = planner.model.encode(**tensors(graph.inputs(), planner.device))
    edge = torch.as_tensor(np.column_stack((a.lengths / HORIZON, a.costs / COST_UNIT)),
                           dtype=x.dtype, device=x.device)
    _, logits, _ = planner.model.step(x, source, x.new_zeros(64), a.ids, edge, MAX_LENGTH)
    scores = logits.cpu().numpy()
    hi, lo = label.pair_high, label.pair_low
    ids = a.ids[label.positive]
    index = np.flatnonzero(label.positive)
    near = graph.routes.lengths[ids][:, ids].tocoo()
    i, j = index[near.row], index[near.col]
    keep = ((abs(label.area[i] - label.area[j]) <= .001) & (abs(label.progress[i] - label.progress[j]) <= .20)
            & (label.costs[i] + .05 < label.costs[j])
            & (abs(graph.xyz[a.ids[i], 2] - graph.xyz[a.ids[j], 2]) <= .05))
    return dict(area_total=len(hi), area_preferred=int((scores[hi] > scores[lo]).sum()),
                cost_total=int(keep.sum()), cost_preferred=int((scores[i[keep]] > scores[j[keep]]).sum()))


@torch.inference_mode()
def controlled(models):
    """Mirror two same-tread probes to hold headings and one scoring axis fixed."""
    cases = []
    for kind in ('equal_cost', 'equal_footprint'):
        for side in (-1, 1):
            ground = np.zeros((1, 22, 17), dtype=np.float32)
            cost = np.zeros_like(ground)
            y = 8 + 2 * side
            if kind == 'equal_cost':
                ground[0, 7:10, y + side] = np.nan
            else:
                cost[0, 8, y] = .6
            tomo = Tomogram(cost, ground, np.zeros(2), .1, True)
            start, goal, preferred, alternative = tomo.xyz([[0, 3, 8], [0, 16, 8],
                                                             [0, 8, 8 - 2 * side], [0, 8, y]])
            graph = SequenceGraph(tomo, goal)
            source = graph.source(start)
            actions = graph.actions(source, visited=[start])
            ids = graph.routes.point_ids(np.array([preferred, alternative]))
            picks = [np.flatnonzero(actions.ids == i) for i in ids]
            if any(len(i) != 1 for i in picks):
                raise AssertionError('Controlled pair has an invalid action')
            choices = [int(i[0]) for i in picks]
            area = support_patches(tomo, [preferred, alternative]).support_area
            assert area[0] > area[1] if kind == 'equal_cost' else abs(area[0] - area[1]) < 1e-6
            record = dict(kind=kind, mirror=side, area_m2=area.tolist(), point_cost=graph.cost[ids].tolist(), scores={})
            for key, planner in models.items():
                x, _, _, _ = planner.model.encode(**tensors(graph.inputs(), planner.device))
                path = torch.as_tensor(np.c_[actions.lengths / HORIZON, actions.costs / COST_UNIT],
                                       dtype=x.dtype, device=x.device)
                _, logit, _ = planner.model.step(x, source, x.new_zeros(64), actions.ids, path, MAX_LENGTH)
                record['scores'][key] = logit[choices].cpu().tolist()
            cases.append(record)
    return cases


def one_case(tomo, start, goal, graph, teacher, source, models, old_models):
    expert, stop = expert_series(graph, teacher, source, goal)
    results = {'expert': dict(steps=expert, reason=stop, **audit(graph, teacher, start, expert))}
    for key, m in {**models, **old_models}.items():
        t = perf_counter()
        if key in models:
            result = m.plan(tomo, start, goal)
            duration = perf_counter() - t
            steps, reason = step_series(result), result.reason
        else:
            steps, reason = baseline(m, tomo, graph, start, goal)
            duration = perf_counter() - t
        results[key] = dict(steps=steps, reason=reason, latency_ms=1e3 * duration,
                            **audit(graph, teacher, start, steps))
        if key in models:
            results[key].update(pair_metrics(m, graph, teacher, source))
    return results


def summary(records):
    grouped = defaultdict(list)
    for row in records:
        for key, result in row['results'].items():
            for group in ('overall', row['kind'], row['variant']):
                grouped[key, group].append((result, row['results']['expert']['reason']))
    report = {}
    for (key, group), vals in grouped.items():
        result, truth = zip(*vals)
        reasons = Counter(x['reason'] for x in result)
        aggregate = dict(windows=len(result), reasons=dict(reasons), areas_m2=[
            stage['area_m2'] for x in result for stage in x['steps']])
        for metric in ('illegal_edges', 'missing_edges', 'gt_invalid_segments', 'cross_layer', 'revisits',
                       'area_total', 'area_preferred', 'cost_total', 'cost_preferred'):
            aggregate[metric] = sum(x.get(metric, 0) for x in result)
        blocked = {'no_confirmed_progress', 'no_start_support', 'low_goal_score', 'support_blocked',
                   'no_observed_support', 'insufficient_support_area', 'no_goal_progress'}
        aggregate['stop_matches'] = sum((x['reason'] in blocked) == (t in blocked) for x, t in vals)
        for metric in ('latency_ms', 'gt_progress'):
            values = [x[metric] for x in result if x.get(metric) is not None]
            aggregate[metric] = (dict(zip(('p50', 'p95'), np.percentile(values, [50, 95]).tolist())) if values else None)
        area = aggregate.pop('areas_m2')
        aggregate['support_area_m2'] = dict(zip(('p10', 'p50'), np.percentile(area, [10, 50]).tolist())) if area else None
        report.setdefault(key, {})[group] = aggregate
    return report


def snapshot(output, row, tomo, start, goal):
    """Reuse the read-only RViz transport for this benchmark's expert and student."""
    def curve(steps):
        xyz = [start]
        for item in steps:
            xyz.extend(item['approach'][1:])
        return np.asarray(xyz).tolist()

    before, after = row['results']['expert']['steps'], row['results']['sequence']['steps']
    report = dict(revision=f"sequence/test/{row['id']}/{row['variant']}", gpmp_result='support_sequence',
                  learned_goals=after, query_start=start.tolist(), query_goal=goal.tolist(),
                  targets=[dict(xyz=start.tolist()), dict(xyz=goal.tolist())],
                  sequence_case=True,
                  rollout={key: dict(approach_xyz=curve(steps), selected_xyz=[start.tolist()] + [p['xyz'] for p in steps],
                                     points=steps) for key, steps in [('before', before), ('after', after)]})
    np.savez_compressed(output, astar=np.empty((0, 3)), gpmp=np.empty((0, 3)), cloud=tomo.cloud(),
                        report=json.dumps(report, allow_nan=False))


def benchmark(a):
    torch.set_num_threads(2)
    c = Corpus(a.expert, a.raw_map)
    a.output.mkdir(parents=True, exist_ok=False)
    models = {'sequence': SequencePlanner(a.checkpoint)}
    if a.ablation:
        models['no_area_loss'] = SequencePlanner(a.ablation)
    old = {}
    if a.baseline:
        old = {'single_step':LandingScorer(a.baseline, route_cost=False),
               'route_center': LandingScorer(a.baseline)}
    probes = set()
    records = []
    for query in c.select('test', a.test_limit, a.seed):
        m, start, goal, tomo, graph, teacher, source = c.episode(query)
        row = dict(id=query[0], kind=m['kind'], variant=query[1],
                   results=one_case(tomo, start, goal, graph, teacher, source, models, old))
        records.append(row)
        if (m['kind'], query[1]) not in probes:
            probes.add((m['kind'], query[1]))
            snapshot(a.output / f"test_{query[0]}_{query[1]}.npz", row, tomo, start, goal)
    if not records:
        raise ValueError('No usable test queries in the expert manifest')
    report = dict(checkpoint_sha256=fingerprint(a.checkpoint),
                  expert_sha256=fingerprint(a.expert / 'manifest.json'),
                  test=summary(records), controlled_pairs=controlled(models))
    dump(a.output / 'test_cases.json', records)
    if a.replay or a.historical:
        raw = read_raw_map(a.raw_map)
        with np.load(a.raw_map, allow_pickle=False) as arrays:
            ceiling = arrays['ceiling']
        gt = SequenceGraph(masked(raw, ~np.isfinite(ceiling) | (ceiling - raw.ground >= .5)), [1e6] * 3)
    if a.replay:
        meta, paths = read_routes(a.replay)
        nominal = Tomogram.read(meta['tomogram'])
        if fingerprint(Path(meta['tomogram'])) != json.loads(a.raw_map.with_suffix('.json').read_text())['grid_sha256']:
            raise ValueError('Recorded replay and GT maps are not aligned')
        pairs = json.loads((a.replay / 'pairs.json').read_text())['frames']
        rows = []
        for frame in pairs:
            i = int(frame['route'])
            item = meta['routes'][i]
            if item['split'] != 'validation':
                continue
            name = a.replay / frame['cloud']
            with np.load(name, allow_pickle=False) as f:
                start = np.asarray(f['position'], dtype=float).copy()
            start[2] -= meta['sensor_height_m']
            goal = nominal.xyz(paths[i][-1:])[0]
            tomo = Tomogram.observed(read_cloud(name), start, grid_phase=nominal.origin)
            graph = SequenceGraph(tomo, goal)
            source = graph.source(start)
            teacher = Teacher(gt, graph, goal) if source >= 0 else None
            rows.append(dict(id=i, kind=item['kind'], variant='recorded', version=frame['version'],
                             results=one_case(tomo, start, goal, graph, teacher, source, models, old)))
        dump(a.output / 'recorded_cases.json', rows)
        report['recorded'] = summary(rows)
    if a.historical:
        data = json.loads(a.historical.read_text())['records']
        legacy = json.loads(a.historical_windows.read_text())
        bank = json.loads((Path(legacy['experts']) / 'manifest.json').read_text())
        nominal_path = Path(json.loads((Path(bank['dataset']) / 'manifest.json').read_text())['tomogram'])
        if fingerprint(nominal_path) != legacy['nominal_sha256']:
            raise ValueError('Historical windows and their nominal map differ')
        nominal = Tomogram.read(nominal_path)
        rows = []
        for case in data:
            i = case['window']
            window = legacy['windows'][i]
            start, goal = np.asarray(window['start']), np.asarray(window['goal'])
            tomo = observation(nominal, start, goal, window['scenario'])
            graph = SequenceGraph(tomo, goal)
            source = graph.source(start)
            teacher = Teacher(gt, graph, goal) if source >= 0 else None
            rows.append(dict(id=i, kind=window['kind'], variant=window['scenario'],
                             results=one_case(tomo, start, goal, graph, teacher, source, models, old)))
        dump(a.output / 'historical_cases.json', rows)
        report['historical'] = summary(rows)
    dump(a.output / 'report.json', report)
    visible = {}
    for row in records:
        if row['variant'] == 'full' or (row['kind'] == 'down' and row['variant'] == 'approach_gap'):
            visible.setdefault((row['kind'], row['variant']), (row['id'], row['variant']))
    preview(c, list(visible.values())[:6], a.checkpoint, a.output / 'heldout_preview', 'cpu',
            title='Held-out test windows')
    print(json.dumps({k: {name: group['overall'] for name, group in v.items()} for k, v in report.items()
                     if k in ('test', 'recorded')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('expert', 'raw-map', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('baseline', 'ablation', 'replay', 'historical', 'historical-windows'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--test-limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=821111)
    args = p.parse_args()
    if bool(args.historical) != bool(args.historical_windows):
        p.error('--historical and --historical-windows are required together')
    benchmark(args)


if __name__ == '__main__':
    main()
