# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.data.landing_gpmp --help
"""Extract validated GPMP checkpoints. Expert samples seed training states only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

from planner.learning_planner.neupan.data.expert import ROOT, fingerprint, native_log, read_routes
from planner.learning_planner.neupan.geometry.landing import HEIGHT_TOLERANCE, MIN_AREA, read_raw_map, support_patches
from planner.learning_planner.neupan.geometry.tomogram import VST_LIMIT


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def arc_length(xyz):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]


class GPMPEngine:
    def __init__(self, payload, block, native_dir):
        sys.path.insert(0, str(native_dir))
        import a_star  # Register the native planner types before importing the wrapper.
        import traj_opt
        import ele_planner

        fields = np.asarray(payload['data'][(slice(None),) + block], dtype=np.float64)
        cost, gx, gy, ground, ceiling = fields
        valid = np.isfinite(ground)
        cost[~valid] = 50.
        ceiling = np.nan_to_num(ceiling, nan=1e6)
        shared = valid[1:] & valid[:-1] & (np.abs(np.diff(ground, axis=0)) < .1)
        gateway = np.zeros_like(cost, dtype=np.int32)
        change = np.diff(cost, axis=0)
        gateway[:-1][shared & (change < -8)] = 2
        gateway[1:][shared & (change > 8)] = -2

        def flat(values):
            return np.ascontiguousarray(values.reshape(-1, values.shape[-1]))

        self.engine = ele_planner.OfflineElePlanner(max_heading_rate=10., use_quintic=True)
        self.engine.init_map(15., 15., float(payload['resolution']), len(cost), 1.,
                             *(flat(arr) for arr in (cost, ground, ceiling, gateway, gy, -gx)))

    def route(self, start, goal):
        if not self.engine.plan(start[[0, 2, 1]], goal[[0, 2, 1]], True):
            return None, None
        optimizer = self.engine.get_trajectory_optimizer_wnoj()
        raw = np.asarray(optimizer.get_result_matrix())
        layers = np.rint(np.asarray(optimizer.get_layers()).ravel()).astype(int)
        height = np.asarray(optimizer.get_heights()).ravel()
        if len(raw) < 2 or len(raw) != len(height) or len(height) != len(layers):
            return None, None
        # Native trajectory stores grid y before grid x. Heights are in metres.
        return np.c_[raw[:, raw.shape[1] // 2], raw[:, 0], height], layers


def dense_support(raw, ceiling, points, layers):
    """Validate each dense GPMP sample against the chosen physical grid surface."""
    if not np.isfinite(points).all():
        return None, 'nonfinite_gpmp'
    phase = arc_length(points)
    keep = np.r_[True, np.diff(phase) > 1e-7]
    phase, points, layers = phase[keep], points[keep], layers[keep]
    if len(phase) < 2:
        return None, 'empty_gpmp'
    target = np.linspace(0., phase[-1], int(np.ceil(phase[-1] / .05)) + 1)
    xyz = np.column_stack([np.interp(target, phase, coord) for coord in points.T])
    segment = (np.searchsorted(phase, target, side='right') - 1).clip(0, len(phase) - 2)
    xy = np.rint((xyz[:, :2] - raw.origin) / raw.resolution).astype(int)
    if not ((xy >= 0) & (xy < raw.ground.shape[1:])).all():
        return None, 'outside_grid'
    # An interpolated point can be in either native layer of its segment.
    choices = np.stack((layers[segment], layers[segment + 1]))
    if np.any((choices < 0) | (choices >= len(raw.ground))):
        return None, 'invalid_layer'
    z = raw.ground[choices, xy[:, 0], xy[:, 1]]
    headroom = ceiling[choices, xy[:, 0], xy[:, 1]] - z
    # The native height may include smoothing and a body-reference offset.
    # A height hint cannot authorize switching to an arbitrary layer.
    difference = np.minimum(np.abs(z - xyz[:, 2]), np.abs(z - (xyz[:, 2] - .1)))
    admissible = np.isfinite(z) & (difference <= .22) & (np.isnan(headroom) | (headroom >= .5))
    admissible &= raw.cost[choices, xy[:, 0], xy[:, 1]] < 50.
    if not admissible.any(axis=0).all():
        return None, 'support_or_clearance'
    candidate = np.argmin(np.where(admissible, difference, np.inf), axis=0)
    xyz[:, 2] = z[candidate, np.arange(len(xyz))]
    if np.any(np.abs(np.diff(xyz[:, 2])) > .300001):
        return None, 'step_too_high'
    return (xyz, choices[candidate, np.arange(len(xyz))], target), None


def checkpoints(raw, dense, *, area_gate=True):
    xyz, layers, phase = dense
    sample_at = list(np.arange(0., phase[-1], .4)) + [phase[-1]]
    steps = np.flatnonzero(np.abs(np.diff(xyz[:, 2])) >= .08) + 1
    sample_at.extend(np.minimum(phase[steps] + .10, phase[-1]))
    if len(xyz) >= 7:
        prev, following = xyz[3:-3, :2] - xyz[:-6, :2], xyz[6:, :2] - xyz[3:-3, :2]
        angle = np.arctan2(prev[:, 0] * following[:, 1] - prev[:, 1] * following[:, 0],
                           (prev * following).sum(axis=1))
        bends = np.flatnonzero(np.abs(angle) > np.deg2rad(35)) + 3
        bends = bends[np.r_[True, np.diff(bends) > 3]] if len(bends) else bends
        sample_at.extend(np.maximum(phase[bends] - .12, 0.))
        sample_at.extend(np.minimum(phase[bends] + .12, phase[-1]))
    sorted_phase = np.unique(np.round(sample_at, 4))
    offsets = np.indices((5, 5)).reshape(2, -1).T - 2
    output = []
    previous = -np.inf
    for s in sorted_phase:
        at = int(np.argmin(np.abs(phase - s)))
        ref = xyz[at]
        local = np.flatnonzero((np.abs(phase - s) <= .15 + 1e-7) & (phase > previous + 1e-6))
        cells = np.rint((xyz[local, :2] - raw.origin) / raw.resolution).astype(int)
        xy = (cells[:, None] + offsets).reshape(-1, 2)
        origin = np.repeat(local, len(offsets))
        inside = ((xy >= 0) & (xy < raw.ground.shape[1:])).all(axis=1)
        xy, origin = xy[inside], origin[inside]
        indices = np.c_[layers[origin], xy]
        candidates = raw.xyz(indices)
        distance = np.linalg.norm(candidates[:, :2] - xyz[origin, :2], axis=1)
        reference = support_patches(raw, [ref], radius=.20, include_cells=True).support_cells[0]
        stride = raw.ground.shape[2]
        same_surface = np.isin(xy[:, 0] * stride + xy[:, 1], reference[:, 0] * stride + reference[:, 1])
        valid = (distance <= .20) & (np.linalg.norm(candidates[:, :2] - ref[:2], axis=1) <= .20)
        valid &= same_surface & (np.abs(candidates[:, 2] - ref[2]) <= HEIGHT_TOLERANCE)
        candidates, origin = candidates[valid], origin[valid]
        observed = support_patches(raw, candidates)
        center_valid = (observed.center_cost >= 0.) & (observed.center_cost <= VST_LIMIT)
        valid = np.flatnonzero(center_valid & ((observed.support_area >= MIN_AREA) if area_gate else True))
        if not len(valid):
            output.append(dict(nominal_s_m=float(s), nominal_xyz=ref.tolist(),
                               reason='insufficient_area' if area_gate else 'unmatched_support'))
            continue
        error = np.linalg.norm(candidates[valid] - ref, axis=1)
        phase_error = np.abs(phase[origin[valid]] - s)
        order = (phase_error, error, -np.round(observed.support_area[valid], 6)) if area_gate else (phase_error, error)
        best = valid[np.lexsort(order)[0]]
        previous = phase[origin[best]]
        output.append(dict(nominal_s_m=float(s), gpmp_s_m=float(previous), nominal_xyz=ref.tolist(),
                           support_xyz=candidates[best].tolist(), area_m2=float(observed.support_area[best]),
                           center_vst_cost=float(observed.center_cost[best]),
                           reason='supported'))
    return output


def pick_routes(manifest, count, seed):
    rng = np.random.default_rng(seed)
    records = manifest['routes']
    if count is None or count >= len(records):
        return list(records)
    force = {313, 3997, 4808}  # Keep the previously visualized held-out stair cases.
    selected = [row for row in records if row['id'] in force and row['split'] == 'test']
    selected = selected[:count]
    used = {row['id'] for row in selected}
    requested = {'train': round(.7 * count), 'validation': round(.15 * count)}
    requested['test'] = count - sum(requested.values())
    for split, target in requested.items():
        already = [row for row in selected if row['split'] == split]
        need = max(0, target - len(already))
        pool = [row for row in records if row['split'] == split and row['id'] not in used]
        kinds = ('cross_floor', 'up', 'down', 'detour', 'flat')
        batches = [list(rng.permutation([row for row in pool if row['kind'] == kind])) for kind in kinds]
        while need and any(batches):
            for candidates in batches:
                if candidates and need:
                    selected.append(candidates.pop())
                    need -= 1
    return selected[:count]


def make_labels(dataset, raw_map, output, native_dir, count, seed, *, sequence=False):
    bank, routes = read_routes(dataset)
    with Path(bank['tomogram']).open('rb') as stream:
        payload = pickle.load(stream)
    if bank['sha256'] != fingerprint(Path(bank['tomogram'])):
        raise ValueError('Nominal VST changed since the original route dataset')
    provenance = json.loads(raw_map.with_suffix('.json').read_text())
    if provenance['grid_sha256'] != bank['sha256'] or provenance['source_scope'] != 'clean_gt':
        raise ValueError('The raw cost field must be aligned clean GT')
    raw = read_raw_map(raw_map)
    with np.load(raw_map, allow_pickle=False) as arr:
        ceiling = arr['ceiling']
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)  # The native planner writes diagnostics into its working directory.
    selected = pick_routes(bank, count, seed)
    fields_by_fold = {}
    if sequence and 'ownership' in bank:
        from planner.learning_planner.neupan.data.sequence_bank import masked
        with np.load(bank['ownership'], allow_pickle=False) as maps:
            folds = maps['folds']
        for i, split in enumerate(bank['splits']):
            owned = folds == i
            layers = payload['data'].copy()
            layers[:, ~owned] = np.nan
            fields_by_fold[split] = (dict(payload, data=layers), masked(raw, owned))
    results = []
    with native_log(output / 'native.log'):
        for row in selected:
            local_payload, local_raw = fields_by_fold.get(row['split'], (payload, raw))
            route = routes[row['id']]
            low = np.maximum(route[:, 1:].min(axis=0) - 8, 0)
            high = np.minimum(route[:, 1:].max(axis=0) + 9, raw.ground.shape[1:])
            block = (slice(None), slice(low[0], high[0]), slice(low[1], high[1]))
            offset = np.r_[0, low]
            try:
                engine = GPMPEngine(local_payload, block, native_dir)
                values, layers = engine.route(route[0] - offset, route[-1] - offset)
            except RuntimeError as exc:
                results.append(dict(row, result='native_error', detail=str(exc)))
                continue
            if values is None:
                results.append(dict(row, result='gpmp_failed'))
                continue
            values[:, :2] = raw.origin + (values[:, :2] + low) * raw.resolution
            dense, error = dense_support(local_raw, ceiling, values, layers)
            records = checkpoints(local_raw, dense, area_gate=not sequence) if dense is not None else []
            file = f'route_{row["id"]}.npz'
            fields = dict(astar_xyz=raw.xyz(route), gpmp_xyz=values, gpmp_layers=layers)
            if dense:
                fields.update(support_xyz=dense[0], layers=dense[1], phase=dense[2])
            np.savez_compressed(output / file, **fields)
            results.append(dict(row, result=error or 'valid', file=file, checkpoints=records))
    counts = dict(Counter(row['result'] for row in results))
    report = dict(dataset=str(dataset), nominal_sha256=bank['sha256'], raw_sha256=fingerprint(raw_map),
                  seed=seed, counts=counts, routes=results, area_gate=not sequence)
    dump(output / 'manifest.json', report)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--raw-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-dir', type=Path, default=ROOT / 'datasets/cache/vst_native_py310')
    parser.add_argument('--count', type=int, default=200, help='Use zero for the entire route bank')
    parser.add_argument('--seed', type=int, default=821111)
    parser.add_argument('--sequence', action='store_true', help='Use geometric support without a footprint-area gate')
    args = parser.parse_args()
    print(json.dumps(make_labels(args.dataset.resolve(), args.raw_map.resolve(), args.output.resolve(),
                                 args.native_dir.resolve(), args.count or None, args.seed,
                                 sequence=args.sequence)), flush=True)


if __name__ == '__main__':
    main()
