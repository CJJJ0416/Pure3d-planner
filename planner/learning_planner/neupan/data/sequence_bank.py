# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.data.sequence_bank --help
"""Generate and fingerprint spatially isolated VST route banks."""
import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree

from planner.learning_planner.neupan.data.expert import NativeExpert, category, fingerprint, native_log
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.geometry.tomogram import Tomogram

SPLITS = ('train', 'validation', 'test')


def grouped(scene, tomo, seed):
    """Assign physical surfaces to stair instances before any route sampling."""
    stairs = sorted(scene['stairs'], key=lambda item: item['id'])
    if len(stairs) < 3:
        raise ValueError('Provide a scene with at least three independent connectors')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stairs))
    count = max(1, round(.15 * len(stairs)))
    labels = np.zeros(len(stairs), dtype=np.int8)
    labels[order[:count]], labels[order[count:2 * count]] = 1, 2
    x, y = np.indices(tomo.cost.shape[1:])
    xy = tomo.origin + tomo.resolution * np.stack((x, y), axis=-1)
    yaw = np.deg2rad(scene['config']['yaw_deg'])
    local = xy @ np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    ground = tomo.ground
    owner = np.full(ground.shape, -1, dtype=np.int8)
    best = np.full(ground.shape, np.inf, dtype=np.float32)
    scores = np.full((len(SPLITS), *ground.shape), np.inf, dtype=np.float32)
    for index, stair in enumerate(stairs):
        low = stair['center_world_xyz'][2]
        high = low + stair['rise_m']
        relative = np.abs(local - stair['center_local_xy'])
        outside = np.maximum(relative - np.asarray(stair['footprint_m']) / 2, 0.)
        dz = np.maximum(np.maximum(low - ground, ground - high), 0.)
        dist = np.sum(outside ** 2, axis=-1)[None] + dz ** 2
        # A reported physical footprint takes precedence over surrounding terrain.
        dist[(relative[..., 0] <= stair['footprint_m'][0] / 2)[None]
             & (relative[..., 1] <= stair['footprint_m'][1] / 2)[None] & (dz < .12)] -= 100.
        take = dist < best
        best[take] = dist[take]
        owner[take] = index
        np.minimum(scores[labels[index]], dist, out=scores[labels[index]], where=np.isfinite(dist))
    valid = owner >= 0
    split = np.where(valid, labels[owner], -1).astype(np.int8)
    # Ambiguous footprints are withheld from both folds. The remaining physical
    # cells are shared neither as student context nor as expert support.
    split[(scores < 0.).sum(axis=0) > 1] = -1
    return split, owner, [dict(stair, split=SPLITS[labels[i]]) for i, stair in enumerate(stairs)]


def masked(tomo, mask):
    return Tomogram(np.where(mask, tomo.cost, np.nan), np.where(mask, tomo.ground, np.nan),
                    tomo.origin, tomo.resolution, tomo.prediction)


def generate(nominal, scene_path, output, count, seed, attempts):
    tomo = Tomogram.read(nominal)
    scene = json.loads(scene_path.read_text())
    mask, owner, groups = grouped(scene, tomo, seed)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'ownership.npz', folds=mask, instances=owner)
    rng = np.random.default_rng(seed)
    paths, rows, failures = [], [], Counter()
    desired = [count - 2 * round(.15 * count), round(.15 * count), round(.15 * count)]
    with native_log(output / 'native.log'):
        for fold, label in enumerate(SPLITS):
            if desired[fold] <= 0:
                continue
            field = masked(tomo, mask == fold)
            astar = NativeExpert(field)
            cells = np.argwhere(field.admitted)
            xyz = field.xyz(cells)
            _, unique = np.unique(np.round(xyz, 3), axis=0, return_index=True)
            cells, xyz = cells[unique], xyz[unique]
            instance = owner[tuple(cells.T)]
            points = cKDTree(xyz)
            cross = []
            for serial, group in enumerate(groups):
                if group['split'] != label:
                    continue
                floor = group['center_world_xyz'][2]
                low = np.flatnonzero((instance == serial) & (abs(xyz[:, 2] - floor) < .24))
                high = np.flatnonzero((instance == serial) & (abs(xyz[:, 2] - floor - group['rise_m']) < .24))
                if len(low) and len(high):
                    cross.append((serial, low, high))
            totals, seen = Counter(), set()
            target = {kind: max(1, desired[fold] // 5) for kind in ('flat', 'detour', 'up', 'down', 'cross_floor')}
            for attempt in range(attempts):
                if sum(totals.values()) >= desired[fold]:
                    break
                if not len(xyz):
                    break
                kind = list(target)[attempt % len(target)]
                if attempt < attempts * .65 and totals[kind] >= target[kind]:
                    continue
                stair = None
                if kind == 'cross_floor':
                    if not cross:
                        failures[f'{label}/no_cross_endpoints'] += 1
                        continue
                    stair, low, high = cross[attempt % len(cross)]
                    if (attempt // len(cross)) % 2:
                        low, high = high, low
                    a, b = int(rng.choice(low)), int(rng.choice(high))
                else:
                    a = int(rng.integers(len(xyz)))
                    ids = np.asarray(points.query_ball_point(xyz[a], 8.), dtype=int)
                    delta = xyz[ids] - xyz[a]
                    dz = delta[:, 2]
                    z_ok = abs(dz) < .18 if kind in ('flat', 'detour') else ((dz if kind == 'up' else -dz) >= .18) & (abs(dz) < 1.2)
                    ids = ids[z_ok & (np.linalg.norm(delta[:, :2], axis=1) >= .8)]
                    if not len(ids):
                        continue
                    b = int(rng.choice(ids))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                path = astar.route(cells[a], cells[b])
                if path is None:
                    failures[f'{label}/astar'] += 1
                    continue
                trajectory = field.xyz(path)
                if category(trajectory) != kind:
                    failures[f'{label}/kind'] += 1
                    continue
                if stair is not None:
                    lower = groups[stair]['center_world_xyz'][2]
                    middle = (trajectory[:, 2] > lower + .30) & (trajectory[:, 2] < lower + groups[stair]['rise_m'] - .30)
                    if not middle.any() or not np.all(owner[tuple(path[middle].T)] == stair):
                        failures[f'{label}/different_connector'] += 1
                        continue
                row = dict(id=len(paths), split=label, region=label, kind=kind, goal=trajectory[-1].tolist())
                if stair is not None:
                    row['connector'] = groups[stair]['id']
                rows.append(row)
                paths.append(path)
                totals[kind] += 1
    path_cells = np.concatenate(paths) if paths else np.empty((0, 3), dtype=np.int32)
    np.savez_compressed(output / 'routes.npz', cells=path_cells, support_xyz=tomo.xyz(path_cells),
                        offsets=np.r_[0, np.cumsum([len(path) for path in paths])])
    dump(output / 'manifest.json', dict(routes=rows, tomogram=str(nominal.resolve()), sha256=fingerprint(nominal),
                                        scene_sha256=fingerprint(scene_path), ownership=str((output / 'ownership.npz').resolve()),
                                        groups=groups, splits=list(SPLITS), failures=dict(failures), seed=seed,
                                        target=count))
    return dict(Counter(f'{r["split"]}/{r["kind"]}' for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nominal', 'scene', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=7200)
    parser.add_argument('--seed', type=int, default=821111)
    parser.add_argument('--attempts', type=int, default=40000, help='Per fold')
    args = parser.parse_args()
    print(json.dumps(generate(args.nominal, args.scene, args.output, args.count, args.seed, args.attempts)))


if __name__ == '__main__':
    main()
