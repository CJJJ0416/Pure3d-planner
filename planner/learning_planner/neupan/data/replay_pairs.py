# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.data.replay_pairs --help
"""Pair a saved, world-aligned traversal cost replay with original VST A* labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from planner.learning_planner.neupan.data.expert import DEFAULT_MAP, NativeExpert, category, fingerprint, native_log
from planner.learning_planner.neupan.data.supervision import read_cloud
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph, Tomogram


def pair_replay(tomogram, frames, output, *, offset, train_y, validation_y, lookahead, train_query_radius=0.):
    if not np.isfinite([offset, train_y, validation_y, lookahead, train_query_radius]).all() or lookahead <= 0 or train_query_radius < 0:
        raise ValueError('Expected finite replay parameters and a positive lookahead')
    if validation_y - train_y < 6.:
        raise ValueError('Keep the two 6 m observation bands separated')
    files = sorted(frames.glob('*.npz'))
    if not files:
        raise ValueError('No replay frames')
    positions = []
    limits = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            positions.append(data['position'])
            limits.append((float(data['lower'][1]), float(data['upper'][1])))
    support = np.array(positions, dtype=float)
    support[:, 2] -= offset
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(support[:, :2], axis=0), axis=1))]
    tomo = Tomogram.read(tomogram)
    cells = np.argwhere(tomo.admitted).astype(np.int32)
    xyz = tomo.xyz(cells)
    points = cKDTree(xyz)
    output.mkdir(parents=True, exist_ok=False)
    sample, paths, manifests = [], [], []
    reject = Counter()

    def closest(p):
        neighbors = np.array(points.query_ball_point(p, .28), dtype=int)
        delta = xyz[neighbors] - p
        accept = (np.linalg.norm(delta[:, :2], axis=1) <= .22) & (np.abs(delta[:, 2]) <= .20)
        neighbors, delta = neighbors[accept], delta[accept]
        if len(neighbors) == 0:
            return None
        return cells[neighbors[np.argmin(np.linalg.norm(delta, axis=1))]]

    validate_start = validation_y - 3.
    train_stop = train_y + 3.
    with native_log(output / 'native.log'):
        engine = NativeExpert(tomo)
        for index, position in enumerate(support):
            if position[1] <= train_y and limits[index][1] <= train_stop + 1e-6:
                split = 'train'
            elif position[1] >= validation_y and limits[index][0] >= validate_start - 1e-6:
                split = 'validation'
            else:
                reject['spatial_buffer'] += 1
                continue
            goal_index = int(distance.searchsorted(distance[index] + lookahead))
            if goal_index >= len(support):
                reject['end_of_recording'] += 1
                continue
            start, goal = closest(position), closest(support[goal_index])
            if start is None or goal is None:
                reject['no_gt_support'] += 1
                continue
            if split == 'train' and train_query_radius:
                graph = SupportGraph(Tomogram.observed(read_cloud(files[index]), position))
                height = np.abs(graph.xyz[:, 2] - position[2])
                planar = np.linalg.norm(graph.xyz[:, :2] - position[:2], axis=1)
                eligible = np.flatnonzero((height <= .15) & (planar <= train_query_radius))
                for node in eligible[np.argsort(planar[eligible])]:
                    proposed = closest(graph.xyz[node])
                    if proposed is not None and graph.match_column(tomo.xyz([proposed])[0]) >= 0:
                        start = proposed
                        break
                else:
                    reject['no_training_query_support'] += 1
                    continue
            route = engine.route(start, goal)
            if route is None:
                reject['no_vst_path'] += 1
                continue
            rid = len(paths)
            paths.append(route)
            xyz_route = tomo.xyz(route)
            sample.append(dict(id=rid, split=split, region=split, kind=category(xyz_route), goal=xyz_route[-1].tolist()))
            manifests.append(dict(route=rid, offset=0, frame='world', version=files[index].stem,
                                  cloud=str(files[index].resolve())))
    if not paths:
        raise ValueError('Replay has no matched VST labels')
    all_cells = np.concatenate(paths)
    np.savez_compressed(output / 'routes.npz', offsets=np.r_[0, np.cumsum([len(route) for route in paths])],
                        cells=all_cells, support_xyz=tomo.xyz(all_cells))
    metadata = dict(tomogram=str(tomogram.resolve()), sha256=fingerprint(tomogram),
                    source=str(frames.resolve()), sensor_height_m=offset, lookahead_m=lookahead, train_query_radius_m=train_query_radius,
                    train_y_max=train_y, validation_y_min=validation_y, generated=len(paths),
                    split_counts=dict(Counter(row['split'] for row in sample)), rejected=dict(reject))
    (output / 'manifest.json').write_text(json.dumps({**metadata, 'routes': sample}, indent=2) + '\n')
    (output / 'pairs.json').write_text(json.dumps(dict(frames=manifests), indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tomogram', type=Path, default=DEFAULT_MAP)
    parser.add_argument('--frames', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sensor-height', required=True, type=float)
    parser.add_argument('--train-y-max', required=True, type=float)
    parser.add_argument('--validation-y-min', required=True, type=float)
    parser.add_argument('--lookahead', type=float, default=2.)
    parser.add_argument('--train-query-radius', type=float, default=0.,
                        help='Training-only offset to a real, mutually matched low-cost support')
    args = parser.parse_args()
    print(json.dumps(pair_replay(args.tomogram, args.frames, args.output, offset=args.sensor_height,
                                 train_y=args.train_y_max, validation_y=args.validation_y_min,
                                 lookahead=args.lookahead, train_query_radius=args.train_query_radius)))


if __name__ == '__main__':
    main()
