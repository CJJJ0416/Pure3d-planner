# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.data.landing_approach --help
"""Measure connected low-cost approach support in the current observation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from planner.learning_planner.neupan.data.expert import fingerprint
from planner.learning_planner.neupan.data.landing_dataset import HORIZON, observed_window
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph, Tomogram


def approach_distances(tomo, start, points, limit=HORIZON):
    """Return supported horizontal approach lengths, or infinity if unconfirmed."""
    distances = np.full(len(points), np.inf, dtype=np.float32)
    if not len(points):
        return distances
    cell = np.rint((start[:2] - tomo.origin) / tomo.resolution).astype(int)
    if np.any(cell < 0) or np.any(cell >= tomo.ground.shape[1:]):
        return distances
    if np.isfinite(limit):
        # The extra cell preserves all corners of edges within the distance budget.
        tomo, _ = tomo.crop(np.r_[0, cell], width=2 * (limit + tomo.resolution))
    graph = SupportGraph(tomo)
    source = graph.match_column(start)
    if source < 0:
        return distances

    rows, slots = np.where(graph.neighbors >= 0)
    cols = graph.neighbors[rows, slots]
    lengths = np.linalg.norm(graph.xyz[rows, :2] - graph.xyz[cols, :2], axis=1)
    adjacency = csr_matrix((lengths, (rows, cols)), shape=(len(graph.xyz), len(graph.xyz)))
    from_start = dijkstra(adjacency, indices=source, limit=limit + 1e-5)

    xy = np.rint((points[:, :2] - tomo.origin) / tomo.resolution).astype(int)
    inside = ((xy >= 0) & (xy < graph.ids.shape[1:])).all(axis=1)
    x, y = xy[inside].T
    ids = graph.ids[:, x, y]
    heights = tomo.ground[:, x, y]
    # Candidates are measured support points. Allow repeated representations of
    # the same tread but never certify one by reaching a different height.
    matched = (ids >= 0) & (np.abs(heights - points[inside, 2]) <= .025)
    distances[inside] = np.min(np.where(matched, from_start[ids], np.inf), axis=0)
    return distances


def add_coverage(dataset, nominal, output):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['nominal_sha256'] != fingerprint(nominal):
        raise ValueError('Coverage requires the same nominal tomogram as the landing windows')
    tomo = Tomogram.read(nominal)
    with np.load(dataset / 'windows.npz', allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    distances = np.empty(len(arrays['xyz']), dtype=np.float32)
    for i, window in enumerate(manifest['windows']):
        sl = slice(*arrays['offsets'][i:i + 2])
        local, start, _ = observed_window(tomo, window)
        distances[sl] = approach_distances(local, start, arrays['xyz'][sl])
        if (i + 1) % 500 == 0:
            print(f'{i + 1} coverage windows', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'windows.npz', **arrays, approach_m=distances)
    manifest['approach_supervision'] = dict(max_distance_m=HORIZON, low_cost_lt=.8,
                                           source='connected observed support', dataset=str(dataset.resolve()))
    dump(output / 'manifest.json', manifest)
    return dict(windows=len(manifest['windows']), candidates=len(distances),
                confirmed=int(np.isfinite(distances).sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'nominal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(add_coverage(args.dataset, args.nominal, args.output)), flush=True)
