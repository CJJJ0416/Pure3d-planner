# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.data.landing_dataset --help
"""Offline landing labels. GPMP provides queries, while GT geometry provides targets."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from planner.learning_planner.neupan.data.expert import fingerprint, read_routes
from planner.learning_planner.neupan.geometry.landing import MIN_AREA, RADIUS, observed_candidates, read_raw_map, support_patches
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph, Tomogram

HORIZON = 1.2
DISK_AREA = np.pi * RADIUS ** 2
FEATURE_DIM = 20
GEOMETRIC_AREA_METRIC = 'connected geometric footprint'


class GoalOracle:
    """Clean-map distance fields independent of the predicted graph and GPMP path."""
    def __init__(self, raw, lower, upper):
        ij_min = np.maximum(0, np.ceil((np.asarray(lower) - raw.origin) / raw.resolution).astype(int))
        ij_max = np.minimum(raw.ground.shape[1:],
                            np.ceil((np.asarray(upper) - raw.origin) / raw.resolution).astype(int))
        cut = (slice(None), slice(ij_min[0], ij_max[0]), slice(ij_min[1], ij_max[1]))
        self.map = Tomogram(raw.cost[cut], raw.ground[cut], raw.origin + ij_min * raw.resolution,
                             raw.resolution)
        self.graph = SupportGraph(self.map)
        rows, slots = np.where(self.graph.neighbors >= 0)
        cols = self.graph.neighbors[rows, slots]
        distance = np.linalg.norm(self.graph.xyz[rows, :2] - self.graph.xyz[cols, :2], axis=1)
        # Merge exact physical ground duplicates across elevation slices. A
        # different raster copy of the same tread is not a different foothold.
        _, representatives, identities = np.unique(np.rint(self.graph.xyz * 1000.).astype(np.int32),
                                                    axis=0, return_index=True, return_inverse=True)
        self.xyz = self.graph.xyz[representatives]
        self.canonical = identities
        edges, unique = np.unique(np.c_[identities[rows], identities[cols]], axis=0, return_index=True)
        n = len(self.xyz)
        self.adjacency = csr_matrix((distance[unique], (edges[:, 0], edges[:, 1])), shape=(n, n))
        self.reverse = self.adjacency.transpose().tocsr()
        self._sources = None
        self._goals = None
        self._goal_distances = None

    def matches(self, points):
        cells = np.rint((points[:, :2] - self.map.origin) / self.map.resolution).astype(int)
        inside = ((cells >= 0) & (cells < self.map.ground.shape[1:])).all(axis=1)
        valid = cells[inside]
        z = self.map.ground[:, valid[:, 0], valid[:, 1]]
        ids = self.graph.ids[:, valid[:, 0], valid[:, 1]]
        difference = np.abs(z - points[inside, 2])
        correct = (ids >= 0) & (difference <= .075)
        best = np.argmin(np.where(correct, difference, np.inf), axis=0)
        result = np.full(len(points), -1, dtype=int)
        result[inside] = np.where(correct.any(axis=0), ids[best, np.arange(len(valid))], -1)
        match = result >= 0
        result[match] = self.canonical[result[match]]
        return result

    def to_goal(self, goal):
        # Route windows with the same target share their clean cost-to-go.
        if self._goals is None or not np.array_equal(goal, self._goals):
            self._goals = goal.copy()
            self._sources = np.flatnonzero(goal_mask(self.xyz, goal))
            self._goal_distances = (dijkstra(self.reverse, indices=self._sources, min_only=True)
                                    if len(self._sources) else None)
        return self._goal_distances

    def labels(self, start, goal, points, area):
        delta = np.zeros(len(points), dtype=np.float32)
        known = np.zeros(len(points), dtype=bool)
        can_continue = np.zeros(len(points), dtype=bool)
        reachable = np.zeros(len(points), dtype=bool)
        start_id = self.matches(start[None])[0]
        to_goal = self.to_goal(goal)
        if start_id < 0 or to_goal is None or not np.isfinite(to_goal[start_id]):
            return delta, known, reachable, can_continue
        ids = self.matches(points)
        valid = ids >= 0
        known = valid.copy()
        reachable[valid] = np.isfinite(to_goal[ids[valid]])
        if reachable.any():
            from_start = dijkstra(self.adjacency, indices=start_id, limit=HORIZON + 1e-5)
            can_continue[reachable] = np.isfinite(from_start[ids[reachable]])
            eligible = reachable & can_continue & (area >= MIN_AREA)
            # Reward reduction in true-map distance, including detours that
            # initially move away in Euclidean coordinates.
            delta[eligible] = np.maximum(0., to_goal[start_id] - to_goal[ids[eligible]])
        return delta, known, reachable, can_continue


def probe_features(tomo, probes, tolerance=.12):
    xy = np.rint((probes[..., :2] - tomo.origin) / tomo.resolution).astype(int)
    inside = ((xy >= 0) & (xy < tomo.cost.shape[1:])).all(axis=-1)
    safe = np.clip(xy, 0, np.asarray(tomo.cost.shape[1:]) - 1)
    z = tomo.ground[:, safe[..., 0], safe[..., 1]]
    cost = tomo.cost[:, safe[..., 0], safe[..., 1]]
    close = inside[None] & (np.abs(z - probes[None, ..., 2]) <= tolerance)
    affordable = cost < .8 if tomo.prediction else cost <= 15.
    return np.c_[close.any(axis=0).mean(axis=1),
                 (close & np.isfinite(cost) & (cost >= 0.) & affordable).any(axis=0).mean(axis=1)]


def features(tomo, start, goal, points, *, geometric_area=False):
    patch = support_patches(tomo, points)
    goal_vector = (goal - start) / 4.
    scale = 1. if tomo.prediction else 50.
    cost = np.nan_to_num(patch.center_cost / scale, nan=1.)
    area = patch.support_area if geometric_area else patch.area
    data = np.c_[(points - start) / HORIZON, np.broadcast_to(goal_vector, (len(points), 3)),
                 (points - goal) / 4., area / DISK_AREA, cost, patch.center_observed]
    line = start[None, None] + (points - start)[:, None, :] * np.array([.15, .3, .5, .7, .85])[None, :, None]
    heading = goal[:2] - points[:, :2]
    heading /= np.linalg.norm(heading, axis=1, keepdims=True).clip(min=1e-6)
    lateral = np.c_[-heading[:, 1], heading[:, 0]]
    forward = np.broadcast_to(points[:, None], (len(points), 4, 3)).copy()
    forward[..., :2] += .2 * heading[:, None] + np.array([-.2, -.08, .08, .2])[None, :, None] * lateral[:, None]
    neighborhood = np.broadcast_to(points[:, None], (len(points), 4, 3)).copy()
    neighborhood[..., :2] += np.array([[.25, 0.], [0., .25], [-.25, 0.], [0., -.25]])
    extras = [probe_features(tomo, samples) for samples in (line[:, :2], line[:, 2:], forward, neighborhood)]
    return np.column_stack((data, *extras)).astype(np.float32), patch


def region_crop(tomo, start):
    # Crop around the measured start at its physical XY, preserving all height slices.
    index = np.rint((start[:2] - tomo.origin) / tomo.resolution).astype(int)
    local, _ = tomo.crop(np.r_[0, index])
    return Tomogram(local.cost / 50., local.ground, local.origin, local.resolution, True)


def hide(tomo, start, direction, scenario):
    x, y = np.indices(tomo.cost.shape[1:])
    dx = tomo.origin[0] + x * tomo.resolution - start[0]
    dy = tomo.origin[1] + y * tomo.resolution - start[1]
    along = dx * direction[0] + dy * direction[1]
    # Hide a strip of observed approach or the distal candidate patch. Clean GT labels
    # are always computed on the original full map, irrespective of this masking.
    remove = (along >= .22) & (along <= .42) if scenario == 'approach_gap' else along >= .63
    cost, ground = tomo.cost.copy(), tomo.ground.copy()
    cost[:, remove], ground[:, remove] = np.nan, np.nan
    return Tomogram(cost, ground, tomo.origin, tomo.resolution, True)


def observed_window(tomo, window):
    start, goal = (np.asarray(window[key], dtype=np.float32) for key in ('start', 'goal'))
    local = region_crop(tomo, start)
    if window['scenario'] != 'full':
        direction = goal[:2] - start[:2]
        local = hide(local, start, direction / np.linalg.norm(direction), window['scenario'])
    return local, start, goal


def collect(dataset, experts, raw_path, regions, output):
    bank, paths = read_routes(dataset)
    supply = json.loads((experts / 'manifest.json').read_text())
    if supply['nominal_sha256'] != bank['sha256'] or supply['raw_sha256'] != fingerprint(raw_path):
        raise ValueError('GPMP checkpoints do not match the input maps')
    inflated, raw = Tomogram.read(bank['tomogram']), read_raw_map(raw_path)
    bounds = {row['id']: row for row in json.loads(regions.read_text())['regions']}
    oracles = {}
    arrays = {key: [] for key in ('features', 'area_gt', 'observed_support_area', 'observed_low_cost_area', 'progress',
                                  'known', 'to_goal', 'incoming', 'xyz')}
    windows, offsets = [], [0]
    for row in supply['routes']:
        route_id = row['id']
        if row['region'] not in oracles:
            bound = bounds[row['region']]
            oracles[row['region']] = GoalOracle(raw, bound['lower'], bound['upper'])
        oracle = oracles[row['region']]
        gt_path = raw.xyz(paths[route_id])
        goals = [np.asarray(c['support_xyz'], dtype=np.float32) for c in row['checkpoints']
                 if c['reason'] == 'supported'] if row.get('file') else []
        # GPMP waypoints identify informative starts only. Candidates are scored
        # on the clean-map outcome instead of distance to the expert centerline.
        if row['result'] != 'valid' or not goals:
            query_starts = [gt_path[0], gt_path[len(gt_path) // 2]]
        else:
            chosen = np.unique(np.linspace(0, max(0, len(goals) - 2), 5, dtype=int))
            query_starts = [goals[i] for i in chosen]
        goal = np.asarray(row['goal'], dtype=np.float32)
        for i, start in enumerate(query_starts):
            if goal_mask(start, goal):
                continue
            local = region_crop(inflated, start)
            points = observed_candidates(local, start)
            if not len(points):
                continue
            area = support_patches(raw, points).support_area
            if row['result'] == 'valid':
                gain, known, reachable, incoming = oracle.labels(start, goal, points, area)
            else:
                gain = np.zeros(len(points), dtype=np.float32)
                known = reachable = incoming = np.zeros(len(points), dtype=bool)
            goal_direction = goal[:2] - start[:2]
            norm = np.linalg.norm(goal_direction)
            variants = [('full', local)]
            if norm > 1e-6 and i == len(query_starts) // 2:
                variants.extend((kind, hide(local, start, goal_direction / norm, kind))
                                for kind in ('approach_gap', 'target_occluded'))
            for scenario, observed in variants:
                x, patches = features(observed, start, goal, points, geometric_area=True)
                targets = dict(features=x, area_gt=area, observed_support_area=patches.support_area,
                               observed_low_cost_area=patches.area,
                               progress=gain, known=known,
                               to_goal=reachable, incoming=incoming, xyz=points)
                for key, value in targets.items():
                    arrays[key].append(value)
                offsets.append(offsets[-1] + len(points))
                windows.append(dict(route=route_id, split=row['split'], kind=row['kind'],
                                    scenario=scenario, start=start.tolist(), goal=goal.tolist()))
        if len(windows) and (route_id == supply['routes'][-1]['id'] or len(windows) % 50 < 3):
            print(f'{len(windows)} windows so far', flush=True)
    if not windows:
        raise ValueError('No support-area windows were generated')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'windows.npz', offsets=np.asarray(offsets),
                        **{k: np.concatenate(v) for k, v in arrays.items()})
    report = dict(experts=str(experts.resolve()), nominal_sha256=bank['sha256'],
                  raw_sha256=supply['raw_sha256'], inputs='nominal inflated VST with synthetic masks',
                  area_metric=GEOMETRIC_AREA_METRIC,
                  windows=windows, fold_counts=dict(Counter(w['split'] for w in windows)),
                  candidates=offsets[-1])
    dump(output / 'manifest.json', report)
    return report['fold_counts']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'experts', 'raw-map', 'regions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.dataset, args.experts, args.raw_map, args.regions, args.output)))


if __name__ == '__main__':
    main()
