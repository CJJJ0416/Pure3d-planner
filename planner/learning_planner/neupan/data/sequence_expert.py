"""Label the actual observed transitions against complete VST support."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra

from planner.learning_planner.neupan.geometry.landing import RADIUS, support_patches
from planner.learning_planner.neupan.data.landing_dataset import HORIZON, hide, region_crop
from planner.learning_planner.neupan.inference.landing_route import MIN_COST_PROGRESS
from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.models.support_sequence import MAX_LENGTH, SequenceGraph


SCENARIOS = ('full', 'approach_gap', 'target_occluded', 'detour_gap')
DISK_AREA = np.pi * RADIUS ** 2


def observation(nominal, point, goal, scenario):
    local = region_crop(nominal, point)
    v = goal[:2] - point[:2]
    norm = np.linalg.norm(v)
    if scenario == 'full' or norm < 1e-7:
        return local
    v /= norm
    if scenario != 'detour_gap':
        return hide(local, point, v, scenario)
    xy = local.origin + local.resolution * np.moveaxis(np.indices(local.ground.shape[1:]), 0, -1)
    delta = xy - point[:2]
    along, across = delta @ v, delta @ np.array([-v[1], v[0]])
    gap = (along >= .22) & (along <= .42) & (abs(across) <= .25)
    local.cost[:, gap] = np.nan
    local.ground[:, gap] = np.nan
    return local


def match_points(graph, xyz, tolerance=.075):
    tomo = graph.tomo
    cells = np.rint((xyz[:, :2] - tomo.origin) / tomo.resolution).astype(int)
    inside = ((cells >= 0) & (cells < tomo.ground.shape[1:])).all(axis=1)
    x, y = cells[inside].T
    ids = graph.routes.graph.ids[:, x, y]
    delta = abs(tomo.ground[:, x, y] - xyz[inside, 2])
    match = (ids >= 0) & (delta <= tolerance)
    nearest = np.argmin(np.where(match, delta, np.inf), axis=0)
    nodes = np.full(len(xyz), -1, dtype=int)
    nodes[inside] = np.where(match.any(axis=0), ids[nearest, np.arange(len(x))], -1)
    valid = nodes >= 0
    nodes[valid] = graph.routes.identity[nodes[valid]]
    return nodes


@dataclass
class Supervision:
    actions: object
    area: np.ndarray
    costs: np.ndarray
    progress: np.ndarray
    positive: np.ndarray
    utility: np.ndarray
    pair_high: np.ndarray
    pair_low: np.ndarray
    pair_weight: np.ndarray


class Teacher:
    def __init__(self, expert_graph, observed, goal):
        self.gt = expert_graph
        self.observed = observed
        self.ids = match_points(expert_graph, observed.xyz)
        eligible = self.ids >= 0
        goals = np.flatnonzero(goal_mask(self.gt.xyz, goal))
        distance = (dijkstra(self.gt.routes.weights.T.tocsr(), indices=goals, min_only=True)
                    if len(goals) else np.full(len(self.gt.xyz), np.inf))
        self.remaining = np.full(len(self.ids), np.inf)
        self.remaining[eligible] = distance[self.ids[eligible]]
        patches = support_patches(self.gt.tomo, observed.xyz)
        # A mapped high-cost surface is a known negative, rather than an unlabeled node.
        self.known = patches.center_observed
        self.coverage = patches.support_area.astype(np.float32)
        self.transition = self._transitions()

    def _transitions(self):
        pairs = self.observed.routes.weights.tocoo()
        src, dst = self.ids[pairs.row], self.ids[pairs.col]
        indices = np.flatnonzero((src >= 0) & (dst >= 0))
        cost = np.asarray(self.gt.routes.weights[src[indices], dst[indices]]).reshape(-1)
        n = len(self.ids)
        return {int(pairs.row[i]) * n + int(pairs.col[i]): float(value)
                for i, value in zip(indices, cost) if value > 0}

    def chain_cost(self, nodes):
        n = len(self.ids)
        edges = (nodes[:-1].astype(np.int64) * n + nodes[1:]).tolist()
        return sum(self.transition.get(key, np.inf) for key in edges)

    def label(self, source, budget=MAX_LENGTH, visited=()):
        actions = self.observed.actions(source, budget, visited)
        ids = actions.ids
        gt_cost = np.array([self.chain_cost(actions.chain(i)) for i in range(len(ids))])
        gt_d = self.remaining
        progress = np.full(len(ids), -np.inf)
        if np.isfinite(gt_d[source]):
            progress = gt_d[source] - gt_d[ids]
        positive = np.isfinite(gt_cost) & (progress >= MIN_COST_PROGRESS)
        utility = np.full(len(ids), -np.inf)
        utility[positive] = 2 * progress[positive] - gt_cost[positive]
        area = self.coverage[ids]
        # Adjacent compatible support actions on the same tread, matched on cost and progress.
        valid_ids = ids[positive]
        indices = np.flatnonzero(positive)
        sub = self.observed.routes.lengths[valid_ids][:, valid_ids].tocoo()
        a, b = indices[sub.row], indices[sub.col]
        delta = self.observed.xyz[ids[a]] - self.observed.xyz[ids[b]]
        point_cost = self.observed.cost[ids]
        keep = ((np.linalg.norm(delta[:, :2], axis=1) <= .20 + 1e-6) & (abs(delta[:, 2]) <= .05)
                & (abs(point_cost[a] - point_cost[b]) <= .05) & (abs(gt_cost[a] - gt_cost[b]) <= .20)
                & (abs(progress[a] - progress[b]) <= .20) & (area[a] > area[b]))
        high, low = a[keep], b[keep]
        return Supervision(actions, area, gt_cost, progress, positive, utility, high, low,
                           (area[high] - area[low]) / DISK_AREA)

    def prefix(self, source, goal, visited=()):
        budget = MAX_LENGTH
        result = []
        past = list(visited)
        for _ in range(3):
            if goal_mask(self.observed.xyz[source], goal):
                return result, 'goal_reached'
            past.append(self.observed.xyz[source])
            decision = self.label(source, budget, past)
            if not decision.positive.any():
                return result, 'no_confirmed_progress'
            idx = int(np.argmax(decision.utility))
            result.append((decision, idx))
            source = int(decision.actions.ids[idx])
            budget -= decision.actions.lengths[idx]
        return result, 'goal_reached' if goal_mask(self.observed.xyz[source], goal) else 'horizon_reached'
