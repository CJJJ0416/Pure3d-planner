"""Cost-weighted routes and goal progress on the observed support graph."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from planner.learning_planner.neupan.data.landing_dataset import HORIZON
from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph


EDGE_COST_WEIGHT = 4.
MIN_COST_PROGRESS = .08


@dataclass
class RouteCosts:
    approach_m: np.ndarray
    approach_cost: np.ndarray
    remaining_cost: np.ndarray
    start_cost: float
    observed_goal: bool
    xyz: np.ndarray
    target_ids: np.ndarray
    weighted_path: np.ndarray
    weight_parent: np.ndarray
    distance_parent: np.ndarray

    @property
    def progress(self):
        return self.start_cost - self.remaining_cost

    def approach_path(self, index):
        node = self.target_ids[index]
        if node < 0 or not np.isfinite(self.approach_m[index]):
            return np.empty((0, 3))
        parent = self.weight_parent if self.weighted_path[node] else self.distance_parent
        chain = [node]
        while parent[node] >= 0:
            node = parent[node]
            chain.append(node)
        return self.xyz[chain[::-1]]


def tree_total(xyz, parent, order, edge_scale=None):
    """Accumulate a second edge metric on a shortest-path tree."""
    children = np.flatnonzero(parent >= 0)
    parents = parent[children]
    values = np.zeros(len(xyz))
    values[children] = np.linalg.norm(xyz[children, :2] - xyz[parents, :2], axis=1)
    if edge_scale is not None:
        values[children] *= (edge_scale[children] + edge_scale[parents]) * .5
    for node in order:
        if parent[node] >= 0:
            values[node] += values[parent[node]]
    return values


class LandingRoutes:
    """Represent measured support once per physical height and reuse both edge metrics."""
    def __init__(self, tomo):
        self.graph = SupportGraph(tomo)
        xyz = self.graph.xyz
        _, unique, self.identity = np.unique(np.rint(xyz * 1000.).astype(np.int64), axis=0,
                                             return_index=True, return_inverse=True)
        self.xyz = xyz[unique]
        scale = np.full(len(self.xyz), np.inf)
        np.minimum.at(scale, self.identity, 1. + EDGE_COST_WEIGHT * self.graph.cost)
        self.scale = scale
        rows, slots = np.where(self.graph.neighbors >= 0)
        pairs = np.unique(np.c_[self.identity[rows], self.identity[self.graph.neighbors[rows, slots]]], axis=0)
        src, dst = pairs.T
        length = np.linalg.norm(self.xyz[dst, :2] - self.xyz[src, :2], axis=1)
        shape = (len(self.xyz), len(self.xyz))
        self.lengths = csr_matrix((length, (src, dst)), shape=shape)
        self.weights = csr_matrix((length * (scale[src] + scale[dst]) * .5, (src, dst)), shape=shape)

    def point_ids(self, points):
        tomo = self.graph.tomo
        cells = np.rint((points[:, :2] - tomo.origin) / tomo.resolution).astype(int)
        inside = ((cells >= 0) & (cells < self.graph.ids.shape[1:])).all(axis=1)
        x, y = cells[inside].T
        ids = self.graph.ids[:, x, y]
        delta = abs(tomo.ground[:, x, y] - points[inside, 2])
        valid = (ids >= 0) & (delta <= .025)
        nearest = np.argmin(np.where(valid, delta, np.inf), axis=0)
        result = np.full(len(points), -1, dtype=int)
        result[inside] = np.where(valid.any(axis=0), ids[nearest, np.arange(len(x))], -1)
        match = result >= 0
        result[match] = self.identity[result[match]]
        return result

    def to_goal(self, goal, source):
        reverse = self.weights.transpose().tocsr()
        goals = np.flatnonzero(goal_mask(self.xyz, goal))
        if len(goals):
            distance = dijkstra(reverse, indices=goals, min_only=True)
            if np.isfinite(distance[source]):
                return distance, True

        # One fixed potential for this snapshot, including goals outside the box.
        # The virtual goal adds a 3-D residual to any possible observation endpoint.
        # A finite proxy cost never creates an edge across an unobserved region.
        xy = np.linalg.norm(self.xyz[:, :2] - goal[:2], axis=1)
        z = abs(self.xyz[:, 2] - goal[2])
        residual = 2. * (xy + 2. * z)
        edges = reverse.tocoo()
        n = len(self.xyz)
        extended = csr_matrix((np.r_[edges.data, residual],
                               (np.r_[edges.row, np.full(n, n)], np.r_[edges.col, np.arange(n)])),
                              shape=(n + 1, n + 1))
        return dijkstra(extended, indices=n)[:n], False

    def costs(self, start, goal, points):
        source = self.graph.match_column(start)
        if source < 0:
            return None
        source = self.identity[source]
        target_ids = self.point_ids(points)
        valid = target_ids >= 0
        dist, distance_parent = dijkstra(self.lengths, indices=source, limit=HORIZON + 1e-5,
                                         return_predecessors=True)
        remaining, observed_goal = self.to_goal(goal, source)
        weight, weight_parent = dijkstra(self.weights, indices=source, return_predecessors=True,
                                         limit=HORIZON * self.scale.max() + 1e-5)
        weighted_length = tree_total(self.xyz, weight_parent, np.argsort(weight))
        shortest_cost = tree_total(self.xyz, distance_parent, np.argsort(dist), self.scale)
        # Prefer the low-cost route within budget. If it detours too far, the
        # shortest supported route still gives a feasible weighted upper bound.
        cheap = np.isfinite(weight) & (weighted_length <= HORIZON + 1e-5)
        approach = np.where(cheap, weighted_length, dist)
        integrated = np.where(cheap, weight, np.where(np.isfinite(dist), shortest_cost, np.inf))
        result = RouteCosts(np.full(len(points), np.inf), np.full(len(points), np.inf),
                            np.full(len(points), np.inf), float(remaining[source]), observed_goal,
                            self.xyz, target_ids, cheap, weight_parent, distance_parent)
        ids = target_ids[valid]
        result.approach_m[valid] = approach[ids]
        result.approach_cost[valid] = integrated[ids]
        result.remaining_cost[valid] = remaining[ids]
        return result
