"""One Dijkstra tree, multiple local endpoints, analytic ranking and path NMS.

Consumes an existing graph without changing its nodes, edges or cost gate.
There is one shortest tree path per endpoint, not K homotopies to one endpoint.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class CandidateOptions:
    horizon_m: float = 3.0
    minimum_length_m: float = 0.8
    max_candidates: int = 16
    z_weight: float = 8.0
    xyz_weight: float = 2.0
    cost_weight: float = 1.0
    dedup_distance_m: float = 0.25
    dedup_height_m: float = 0.10
    resample_points: int = 24
    goal_tolerance_m: float = 0.15
    direction_sectors: int = 0

    def __post_init__(self):
        positive = [self.horizon_m, self.minimum_length_m, self.dedup_distance_m,
                    self.dedup_height_m, self.goal_tolerance_m]
        weights = [self.z_weight, self.xyz_weight, self.cost_weight]
        if (not np.isfinite(positive + weights).all() or min(positive) <= 0 or
                min(weights) < 0 or self.minimum_length_m > self.horizon_m or
                self.max_candidates < 1 or self.resample_points < 2 or
                not 0 <= self.direction_sectors <= self.max_candidates):
            raise ValueError('Invalid local candidate options')


def score_features(z_progress, xyz_progress, integral_cost, horizon, cost_limit,
                   z_weight=8.0, xyz_weight=2.0, cost_weight=1.0):
    """Z is progress toward goal height, so descending is rewarded as well."""
    return (z_weight * np.asarray(z_progress) / horizon +
            xyz_weight * np.asarray(xyz_progress) / horizon -
            cost_weight * np.asarray(integral_cost) / (horizon * cost_limit))


def backtrack(parents, start, endpoint):
    chain = [int(endpoint)]
    while chain[-1] != start:
        parent = int(parents[chain[-1]])
        if parent < 0 or len(chain) > len(parents):
            raise ValueError('Invalid predecessor tree')
        chain.append(parent)
    return np.asarray(chain[::-1], dtype=np.int64)


def resample_path(xyz, count):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    samples = np.linspace(0., distance[-1], count)
    return np.column_stack([np.interp(samples, distance, xyz[:, k]) for k in range(3)])


def paths_duplicate(a, b, up, distance_m, height_m):
    # Preserve distinct terminal heights even if most of the route is shared.
    height_difference = abs(float((a[-1] - b[-1]) @ up))
    rms = float(np.sqrt(np.mean(np.sum((a - b)**2, axis=1))))
    return height_difference <= height_m and rms <= distance_m


class LocalCandidateSearch:
    """A weighted snapshot of an existing CSR surface graph, reusable per goal."""

    def __init__(self, graph, xyz, cost, *, astar_cost_weight=5., cost_limit=.8,
                 gravity_up=(0., 0., 1.)):
        began = time.perf_counter()
        self.xyz = np.array(xyz, dtype=np.float64, order='C', copy=True)
        self.cost = np.array(cost, dtype=np.float64, copy=True)
        self.up = np.asarray(gravity_up, dtype=float)
        n = len(self.xyz)
        if (self.xyz.shape != (n, 3) or self.cost.shape != (n,) or
                graph.shape != (n, n) or not np.isfinite(self.xyz).all() or
                not np.isfinite(self.cost).all() or np.any(self.cost < 0) or
                not np.isfinite(cost_limit) or not 0 < cost_limit <= 1 or
                np.any(self.cost >= cost_limit) or self.up.shape != (3,) or
                not np.isfinite(self.up).all() or np.linalg.norm(self.up) < 1e-9 or
                not np.isfinite(astar_cost_weight) or astar_cost_weight < 0):
            raise ValueError('Invalid graph snapshot or cost parameters')
        self.up = self.up / np.linalg.norm(self.up)
        self.cost_limit = cost_limit
        self.astar_cost_weight = astar_cost_weight
        self.graph = graph.tocsr(copy=True).astype(np.float64)
        rows = np.repeat(np.arange(n), np.diff(self.graph.indptr))
        cols = self.graph.indices
        lengths = np.linalg.norm(self.xyz[rows] - self.xyz[cols], axis=1)
        self.graph.data = lengths * (1. + astar_cost_weight *
                                     (self.cost[rows] + self.cost[cols]) / 2.)
        self.prepare_s = time.perf_counter() - began

    def query(self, start, goal, options=None, *, blocked_edges=None):
        options = options or CandidateOptions()
        goal = np.asarray(goal, dtype=float)
        if not 0 <= start < len(self.xyz) or goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError('Invalid start node or goal')
        began = time.perf_counter()
        # Any path of length <= H has objective <= H*(1+alpha*cost_limit).
        # This bound does not exclude an eligible shortest-tree path.
        limit = options.horizon_m * (1. + self.astar_cost_weight * self.cost_limit)
        graph = self.graph
        if blocked_edges is not None and np.any(blocked_edges):
            graph = self.graph.copy()
            graph.data[blocked_edges] = np.inf
        distances, parents = dijkstra(graph, directed=True, indices=int(start),
                                     return_predecessors=True, limit=limit)
        search_s = time.perf_counter() - began
        reached = np.flatnonzero(np.isfinite(distances))
        tick = time.perf_counter()
        length = np.full(len(self.xyz), np.nan)
        integral = np.full(len(self.xyz), np.nan)
        length[start] = integral[start] = 0.
        # Memoized tree traversal also handles equal-cost / zero-length ties.
        for endpoint in reached:
            node = int(endpoint)
            pending = []
            while not np.isfinite(length[node]):
                pending.append(node)
                node = int(parents[node])
                if node < 0 or len(pending) > len(parents):
                    raise ValueError('Broken Dijkstra predecessor tree')
            for node in reversed(pending):
                parent = int(parents[node])
                edge = float(np.linalg.norm(self.xyz[node] - self.xyz[parent]))
                length[node] = length[parent] + edge
                integral[node] = integral[parent] + edge*(self.cost[node]+self.cost[parent])/2.
        tree_s = time.perf_counter() - tick
        tick = time.perf_counter()
        root = self.xyz[start]
        error = np.linalg.norm(self.xyz - goal, axis=1)
        arrived = float(error[start]) <= options.goal_tolerance_m
        eligible = (length >= options.minimum_length_m) & (length <= options.horizon_m + 1e-9)
        # A goal within the local horizon remains an option even if closer than min length.
        eligible |= (error <= options.goal_tolerance_m) & (length > 1e-9) & (length <= options.horizon_m)
        ids = np.flatnonzero(eligible)
        if arrived:
            ids = np.array([start])
        z_progress = abs(float((root-goal) @ self.up)) - np.abs((self.xyz[ids]-goal) @ self.up)
        xyz_progress = float(np.linalg.norm(root-goal)) - error[ids]
        scores = score_features(z_progress, xyz_progress, integral[ids], options.horizon_m,
                                self.cost_limit, options.z_weight, options.xyz_weight,
                                options.cost_weight)
        order = np.lexsort((ids, length[ids], -scores))
        # Once the manual goal is reachable within this horizon, terminate there.
        # A cheaper/longer continuation must not displace a reachable goal.
        goal_candidates=error[ids]<=options.goal_tolerance_m
        goal_reachable=bool(np.any(goal_candidates))
        sectors = None
        if options.direction_sectors:
            axis = goal - root
            axis -= (axis @ self.up) * self.up
            if np.linalg.norm(axis) < 1e-9:
                axis = np.eye(3)[np.argmin(np.abs(self.up))]
                axis -= (axis @ self.up) * self.up
            axis /= np.linalg.norm(axis)
            offsets = self.xyz[ids] - root
            angles = np.arctan2(offsets @ np.cross(self.up, axis), offsets @ axis)
            sectors = np.floor((angles + np.pi) * options.direction_sectors / (2*np.pi)).astype(int)
            sectors %= options.direction_sectors
            _, first = np.unique(sectors[order], return_index=True)
            priority = np.zeros(len(order), dtype=bool)
            priority[first] = True
            order = np.r_[order[priority], order[~priority]]
        if goal_reachable:order=np.r_[order[goal_candidates[order]],order[~goal_candidates[order]]]
        features = dict(node_id=ids, length_m=length[ids], integral_cost_m=integral[ids],
                        z_progress_m=z_progress, xyz_progress_m=xyz_progress, score=scores)
        score_s = time.perf_counter() - tick
        tick = time.perf_counter()
        candidates, sampled, duplicates = [], [], []
        examined = 0
        for feature_index in order:
            examined += 1
            node = int(ids[feature_index])
            path = backtrack(parents, start, node)
            samples = resample_path(self.xyz[path], options.resample_points)
            duplicate = next((i for i, other in enumerate(sampled)
                              if paths_duplicate(samples, other, self.up,
                                                 options.dedup_distance_m, options.dedup_height_m)), None)
            if duplicate is not None:
                duplicates.append(dict(node_id=node, duplicate_of=duplicate))
                continue
            record = {k: float(v[feature_index]) for k, v in features.items() if k != 'node_id'}
            record.update(candidate_id=len(candidates), node_id=node, node_path=path,
                          reaches_goal=bool(goal_candidates[feature_index]),
                          endpoint=self.xyz[node].tolist(),
                          mean_cost=float(integral[node]/max(length[node], 1e-12)),
                          max_cost=float(self.cost[path].max()))
            if sectors is not None:
                record['direction_sector'] = int(sectors[feature_index])
            candidates.append(record)
            sampled.append(samples)
            if len(candidates) >= options.max_candidates:
                break
        dedup_s = time.perf_counter() - tick
        if sectors is not None:
            candidates.sort(key=lambda row: (not row['reaches_goal'],-row['score'], row['length_m'], row['node_id']))
            new_ids = {row['candidate_id']: index for index, row in enumerate(candidates)}
            for index, row in enumerate(candidates):
                row['candidate_id'] = index
            for row in duplicates:
                row['duplicate_of'] = new_ids[row['duplicate_of']]
        tick = time.perf_counter()
        for record in candidates:
            path = record['node_path']
            # Exact graph membership, without GPMP, interpolation or new edges.
            valid = all(int(v) in self.graph.indices[self.graph.indptr[u]:self.graph.indptr[u+1]]
                        for u, v in zip(path[:-1], path[1:]))
            if not valid:
                raise ValueError('Candidate contains an edge absent from the input graph')
            record['existing_graph_edges_valid'] = True
        validation_s = time.perf_counter() - tick
        return dict(success=bool(candidates), reason='already_at_goal' if arrived else
                    ('local_candidate_selected' if candidates else 'no_reachable_endpoint_within_horizon'),
                    selected_id=0 if candidates else None, candidates=candidates,
                    parents=parents, distances=distances, features=features,
                    duplicates=duplicates, options=asdict(options),
                    metrics=dict(graph_search_calls=1, reached_nodes=len(reached),
                                 eligible_endpoints=len(ids), dedup_examined=examined,
                                 duplicate_paths=len(duplicates), candidates=len(candidates),
                                 unexamined_due_to_cap=len(ids)-examined,
                                 weighted_search_limit=limit, search_s=search_s, tree_s=tree_s,
                                 scoring_s=score_s, backtrack_dedup_s=dedup_s,
                                 validation_s=validation_s, total_query_s=time.perf_counter()-began))
