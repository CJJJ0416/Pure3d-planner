"""Local, observation-constrained labels for a full-map VST expert route."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from planner.learning_planner.neupan.data.expert import fingerprint, read_routes
from planner.learning_planner.neupan.geometry.tomogram import STOP, SupportGraph, Tomogram

HORIZON = 3.
MAX_STEPS = 64


@dataclass
class Window:
    graph: SupportGraph
    chain: np.ndarray
    goal: np.ndarray
    end: str
    scenario: str
    build_ms: float = 0.

    def actions(self):
        if not len(self.chain):
            return []
        visited = set()
        result = []
        for source, target in zip(self.chain, self.chain[1:]):
            visited.add(int(source))
            mask = np.isin(self.graph.neighbors[source], tuple(visited))
            slot = np.flatnonzero(self.graph.neighbors[source] == target)
            if len(slot) != 1 or mask[slot[0]]:
                raise ValueError('Expert prefix contains an edge outside the local action set')
            result.append((source, mask, slot[0]))
        if self.end in ('goal_reached', 'support_blocked'):
            source = self.chain[-1]
            result.append((source, np.isin(self.graph.neighbors[source], tuple(visited)), STOP))
        return result


def expert_prefix(graph, intended, goal, scenario, start=None):
    """Follow the VST labels, rejoining the intended route on the strict VST graph."""
    if start is None:
        start = int(intended[0])
    if start < 0:
        return Window(graph, np.empty(0, dtype=np.int32), goal, 'no_start_support', scenario)

    chain = [start]
    travelled = 0.
    end = 'goal_reached'
    matrix = None
    visited = {start}
    cursor = 1
    while cursor < len(intended):
        here = chain[-1]
        next_id = int(intended[cursor])
        if next_id < 0 or next_id not in graph.neighbors[here] or next_id in visited:
            # Search the same directed VST support adjacency used by the policy.
            # A local alternative must actually rejoin a later expert support.
            if matrix is None:
                rows, slots = np.where(graph.neighbors >= 0)
                cols = graph.neighbors[rows, slots]
                dist = np.linalg.norm(graph.xyz[rows] - graph.xyz[cols], axis=1)
                matrix = csr_matrix((dist * (1. + graph.cost[cols]), (rows, cols)),
                                    shape=(len(graph.cells), len(graph.cells)))
            allowed = np.ones(len(graph.cells), dtype=bool)
            allowed[list(visited - {here})] = False
            rows, cols = matrix.nonzero()
            retained = allowed[rows] & allowed[cols]
            local = csr_matrix((matrix.data[retained], (rows[retained], cols[retained])), shape=matrix.shape)
            distance, parent = dijkstra(local, indices=here, return_predecessors=True)
            chosen = next((i for i in range(cursor, len(intended)) if intended[i] >= 0
                           and intended[i] not in visited and np.isfinite(distance[intended[i]])), None)
            if chosen is None:
                end = 'support_blocked'
                break
            cursor = chosen
            segment = [int(intended[cursor])]
            while parent[segment[-1]] != here:
                segment.append(int(parent[segment[-1]]))
            segment.reverse()
        else:
            segment = [next_id]
        for target in segment:
            step = np.linalg.norm(graph.xyz[target, :2] - graph.xyz[chain[-1], :2])
            if travelled + step > HORIZON + 1e-6 or len(chain) > MAX_STEPS:
                return Window(graph, np.array(chain), goal, 'local_horizon', scenario)
            travelled += step
            chain.append(target)
            visited.add(target)
        cursor += 1
    return Window(graph, np.array(chain), goal, end, scenario)


def local_window(tomo, route, offset, goal, scenario):
    goal = np.asarray(goal, dtype=np.float32)
    started = perf_counter()
    local, lower = tomo.crop(route[offset])
    original = route[offset:] - lower
    inside = np.all((original >= 0) & (original < local.cost.shape), axis=1)
    outside = np.flatnonzero(~inside)
    prefix = original[:outside[0]] if len(outside) else original
    if scenario in ('occluded', 'recovered'):
        # Simulate the first downward visibility break. The strip stays open in
        # the recovered observation and can still obscure farther-away support.
        xyz = local.xyz(prefix)
        down = np.flatnonzero((xyz[:, 2] <= xyz[0, 2] - .08)
                              & (np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1) >= .15))
        if len(down):
            edge = xyz[down[0]]
            direction = edge[:2] - xyz[0, :2]
            direction /= np.linalg.norm(direction)
            xy = np.indices(local.cost.shape[1:]).transpose(1, 2, 0) * local.resolution + local.origin
            ahead = (xy - edge[:2]) @ direction
            mask = (ahead >= (-.02 if scenario == 'occluded' else .65))[None]
            lower_support = local.ground <= xyz[0, 2] - .06
            local.ground[mask & lower_support] = np.nan
            local.cost[mask & lower_support] = np.nan
    graph = SupportGraph(local)
    build_ms = (perf_counter() - started) * 1000.
    ids = graph.lookup(prefix)
    window = expert_prefix(graph, ids, goal, scenario)
    window.build_ms = build_ms
    if len(outside) and window.end == 'goal_reached':
        window.end = 'local_horizon'
    return window


def prediction_window(tomo, route, offset, goal, cloud):
    started = perf_counter()
    xyz = tomo.xyz(route[offset:])
    observed = Tomogram.observed(cloud, xyz[0])
    graph = SupportGraph(observed)
    build_ms = (perf_counter() - started) * 1000.
    xy = np.rint((xyz[:, :2] - observed.origin) / observed.resolution)
    outside = np.flatnonzero(np.any((xy < 0) | (xy >= observed.cost.shape[1:]), axis=1))
    if len(outside):
        xyz = xyz[:outside[0]]
    indices = np.array([graph.match_column(p) for p in xyz], dtype=np.int32)
    window = expert_prefix(graph, indices, np.asarray(goal, dtype=np.float32), 'prediction')
    window.build_ms = build_ms
    if len(outside) and window.end == 'goal_reached':
        window.end = 'local_horizon'
    return window


class Windows:
    def __init__(self, dataset, split, pairs=None):
        self.directory = Path(dataset)
        self.meta, self.routes = read_routes(dataset)
        path = Path(self.meta['tomogram'])
        if fingerprint(path) != self.meta['sha256']:
            raise ValueError(f'Tomogram changed after labeling: {path}')
        self.tomo = Tomogram.read(path)
        self.index = []
        self.versions = {}
        if pairs:
            import json
            pairs = Path(pairs)
            for rec in json.loads(pairs.read_text())['frames']:
                rid, offset = int(rec['route']), int(rec['offset'])
                if not 0 <= rid < len(self.routes) or not 0 <= offset < len(self.routes[rid]):
                    raise ValueError(f'Paired frame references an invalid expert route index: {rec}')
                if rec['frame'] != 'world':
                    raise ValueError('Aligned predicted snapshots must use the VST world frame')
                if self.meta['routes'][rid]['split'] == split:
                    cloud = pairs.parent / rec['cloud']
                    self.versions[len(self.index)] = str(rec.get('version') or cloud.stem)
                    self.index.append((rid, offset, 'prediction', cloud))
            return
        for rec, route in zip(self.meta['routes'], self.routes):
            if rec['split'] != split:
                continue
            spatial = self.tomo.xyz(route)
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(spatial[:, :2], axis=0), axis=1))]
            offsets = np.unique(length.searchsorted(np.arange(0., length[-1], 1.2)))[:8]
            # Include a near-goal window so the learner sees genuine arrival labels.
            offsets = np.unique(np.r_[offsets, len(route) - 2])
            for offset in offsets:
                self.index.append((rec['id'], offset, 'full', None))
            lower = np.flatnonzero(spatial[1:, 2] <= spatial[:-1, 2] - .08)
            for step in lower[:2]:
                offset = max(0, int(step) - 2)
                for scenario in ('occluded', 'recovered'):
                    self.index.append((rec['id'], offset, scenario, None))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rid, offset, scenario, path = self.index[idx]
        route, goal = self.routes[rid], self.meta['routes'][rid]['goal']
        if path is not None:
            return prediction_window(self.tomo, route, offset, goal, read_cloud(path))
        return local_window(self.tomo, route, offset, goal, scenario)


def read_cloud(path):
    if Path(path).suffix == '.npy':
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as arrays:
        if 'xyzc' in arrays:
            return arrays['xyzc']
        for key in ('trav_cost', 'prediction', 'cost'):
            if key in arrays:
                return np.column_stack((arrays['xyz'], arrays[key]))
        raise ValueError(f'No traversability cost field in {path}')
