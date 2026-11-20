"""Area-supervised planning on an observed layered support graph."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn
from torch.nn import functional as F

from planner.learning_planner.neupan.geometry.landing import support_patches
from planner.learning_planner.neupan.data.landing_dataset import HORIZON
from planner.learning_planner.neupan.inference.landing_route import LandingRoutes, MIN_COST_PROGRESS, tree_total
from planner.learning_planner.neupan.inference.reachability import goal_mask

MAX_POINTS = 3
MAX_LENGTH = 3.
COST_UNIT = 10.
CHECKPOINT_TYPE = 'support_sequence_v1'


def patches(tomo, xyz):
    """Raw geometry only. Fifty channels encode occupancy and relative height."""
    offsets = np.indices((5, 5)).reshape(2, -1).T - 2
    ij = np.rint((xyz[:, :2] - tomo.origin) / tomo.resolution).astype(int)[:, None] + offsets
    inside = ((ij >= 0) & (ij < tomo.ground.shape[1:])).all(axis=-1)
    x, y = np.moveaxis(np.clip(ij, 0, np.array(tomo.ground.shape[1:]) - 1), -1, 0)
    delta = tomo.ground[:, x, y] - xyz[None, :, 2, None]
    nearest = np.argmin(np.where(np.isfinite(delta), abs(delta), np.inf), axis=0)
    height = np.take_along_axis(delta, nearest[None], axis=0)[0]
    observed = inside & np.isfinite(height) & (abs(height) <= .30)
    return np.column_stack((observed, np.where(observed, height / .30, 0))).astype(np.float32)


@dataclass
class Actions:
    ids: np.ndarray
    lengths: np.ndarray
    costs: np.ndarray
    parents: np.ndarray
    geometric_parents: np.ndarray
    weighted: np.ndarray

    def chain(self, index):
        node = self.ids[index]
        parent = self.parents if self.weighted[node] else self.geometric_parents
        nodes = [node]
        while parent[node] >= 0:
            node = parent[node]
            nodes.append(node)
        return np.asarray(nodes[::-1], dtype=np.int32)


class SequenceGraph:
    def __init__(self, tomo, goal):
        self.tomo = tomo
        self.routes = LandingRoutes(tomo)
        self.xyz = self.routes.xyz
        self.cost = ((self.routes.scale - 1.) / 4.).astype(np.float32)
        self.goal = np.asarray(goal)
        self.goals = goal_mask(self.xyz, goal)
        self.remaining = np.full(len(self.xyz), np.inf)
        if self.goals.any():
            self.remaining = dijkstra(self.routes.weights.T.tocsr(), indices=np.flatnonzero(self.goals), min_only=True)

    def source(self, point):
        node = self.routes.graph.match_column(point)
        return int(self.routes.identity[node]) if node >= 0 else -1

    def inputs(self):
        edges = self.routes.weights.tocoo()
        src, dst = edges.row, edges.col
        delta = self.xyz[dst] - self.xyz[src]
        edge = np.column_stack((delta, edges.data)).astype(np.float32)
        node = np.column_stack(((self.goal - self.xyz) / 4., self.cost)).astype(np.float32)
        return dict(geometry=patches(self.tomo, self.xyz), node=node,
                    src=src.astype(np.int64), dst=dst.astype(np.int64), edge=edge)

    def actions(self, source, budget=HORIZON, visited=()):
        budget = min(HORIZON, max(0., budget))
        routes = self.routes
        dist, geom_parent = dijkstra(routes.lengths, indices=source, limit=budget + 1e-5,
                                     return_predecessors=True)
        weight, parent = dijkstra(routes.weights, indices=source, limit=budget * routes.scale.max() + 1e-5,
                                  return_predecessors=True)
        weighted_length = tree_total(self.xyz, parent, np.argsort(weight))
        cheap = np.isfinite(weight) & (weighted_length <= budget + 1e-5)
        length = np.where(cheap, weighted_length, dist)
        valid = np.isfinite(length) & (length >= .14999)
        if np.isfinite(self.remaining[source]):
            valid &= self.remaining[source] - self.remaining >= MIN_COST_PROGRESS
        for prior in visited:
            prior = np.asarray(prior)
            close = np.linalg.norm(self.xyz[:, :2] - prior[:2], axis=1) < .10
            valid &= ~(close & (abs(self.xyz[:, 2] - prior[2]) < .08))
        ids = np.flatnonzero(valid)
        geometric_cost = tree_total(self.xyz, geom_parent, np.argsort(dist), routes.scale)
        return Actions(ids, length[ids], np.where(cheap, weight, geometric_cost)[ids], parent, geom_parent, cheap)


class MessageLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(68, 64), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU())

    def forward(self, x, src, dst, edge):
        messages = self.message(torch.cat((x[src], edge), dim=1))
        count = x.new_zeros((len(x), 1)).index_add_(0, dst, x.new_ones((len(dst), 1)))
        pooled = torch.zeros_like(x).index_add_(0, dst, messages) / count.clamp_min(1)
        return x + self.update(torch.cat((x, pooled), dim=1))


class SequenceNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry = nn.Sequential(nn.Linear(50, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.cost = nn.Sequential(nn.Linear(4, 64), nn.SiLU())
        self.layers = nn.ModuleList(MessageLayer() for _ in range(3))
        self.area = nn.Linear(32, 1)
        self.value = nn.Linear(64, 2)
        self.history = nn.GRUCell(96, 64)
        self.pointer = nn.Sequential(nn.Linear(96 + 64 + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        self.stop = nn.Sequential(nn.Linear(96 + 64 + 1, 64), nn.SiLU(), nn.Linear(64, 1))

    def encode(self, geometry, node, src, dst, edge):
        geom = self.geometry(geometry)
        cost = self.cost(node)
        for layer in self.layers:
            cost = layer(cost, src, dst, edge)
        value = self.value(cost)
        return torch.cat((geom, cost), dim=1), self.area(geom).flatten(), F.softplus(value[:, 0]), value[:, 1]

    def step(self, encoded, source, history, ids, edge, budget):
        history = self.history(encoded[source], history)
        logits = self.pointer(torch.cat((encoded[ids], history.expand(len(ids), -1), edge), dim=1)).flatten()
        stop = self.stop(torch.cat((encoded[source], history, encoded.new_tensor([budget / MAX_LENGTH]))))
        return history, logits, stop[0]


def tensors(data, device):
    return {key: torch.as_tensor(value, device=device) for key, value in data.items()}


@dataclass
class SupportPoint:
    xyz: np.ndarray
    approach_xyz: np.ndarray
    approach_m: float
    approach_cost: float
    trav_cost: float
    coverage_m2: float
    estimated_coverage_m2: float
    score: float


@dataclass
class SupportSequence:
    points: list[SupportPoint]
    reason: str


class SequencePlanner:
    def __init__(self, checkpoint, device='cpu'):
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state['type'] != CHECKPOINT_TYPE:
            raise ValueError('Expected a support-sequence checkpoint')
        self.device = device
        self.model = SequenceNetwork().to(device).eval()
        self.model.load_state_dict(state['model'])
        self.stop_threshold = state['stop_threshold']
        self.connect_threshold = state['connect_threshold']

    @torch.inference_mode()
    def plan(self, tomo, start, goal, *, visited=()):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
            raise ValueError('Expected finite support XYZ and goal XYZ')
        if goal_mask(start, goal):
            return SupportSequence([], 'goal_reached')
        graph = SequenceGraph(tomo, goal)
        source = graph.source(start)
        if source < 0:
            return SupportSequence([], 'no_start_support')
        encoded, area, value, connectivity = self.model.encode(**tensors(graph.inputs(), self.device))
        history = encoded.new_zeros(64)
        value = value.cpu().numpy() * COST_UNIT
        connected = connectivity.sigmoid().cpu().numpy()
        budget = MAX_LENGTH
        result = []
        past = [*visited, start]
        for _ in range(MAX_POINTS):
            actions = graph.actions(source, budget, past)
            if not len(actions.ids):
                return SupportSequence(result, 'horizon_reached' if budget < .15 else 'no_confirmed_progress')
            edge = np.column_stack((actions.lengths / HORIZON, actions.costs / COST_UNIT)).astype(np.float32)
            history, logits, stop = self.model.step(encoded, source, history, actions.ids,
                                                      torch.as_tensor(edge, device=self.device), budget)
            allowed = np.ones(len(actions.ids), dtype=bool)
            if not np.isfinite(graph.remaining[source]):
                allowed &= connected[actions.ids] >= self.connect_threshold
                allowed &= value[source] - value[actions.ids] >= MIN_COST_PROGRESS
                if stop.sigmoid().item() >= self.stop_threshold:
                    return SupportSequence(result, 'no_confirmed_progress')
            if not allowed.any():
                return SupportSequence(result, 'no_confirmed_progress')
            scores = logits.cpu().numpy()
            index = int(np.argmax(np.where(allowed, scores, -np.inf)))
            source = int(actions.ids[index])
            xyz = graph.xyz[source]
            # Area is evaluated only after selection.
            coverage = support_patches(tomo, [xyz]).support_area[0]
            result.append(SupportPoint(xyz, graph.xyz[actions.chain(index)], float(actions.lengths[index]),
                                       float(actions.costs[index]), float(graph.cost[source]), float(coverage),
                                       float(area[source].sigmoid().item() * np.pi * .15**2), float(scores[index])))
            past.append(xyz)
            budget -= actions.lengths[index]
            if graph.goals[source]:
                return SupportSequence(result, 'goal_reached')
        return SupportSequence(result, 'horizon_reached')
