"""Public, ROS-independent interface for the learned local path component."""
from dataclasses import dataclass
from math import atan2
from time import perf_counter

import numpy as np
import torch

from planner.learning_planner.neupan.models.support_policy import SupportPolicy, decode
from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph, Tomogram


def reached(position, goal):
    return bool(goal_mask(position, goal))


@dataclass
class PathResult:
    node_chain: np.ndarray
    support_xyz: np.ndarray
    reason: str
    cost_version: str
    build_ms: float
    decode_ms: float

    @property
    def length_m(self):
        return float(np.linalg.norm(np.diff(self.support_xyz[:, :2], axis=0), axis=1).sum())

    def body_path(self, height=.4):
        """A path usable by SCAN. A stationary result is represented by an empty path."""
        if len(self.support_xyz) < 2:
            return np.empty((0, 3), dtype=np.float32)
        result = self.support_xyz.copy()
        result[:, 2] += height
        return result


@dataclass
class LocalGoal:
    support_xyz: np.ndarray
    support_cell: np.ndarray
    yaw: float
    approach_m: float
    delta_z_m: float
    reason: str


class PathGenerator:
    def __init__(self, checkpoint, device='cpu'):
        bundle = torch.load(checkpoint, map_location=device, weights_only=True)
        if bundle['schema'] not in (1, 2):
            raise ValueError('Unsupported planner.learning_planner checkpoint schema')
        self.model = SupportPolicy(schema=bundle['schema']).to(device)
        self.model.load_state_dict(bundle['model'])
        self.model.stop_logit_threshold = float(bundle.get('stop_logit_threshold', 0.))
        self.model.eval()

    def plan(self, local, start, goal, *, cost_version='', horizon=3., max_steps=64):
        """Plan from finite support-frame XYZ in the same frame as local cost data.

        local is a Tomogram or an N x 4 measured XYZ/trav_cost array. Supply the
        goal's support height and the robot's estimated support position.
        """
        stamp = perf_counter()
        start, goal = np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32)
        if (start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all()
                or not np.isfinite(horizon) or horizon <= 0 or max_steps <= 0):
            raise ValueError('Expected finite support and goal XYZ and positive planning limits')
        tomo = local if isinstance(local, Tomogram) else Tomogram.observed(local, start)
        graph = SupportGraph(tomo)
        return self.plan_graph(graph, start, goal, cost_version=cost_version, horizon=horizon,
                               max_steps=max_steps, build_ms=(perf_counter() - stamp) * 1000.)

    def plan_graph(self, graph, start, goal, *, cost_version='', horizon=3., max_steps=64, build_ms=0.):
        before = perf_counter()
        node = graph.match(start)
        chain, reason = decode(self.model, graph, node, goal, horizon, max_steps)
        ids = np.asarray(chain, dtype=np.int32)
        return PathResult(graph.cells[ids], graph.xyz[ids], reason, str(cost_version),
                          build_ms, (perf_counter() - before) * 1000.)

    def plan_candidates_graph(self, graph, start, goal, *, cost_version='', horizon=3., max_steps=64,
                              count=3, width=64, build_ms=0.):
        """Return up to count geometrically distinct, observed-support paths."""
        from planner.learning_planner.neupan.inference.beam import decode_beam

        before = perf_counter()
        matches = decode_beam(self.model, graph, graph.match(start), goal, horizon, max_steps,
                              count=count, width=width)
        elapsed = (perf_counter() - before) * 1000.
        results = []
        for match in matches:
            ids = np.asarray(match.nodes, dtype=np.int32)
            results.append(PathResult(graph.cells[ids], graph.xyz[ids], match.reason, str(cost_version),
                                      build_ms, elapsed))
        return results

    def local_goals(self, graph, start, goal, *, lookahead=1.2, count=3, width=64, separation=.30):
        """Select support-height waypoints without exposing or dispatching a path."""
        from planner.learning_planner.neupan.inference.beam import select_local_goals

        matches = select_local_goals(self.model, graph, graph.match(start), goal,
                                    lookahead=lookahead, count=count, width=width, separation=separation)
        results = []
        for state in matches:
            node = state.nodes[-1]
            pos = graph.xyz[node]
            dx, dy = pos[:2] - graph.xyz[state.nodes[-2], :2]
            results.append(LocalGoal(pos.copy(), graph.cells[node].copy(), atan2(dy, dx),
                                     state.length, float(pos[2] - graph.xyz[state.nodes[0], 2]), state.reason))
        return results
