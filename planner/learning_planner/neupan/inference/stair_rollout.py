"""Offline local-goal rollout on a VST map or a fixed observed cost snapshot."""
from time import perf_counter

import numpy as np

from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph, Tomogram


def rollout_local_goals(planner, tomogram, start_cell, goal, lookahead=1.2, width=64, max_goals=20):
    def graph_at(point):
        xy = np.rint((point[:2] - tomogram.origin) / tomogram.resolution).astype(int)
        local, _ = tomogram.crop(np.r_[0, xy])
        return SupportGraph(local)

    return _rollout(planner, graph_at, tomogram.xyz([start_cell])[0], goal, lookahead, width, max_goals)


def rollout_observed_goals(planner, cloud, start, goal, lookahead=1.2, width=64, max_goals=20):
    def graph_at(point):
        return SupportGraph(Tomogram.observed(cloud, point))

    return _rollout(planner, graph_at, start, goal, lookahead, width, max_goals)


def _rollout(planner, graph_at, position, goal, lookahead, width, max_goals):
    if not np.isfinite(position).all():
        return [], 'no_start_support', 0., 0.
    visited = [position.copy()]
    selected = []
    graph_ms = 0.
    search_ms = 0.
    reason = 'local_goal_limit'
    for _ in range(max_goals):
        if goal_mask(position, goal):
            reason = 'goal_reached'
            break
        before = perf_counter()
        graph = graph_at(position)
        graph_ms += (perf_counter() - before) * 1000.
        before = perf_counter()
        options = planner.local_goals(graph, position, goal, lookahead=lookahead, width=width)
        search_ms += (perf_counter() - before) * 1000.
        # Require actual motion between goals and avoid revisiting an old foothold.
        choice = next((candidate for candidate in options if not
                       (np.linalg.norm(np.asarray(visited) - candidate.support_xyz, axis=1) < .12).any()), None)
        if choice is None:
            reason = 'no_local_goal' if not options else 'revisited_goal'
            break
        position = choice.support_xyz
        visited.append(position)
        selected.append(choice)
    if goal_mask(position, goal):
        reason = 'goal_reached'
    return selected, reason, graph_ms, search_ms
