"""Directed, observed-support reachability for a bounded local decode."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def goal_mask(xyz, goal):
    horizontal = np.linalg.norm(xyz[..., :2] - goal[:2], axis=-1)
    vertical = np.abs(xyz[..., 2] - goal[2])
    return (horizontal <= .14) & (vertical <= .12)


class GoalReachability:
    def __init__(self, graph, goal, horizon, max_steps):
        goals = np.flatnonzero(goal_mask(graph.xyz, goal))
        self.distance = None
        self.step_distance = None
        if not len(goals):
            return
        rows, slots = np.where(graph.neighbors >= 0)
        cols = graph.neighbors[rows, slots]
        lengths = np.linalg.norm(graph.xyz[cols, :2] - graph.xyz[rows, :2], axis=1).astype('float64')
        shape = (len(graph.xyz), len(graph.xyz))
        reverse = csr_matrix((lengths, (cols, rows)), shape=shape)
        self.distance = dijkstra(reverse, indices=goals, min_only=True)
        # With the default 3 m/64 steps, every 0.1 m grid path within the
        # distance budget also fits the step budget. Handle tighter caller limits.
        if len(lengths) and lengths.min() * (max_steps + 1) <= horizon + 1e-5:
            values = np.full(shape[0], np.inf)
            values[goals] = 0.
            by_steps = [values]
            for _ in range(max_steps):
                new = values.copy()
                np.minimum.at(new, rows, lengths + values[cols])
                values = new
                by_steps.append(values)
            self.step_distance = by_steps

    def distances(self, remaining_steps):
        if self.step_distance is not None:
            return self.step_distance[remaining_steps]
        return self.distance


def frontier_progress(position, candidates, goal):
    """Allow a local prefix toward a goal whose support component is unknown."""
    height_error = abs(position[2] - goal[2])
    next_height_error = np.abs(candidates[:, 2] - goal[2])
    planar_error = np.linalg.norm(position[:2] - goal[:2])
    next_planar_error = np.linalg.norm(candidates[:, :2] - goal[:2], axis=1)
    closer_xy = next_planar_error < planar_error - .001
    if height_error <= .12:
        return closer_xy & (next_height_error <= .12)
    closer_height = next_height_error < height_error - .025
    level_approach = closer_xy & (next_height_error <= height_error + .025)
    return closer_height | level_approach
