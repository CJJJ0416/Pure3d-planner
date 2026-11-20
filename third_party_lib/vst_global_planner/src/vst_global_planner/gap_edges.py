"""Convert confirmed CUDA Tomogram gap pairs into the native A* edge table."""

import numpy as np


def grid_centers(payload):
    resolution = payload["resolution"]
    return tuple(
        payload["center"][axis] + resolution * (np.arange(size) - size // 2)
        for axis, size in enumerate(payload["data"].shape[-2:])
    )


def _anchor(ground, ceiling, cost, rim, direction, threshold, headroom):
    """Walk a continuous bank toward its nearest eligible stance cell."""
    layer, x, y = rim
    height = ground[rim]
    while True:
        x, y = x + direction[0], y + direction[1]
        if not (0 <= x < ground.shape[1] and 0 <= y < ground.shape[2]):
            return None
        cell = layer, x, y
        z, c, upper = ground[cell], cost[cell], ceiling[cell]
        if not np.isfinite(z) or abs(z - height) > .025 or (np.isfinite(upper) and upper - z < headroom):
            return None
        if 0 <= c <= threshold:
            return cell


def gap_edge_table(payload, threshold, headroom, cost_map=None):
    """Read the CUDA-confirmed pairs. Optionally score against the original cost map."""
    cost, _, _, ground, ceiling = payload["data"].astype(np.float32)
    if cost_map is not None:
        cost = np.asarray(cost_map, dtype=np.float32)
    limit, resolution = payload.get("gap_max", 0.), payload["resolution"]
    pairs = payload.get("gap_pairs", {"cells": [], "widths_m": []})
    x, y = grid_centers(payload)
    endpoints, weights, records = [], [], {}
    for (layer, ax, ay, bx, by), width in zip(pairs["cells"], pairs["widths_m"]):
        a, b = (int(layer), int(ax), int(ay)), (int(layer), int(bx), int(by))
        direction = tuple(map(int, np.sign(np.subtract(b[1:], a[1:]))))
        opposite = tuple(-step for step in direction)
        start = _anchor(ground, ceiling, cost, a, opposite, threshold, headroom)
        goal = _anchor(ground, ceiling, cost, b, direction, threshold, headroom)
        if start is None or goal is None:
            continue
        vertices = np.array([[x[i], y[j], ground[k, i, j]] for k, i, j in (start, goal)])
        length = float(np.linalg.norm(vertices[1] - vertices[0]))
        # Same-axis crossing. The bound also covers VST's integer-height
        # rounding when the two banks are at slightly different heights.
        flat_steps = abs(goal[1] - start[1]) + abs(goal[2] - start[2])
        z_steps = abs(int(ground[goal] / resolution) - int(ground[start] / resolution))
        heuristic_bound = max(flat_steps, z_steps) + (np.sqrt(2) - 1) * min(flat_steps, z_steps)
        width_penalty = (float(width) / limit) ** 2
        weight = max(length / resolution, heuristic_bound) + np.mean([cost[start], cost[goal]]) + width_penalty
        row = dict(width_m=float(width), rims=[a, b], anchors=[start, goal],
                   rim_costs=[float(cost[a]), float(cost[b])], length_m=length,
                   height_delta_m=float(ground[b] - ground[a]), weight=float(weight))
        for u, v in ((start, goal), (goal, start)):
            endpoints.append(u + v)
            weights.append(weight)
            records[u, v] = row
    return np.asarray(endpoints, dtype=np.int32).reshape(-1, 6), np.asarray(weights, dtype=float), records
