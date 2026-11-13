"""工作空间配置、体素采样、空间分块与旧 VST 查表工具。"""
import json
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONDITION_NAMES = ["clearance_min", "slope_max", "step_max", "interval_free", "stand_ratio"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def workspace_path(path):
    """Configuration paths are relative to the workspace, never the shell cwd."""
    return (ROOT / path).resolve()


def voxel_downsample(coord, voxel_size):
    """One centroid per occupied world-aligned voxel. XYZ remains in metres."""
    grid = np.floor(coord.astype(np.float64) / voxel_size).astype(np.int64)
    grid -= grid.min(0)
    key = np.ravel_multi_index(grid.T, tuple(grid.max(0) + 1))
    del grid
    _, inverse = np.unique(key, return_inverse=True)
    count = np.bincount(inverse)
    result = np.empty((len(count), 3), dtype=np.float32)
    for axis in range(3):
        result[:, axis] = np.bincount(inverse, weights=coord[:, axis]) / count
    return result


def spatial_blocks(coord, size_m, context_m):
    """Disjoint 3-D cores, overlapping context. Every point belongs to one core."""
    size = np.asarray(size_m, dtype=np.float64)
    origin = np.floor(coord.min(0) / size) * size
    cells = np.floor((coord - origin) / size).astype(np.int64)
    key = np.ravel_multi_index(cells.T, tuple(cells.max(0) + 1))
    order = np.argsort(key, kind="stable")
    _, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    groups = {
        tuple(cells[order[start]]): order[start:start + count]
        for start, count in zip(starts, counts)
    }
    radius = np.ceil(context_m / size).astype(int)
    neighbours = list(product(*(range(-r, r + 1) for r in radius)))
    for cell, core_indices in groups.items():
        cell = np.asarray(cell)
        lower = origin + cell * size
        upper = lower + size
        candidates = np.concatenate([
            groups[neighbour] for offset in neighbours
            if (neighbour := tuple(cell + offset)) in groups
        ])
        local = coord[candidates]
        indices = candidates[((local >= lower - context_m) & (local < upper + context_m)).all(1)]
        core = ((coord[indices] >= lower) & (coord[indices] < upper)).all(1)
        yield indices, core, lower


def query_vst(coord, cost_raw, ground, center, resolution, chunk_size=65536):
    """Closest finite ground at native XY, minimum cost for equal-height layers.

    This labels every input point with the cost of its nearest ground layer.
    It adds no surface-match, continuity or ray-coverage mask. Unknown is NaN.
    """
    shape = np.asarray(cost_raw.shape[1:])
    relative = (coord[:, :2] - np.asarray(center, dtype=np.float32)) / np.float32(resolution)
    # Native tomography utils_point uses C round(): half ties away from zero.
    grid = (np.sign(relative) * np.floor(np.abs(relative) + 0.5)).astype(np.int64) + shape // 2
    inside = ((grid >= 0) & (grid < shape)).all(1)
    result = np.full(len(coord), np.nan, dtype=np.float32)
    for first in range(0, len(coord), chunk_size):
        last = min(first + chunk_size, len(coord))
        x, y = np.clip(grid[first:last], 0, shape - 1).T
        values = np.asarray(cost_raw[:, x, y], dtype=np.float32)
        heights = np.asarray(ground[:, x, y], dtype=np.float32)
        valid = np.isfinite(values) & np.isfinite(heights)
        distance = np.where(valid, np.abs(heights - coord[None, first:last, 2]), np.inf)
        nearest = valid & (distance <= distance.min(0)[None] + 0.001)
        layer = np.argmin(np.where(nearest, values, np.inf), axis=0)
        selected = values[layer, np.arange(last - first)]
        result[first:last] = np.where(inside[first:last] & valid.any(0), selected, np.nan)
    return result
