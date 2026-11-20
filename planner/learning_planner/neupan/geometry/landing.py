"""Continuous, center-connected support and low-cost area around landing targets."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from planner.learning_planner.neupan.geometry.tomogram import Tomogram

RADIUS = .15
MIN_AREA = .04
HEIGHT_TOLERANCE = .075
GRADE_LIMIT = .4


@dataclass
class Patches:
    center_observed: np.ndarray
    area: np.ndarray
    center_cost: np.ndarray
    support_area: np.ndarray
    support_cells: list[np.ndarray] | None = None


def _sector(x, y, radius):
    """Oriented area of the disk in a rectangle from (0,0) to (x,y)."""
    ax, ay = np.minimum(np.abs(x), radius), np.minimum(np.abs(y), radius)
    turn = np.minimum(ax, np.sqrt(np.maximum(radius ** 2 - ay ** 2, 0.)))

    def primitive(t):
        return (t * np.sqrt(np.maximum(radius ** 2 - t ** 2, 0.))
                + radius ** 2 * np.arcsin(np.clip(t / radius, 0., 1.))) / 2.

    return np.sign(x) * np.sign(y) * (turn * ay + primitive(ax) - primitive(turn))


def cell_areas(delta_xy, resolution, radius):
    low = delta_xy - resolution / 2.
    high = delta_xy + resolution / 2.
    return np.maximum(0., _sector(high[..., 0], high[..., 1], radius)
                      - _sector(low[..., 0], high[..., 1], radius)
                      - _sector(high[..., 0], low[..., 1], radius)
                      + _sector(low[..., 0], low[..., 1], radius))


@lru_cache(maxsize=8)
def _stencil(resolution, radius):
    half = int(np.ceil(radius / resolution)) + 1
    side = half * 2 + 1
    offsets = np.indices((side, side)).reshape(2, -1).T - half
    return offsets, side, half


def connected_cells(valid, vertical, horizontal, center):
    found = np.zeros_like(valid)
    found[:, center, center] = valid[:, center, center]
    while True:
        grown = found.copy()
        grown[:, 1:] |= found[:, :-1] & vertical
        grown[:, :-1] |= found[:, 1:] & vertical
        grown[:, :, 1:] |= found[:, :, :-1] & horizontal
        grown[:, :, :-1] |= found[:, :, 1:] & horizontal
        grown &= valid
        if np.array_equal(grown, found):
            return found
        found = grown


def support_patches(tomo, points, radius=RADIUS, height_tolerance=HEIGHT_TOLERANCE, *, include_cells=False):
    """Measure over clipped raster cells, using at most one matching height per XY."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or radius <= 0:
        raise ValueError('Expected finite support XYZ and a positive patch radius')
    offsets, side, half = _stencil(tomo.resolution, radius)
    extent = np.asarray(tomo.cost.shape[1:])
    center = np.rint((points[:, :2] - tomo.origin) / tomo.resolution).astype(int)
    origin_observed = np.zeros(len(points), dtype=bool)
    result = np.zeros(len(points))
    coverage = np.zeros(len(points))
    center_cost = np.full(len(points), np.nan)
    support_cells = [] if include_cells else None
    # Batching bounds the temporary height stack even when labeling a whole tile.
    for start in range(0, len(points), 384):
        stop = min(start + 384, len(points))
        part = points[start:stop]
        location = center[start:stop, None, :] + offsets[None]
        inside = ((location >= 0) & (location < extent)).all(axis=-1)
        safe = np.clip(location, 0, extent - 1)
        scores = tomo.cost[:, safe[..., 0], safe[..., 1]]
        heights = tomo.ground[:, safe[..., 0], safe[..., 1]]
        diff = np.abs(heights - part[None, :, None, 2])
        match = inside[None] & np.isfinite(diff) & (diff <= height_tolerance)
        nearest = np.min(np.where(match, diff, np.inf), axis=0)
        tied = match & (diff <= nearest[None] + 1e-6)
        layer = np.argmin(np.where(tied, np.where(np.isfinite(scores), scores, 1e9), np.inf), axis=0)
        # Keep observed center distinct from the predicted cost gate.
        z = np.take_along_axis(heights, layer[None], axis=0)[0].reshape(-1, side, side)
        cost = np.take_along_axis(scores, layer[None], axis=0)[0].reshape(-1, side, side)
        observed = np.isfinite(nearest).reshape(-1, side, side)
        origin_observed[start:stop] = observed[:, half, half]
        center_cost[start:stop] = np.where(observed[:, half, half], cost[:, half, half], np.nan)
        xy = tomo.origin + location * tomo.resolution
        areas = cell_areas(xy - part[:, None, :2], tomo.resolution, radius).reshape(-1, side, side)
        cheap = cost < .8 if tomo.prediction else cost <= 15.
        solid = observed & (areas > 1e-12)
        valid = solid & np.isfinite(cost) & (cost >= 0) & cheap
        # Four-neighbor growth prevents corner-only contacts from accumulating area.
        dz_limit = np.tan(GRADE_LIMIT) * tomo.resolution + .01
        vertical = np.abs(z[:, 1:] - z[:, :-1]) <= dz_limit
        horizontal = np.abs(z[:, :, 1:] - z[:, :, :-1]) <= dz_limit
        geometry = connected_cells(solid, vertical, horizontal, half)
        low_cost = connected_cells(valid, vertical, horizontal, half)
        result[start:stop] = (areas * low_cost).sum(axis=(1, 2))
        coverage[start:stop] = (areas * geometry).sum(axis=(1, 2))
        if support_cells is not None:
            support_cells.extend(cells[keep.ravel()] for cells, keep in zip(location, geometry))
    return Patches(origin_observed, result, center_cost, coverage, support_cells)


def observed_candidates(tomo, start, horizon=1.2):
    cells = np.argwhere(np.isfinite(tomo.ground) & np.isfinite(tomo.cost))
    xyz = tomo.xyz(cells)
    distance = np.linalg.norm(xyz[:, :2] - start[:2], axis=1)
    valid = (distance <= horizon + 1e-7) & (np.linalg.norm(xyz - start, axis=1) >= .15)
    unique = np.unique(np.round(xyz[valid], 5), axis=0, return_index=True)[1]
    return xyz[valid][np.sort(unique)]


def read_raw_map(path):
    with np.load(path, allow_pickle=False) as data:
        return Tomogram(data['cost'], data['ground'], data['origin'], float(data['resolution']))
