"""Local VST-style support slices. Indices everywhere here are (layer, x, y)."""
from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


DIRECTIONS = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy])
STOP = len(DIRECTIONS)
VST_LIMIT = 15.
PREDICTION_LIMIT = .8


@dataclass
class Tomogram:
    cost: np.ndarray
    ground: np.ndarray
    origin: np.ndarray
    resolution: float
    prediction: bool = False

    @classmethod
    def read(cls, path):
        # VST pickle inputs must come from a trusted workspace.
        with Path(path).open('rb') as stream:
            payload = pickle.load(stream)
        data = np.asarray(payload['data'], dtype=np.float32)
        resolution = float(payload['resolution'])
        if data.ndim != 4 or data.shape[0] < 5 or not all(data.shape[1:]) or resolution <= 0:
            raise ValueError('Invalid VST Tomogram dimensions or resolution')
        origin = np.asarray(payload['center'], dtype=float) - (np.array(data.shape[2:]) // 2) * resolution
        return cls(data[0], data[3], origin, resolution)

    @classmethod
    def observed(cls, cloud, start, resolution=.1, height_step=.5, width=6., height=5., grid_phase=None):
        """Project measured XYZ as in VST: highest observed point at or below each slice.

        Carry each measured point's cost with that height, pooling only hits at nearly
        identical heights in the same column. Missing columns remain NaN.
        """
        cloud = np.asarray(cloud, dtype=np.float32)
        start = np.asarray(start, dtype=float)
        if (cloud.ndim != 2 or cloud.shape[1] != 4 or start.shape != (3,)
                or not np.isfinite(start).all()):
            raise ValueError('Expected finite start XYZ and an N x 4 XYZ/trav_cost cloud')
        half_cells = int(np.ceil(width / (2 * resolution)))
        size = 2 * half_cells + 1
        phase = np.zeros(2) if grid_phase is None else np.asarray(grid_phase, dtype=float)
        if phase.shape != (2,) or not np.isfinite(phase).all():
            raise ValueError('Invalid XY grid phase')
        origin = phase + (np.rint((start[:2] - phase) / resolution) - half_cells) * resolution
        bottom, top = start[2] - height / 2, start[2] + height / 2
        cloud = cloud[np.isfinite(cloud).all(axis=1) & (cloud[:, 3] >= 0)
                      & (cloud[:, 3] <= 1) & (cloud[:, 2] >= bottom) & (cloud[:, 2] <= top)]
        xy = np.rint((cloud[:, :2] - origin) / resolution).astype(np.int32)
        inside = np.all((xy >= 0) & (xy < size), axis=1)
        cloud, xy = cloud[inside], xy[inside]
        column = xy[:, 0] * size + xy[:, 1]
        # The phase is anchored in the gravity-aligned frame rather than following base Z.
        levels = np.arange(np.floor(bottom / height_step) + 1, np.ceil(top / height_step) + 1) * height_step
        ground = np.full((len(levels), size * size), -np.inf, dtype=np.float32)
        cost = np.full_like(ground, np.nan)
        for layer, level in enumerate(levels):
            observed = cloud[:, 2] <= level
            np.maximum.at(ground[layer], column[observed], cloud[observed, 2])
            measured = observed & (cloud[:, 2] >= ground[layer, column] - .025)
            count = np.bincount(column[measured], minlength=size * size)
            total = np.bincount(column[measured], weights=cloud[measured, 3], minlength=size * size)
            np.divide(total, count, out=cost[layer], where=count > 0)
        ground[~np.isfinite(ground)] = np.nan
        return cls(cost.reshape(-1, size, size), ground.reshape(-1, size, size), origin, resolution, True)

    @property
    def admitted(self):
        low = self.cost < PREDICTION_LIMIT if self.prediction else self.cost <= VST_LIMIT
        return np.isfinite(self.ground) & np.isfinite(self.cost) & (self.cost >= 0) & low

    @property
    def feature_cost(self):
        return self.cost if self.prediction else np.clip(self.cost / 50., 0., 1.)

    def xyz(self, cells):
        cells = np.asarray(cells, dtype=np.int32).reshape(-1, 3)
        return np.column_stack((self.origin + self.resolution * cells[:, 1:],
                                self.ground[tuple(cells.T)])).astype(np.float32)

    def crop(self, center, width=6.):
        center = np.asarray(center)
        radius = int(np.floor(width / (2 * self.resolution) + 1e-6))
        lower = np.maximum(center[1:] - radius, 0)
        upper = np.minimum(center[1:] + radius + 1, self.cost.shape[1:])
        block = (slice(None), slice(lower[0], upper[0]), slice(lower[1], upper[1]))
        return (Tomogram(self.cost[block].copy(), self.ground[block].copy(),
                         self.origin + self.resolution * lower, self.resolution, self.prediction),
                np.r_[0, lower])

    def cloud(self):
        cells = np.argwhere(np.isfinite(self.ground) & np.isfinite(self.cost))
        return np.column_stack((self.xyz(cells), self.feature_cost[tuple(cells.T)], cells[:, 0]))


class SupportGraph:
    def __init__(self, tomo, step=.30):
        self.tomo = tomo
        keep = tomo.admitted
        self.cells = np.argwhere(keep).astype(np.int32)
        self.xyz = tomo.xyz(self.cells)
        self.ids = np.full(keep.shape, -1, dtype=np.int32)
        self.ids[tuple(self.cells.T)] = np.arange(len(self.cells))
        self.cost = tomo.feature_cost[tuple(self.cells.T)]
        self.neighbors = np.full((len(self.cells), STOP), -1, dtype=np.int32)
        layers, nx, ny = keep.shape
        source_layer, x, y = self.cells.T
        source_z = self.xyz[:, 2]

        # VST DecideLayer(): first strictly cheaper valid support in the same XY
        # column, with |dz| <= 0.2 m; ties follow 0, -1, +1, -2, +2, ...
        active = source_layer.copy()
        best_cost = tomo.cost[tuple(self.cells.T)].copy()
        for shift in range(1, layers):
            for offset in (-shift, shift):
                candidate = source_layer + offset
                layer = candidate.clip(0, layers - 1)
                values = tomo.cost[layer, x, y]
                agree = (abs(tomo.ground[layer, x, y] - source_z) <= .200001)
                better = (candidate == layer) & keep[layer, x, y] & agree & (values < best_cost)
                active[better], best_cost[better] = layer[better], values[better]

        def candidates(layer, xp, yp, inside, dx, dy):
            target_z = tomo.ground[layer, xp, yp]
            valid = inside & keep[layer, xp, yp] & (abs(target_z - source_z) <= step + 1e-6)
            if dx and dy:
                # Require both corners; enforce height compatibility along each two-edge turn.
                for ax, ay in ((xp, y), (x, yp)):
                    corner_z = tomo.ground[layer, ax, ay]
                    valid &= (keep[layer, ax, ay] & (abs(corner_z - source_z) <= step + 1e-6)
                              & (abs(corner_z - target_z) <= step + 1e-6))
            return valid

        for slot, (dx, dy) in enumerate(DIRECTIONS):
            xp, yp = (x + dx).clip(0, nx - 1), (y + dy).clip(0, ny - 1)
            inside = (x + dx == xp) & (y + dy == yp)
            valid = candidates(active, xp, yp, inside, dx, dy)
            self.neighbors[valid, slot] = self.ids[active[valid], xp[valid], yp[valid]]

            # An observed ground hit can be repeated in several VST slices with
            # identical cost. Try another representation when the active one has
            # no valid successor in this direction. The departure height must agree.
            if tomo.prediction:
                for shift in range(1, layers):
                    for offset in (-shift, shift):
                        alt = active + offset
                        layer = alt.clip(0, layers - 1)
                        same = keep[layer, x, y] & (abs(tomo.ground[layer, x, y] - source_z) <= .025)
                        use = ((alt == layer) & same & (self.neighbors[:, slot] < 0)
                               & candidates(layer, xp, yp, inside, dx, dy))
                        self.neighbors[use, slot] = self.ids[layer[use], xp[use], yp[use]]

    def lookup(self, cells):
        cells = np.asarray(cells, dtype=np.int32).reshape(-1, 3)
        valid = np.all((cells >= 0) & (cells < self.ids.shape), axis=1)
        result = np.full(len(cells), -1, dtype=np.int32)
        result[valid] = self.ids[tuple(cells[valid].T)]
        return result

    def match(self, xyz, horizontal=.12, vertical=.15):
        delta = np.abs(self.xyz - xyz)
        distance = np.linalg.norm(delta[:, :2], axis=1)
        choices = np.flatnonzero((distance <= horizontal + 1e-6) & (delta[:, 2] <= vertical))
        if len(choices) == 0:
            return -1
        order = np.lexsort((self.cost[choices], distance[choices], delta[choices, 2]))
        return int(choices[order[0]])

    def match_column(self, xyz, vertical=.075):
        """Transfer a GT point only into the same observed XY column and height."""
        xy = np.rint((np.asarray(xyz)[:2] - self.tomo.origin) / self.tomo.resolution).astype(int)
        if np.any(xy < 0) or np.any(xy >= self.ids.shape[1:]):
            return -1
        choices = self.ids[:, xy[0], xy[1]]
        choices = choices[choices >= 0]
        delta = np.abs(self.xyz[choices, 2] - xyz[2])
        if len(delta) == 0 or delta.min() > vertical:
            return -1
        return int(choices[np.lexsort((self.cost[choices], delta))[0]])
