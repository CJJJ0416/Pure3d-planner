"""Full-parameter VST teacher with five configurable teacher overrides.

The full native parameter set is retained. Teacher profiles override
safe_margin, slope_max, step_max, interval_free and standable_ratio. The
student selects clearance_min, slope_max, step_max and stand_ratio, with
teacher interval_free fixed to 0.5 m.
"""
import json
import sys
from pathlib import Path

import cupy as cp
import cv2
import numpy as np

NATIVE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NATIVE_ROOT))
from tomography.kernels import (
    adjustCostByDistanceKernel,
    distanceTransformKernel,
    inflationKernel,
    skeletonDistanceKernel,
    tomographyKernel,
    travKernel,
)

VERSION = "vst_full_parameter_stand_ratio_v2"
DEFAULT_TEACHER_CONFIG = NATIVE_ROOT / "configs/teacher_building.json"


class ConditionedTeacher:
    def __init__(self, coord, resolution=None, slice_dh=None, ground_h=None,
                 teacher_config=DEFAULT_TEACHER_CONFIG, grid=None):
        """Build geometry once, then evaluate many robot conditions."""
        self.config = json.loads(Path(teacher_config).read_text())
        self.trav = self.config["trav"].copy()
        self.resolution = float(self.config["map"]["resolution"] if resolution is None else resolution)
        self.slice_dh = float(self.config["map"]["slice_dh"] if slice_dh is None else slice_dh)
        self.ground_h = float(self.config["map"]["ground_h"] if ground_h is None else ground_h)
        self.cost_scale = float(self.trav["cost_barrier"])
        lower, upper = coord.min(0), coord.max(0)
        self.center = ((lower[:2] + upper[:2]) * 0.5).astype(np.float32)
        self.shape_xy = np.ceil((upper[:2] - lower[:2]) / self.resolution).astype(int) + 4
        self.slice_h0 = self.ground_h + self.slice_dh
        count = int(np.ceil((upper[2] - self.ground_h) / self.slice_dh))
        # Evaluation can hold the native grid fixed across GT/input/repair.
        # Existing teacher callers retain their original automatic bounds.
        if grid is not None:
            self.center = np.asarray(grid['center'], dtype=np.float32)
            self.shape_xy = np.asarray(grid['shape'][1:], dtype=int)
            self.slice_h0 = float(grid['slice_h0'])
            count = int(grid['shape'][0])
        shape = (count, *self.shape_xy)
        ground = cp.full(shape, -1e6, dtype=cp.float32)
        ceiling = cp.full(shape, 1e6, dtype=cp.float32)
        kernel = tomographyKernel(
            self.resolution, int(shape[1]), int(shape[2]), count, self.slice_h0, self.slice_dh
        )
        kernel(cp.asarray(coord, dtype=cp.float32), cp.asarray(self.center), ground, ceiling, size=len(coord))
        self.valid_ground = ground > -1e6
        self.ground = cp.where(self.valid_ground, ground, cp.nan).get()
        self.ceiling = cp.where(ceiling < 1e6, ceiling, cp.nan).get()
        self.interval = ceiling - ground
        dx = cp.maximum((ground[:, 1:-1] - ground[:, :-2]) ** 2,
                        (ground[:, 1:-1] - ground[:, 2:]) ** 2)
        dy = cp.maximum((ground[:, :, 1:-1] - ground[:, :, :-2]) ** 2,
                        (ground[:, :, 1:-1] - ground[:, :, 2:]) ** 2)
        self.grad_sq = cp.zeros(shape, dtype=cp.float32)
        self.grad_max = cp.zeros(shape, dtype=cp.float32)
        self.grad_sq[:, 1:-1, 1:-1] = dx[:, :, 1:-1] + dy[:, 1:-1, :]
        self.grad_max[:, 1:-1, 1:-1] = cp.maximum(dx[:, :, 1:-1], dy[:, 1:-1, :])

    def parameters(self, condition):
        trav = self.trav.copy()
        trav.update(zip(["safe_margin", "slope_max", "step_max", "interval_free", "standable_ratio"],
                        map(float, condition)))
        return dict(map=dict(resolution=self.resolution, ground_h=self.ground_h, slice_dh=self.slice_dh),
                    trav=trav)

    def cost(self, condition):
        trav = self.parameters(condition)["trav"]
        clearance, inflation = trav["safe_margin"], trav["inflation"]
        half_trav = int(trav["kernel_size"] / 2)
        standable_th = int(trav["standable_ratio"] * (2 * half_trav + 1) ** 2) - 1
        nx, ny = map(int, self.shape_xy)
        raw = cp.zeros_like(self.grad_sq)
        kernel = travKernel(
            nx, ny, half_trav, trav["interval_min"], trav["interval_free"],
            trav["step_max"], 1.2 * self.resolution * np.tan(trav["slope_max"]),
            standable_th, self.cost_scale,
        )
        kernel(self.interval, self.grad_sq, self.grad_max, raw, size=raw.size)
        raw[~self.valid_ground] = self.cost_scale
        # Keep the native window calculation, including its integer truncation.
        half = int((clearance + inflation) / self.resolution)
        distance = cp.zeros_like(raw)
        distanceTransformKernel(nx, ny, half, self.cost_scale)(raw, distance, size=raw.size)

        # Same morphological skeleton and skeleton-width weighting as native VST.
        raw_cpu, distance_cpu = raw.get(), distance.get()
        skeleton = np.zeros(raw.shape, dtype=np.float32)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for layer, values in enumerate(raw_cpu):
            image = (values < self.cost_scale - 0.001).astype(np.uint8) * 255
            medial = np.zeros_like(image)
            while cv2.countNonZero(image):
                eroded = cv2.erode(image, element)
                medial |= cv2.subtract(image, cv2.dilate(eroded, element))
                image = eroded
            mask = medial > 0
            # Native VST requires at least three skeleton pixels in a layer.
            if np.count_nonzero(mask) >= 3:
                skeleton[layer, mask] = distance_cpu[layer, mask]
        local_width = cp.zeros_like(raw)
        skeletonDistanceKernel(nx, ny, half * 2)(
            cp.asarray(skeleton), local_width, size=raw.size
        )
        adjusted = cp.empty_like(raw)
        adjustCostByDistanceKernel(nx, ny, clearance, self.cost_scale, self.resolution,
                                   p=trav["adjust_cost_p"])(
            raw, distance, local_width, adjusted, size=raw.size
        )
        index = np.arange(2 * half + 1) - half
        dist = np.sqrt((self.resolution * index[:, None]) ** 2
                       + (self.resolution * index[None, :]) ** 2)
        table = cp.asarray(np.clip(1 - (dist - inflation) / (clearance + self.resolution), 0, 1),
                           dtype=cp.float32)
        inflated = cp.zeros_like(raw)
        inflationKernel(nx, ny, half)(adjusted, table, local_width, inflated, size=raw.size)
        inflated[~self.valid_ground] = self.cost_scale
        # Export every geometric slice for supervision. Native Tomogram merging
        # produces the point cloud afterwards, without cost-dependent slice pruning.
        return inflated.get()

    def metadata(self):
        return dict(
            teacher_version=VERSION, cost_scale=self.cost_scale,
            resolution=self.resolution, slice_dh=self.slice_dh,
            slice_h0=self.slice_h0, ground_h=self.ground_h,
            center=self.center.tolist(), shape=list(self.ground.shape),
            map=dict(resolution=self.resolution, ground_h=self.ground_h, slice_dh=self.slice_dh),
            trav=self.trav.copy(), teacher_parameters=self.parameters([
                self.trav["safe_margin"], self.trav["slope_max"],
                self.trav["step_max"], self.trav["interval_free"], self.trav["standable_ratio"]]),
            slice_export="all geometric slices before native cost-dependent simplification",
        )

    def payload(self, condition, raw):
        """Native VST pickle layout: cost, gradient_x, gradient_y, ground, ceiling."""
        grad_x, grad_y = np.zeros_like(raw), np.zeros_like(raw)
        grad_x[:, 1:-1] = raw[:, 2:] - raw[:, :-2]
        grad_y[:, :, 1:-1] = raw[:, :, 2:] - raw[:, :, :-2]
        return dict(
            data=np.stack((raw, grad_x, grad_y, self.ground, self.ceiling)).astype(np.float16),
            center=self.center, resolution=self.resolution,
            slice_h0=self.slice_h0, slice_dh=self.slice_dh,
            condition=list(condition), cost_scale=self.cost_scale, teacher_version=VERSION,
            teacher_parameters=self.parameters(condition),
            channel_order=["cost", "gradient_x", "gradient_y", "ground", "ceiling"],
        )
