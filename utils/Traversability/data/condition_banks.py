"""Condition banks consumed by the production slice-label generator."""

from itertools import combinations
import numpy as np
from scipy.ndimage import uniform_filter
from utils.Traversability.data.teacher_conditions import FIXED

NAMES = ["clearance_min", "slope_max", "step_max", "stand_ratio"]


def nine_values(lo, hi, center, offsets):
    """在 [lo,hi] 内生成 9 个去重后的 float32 条件值，优先围绕 center 取偏移。"""
    candidates = (
        list(np.clip(np.r_[lo, hi, center + np.asarray(offsets)], lo, hi))
        if center is not None
        else list(np.linspace(lo, hi, 9))
    )
    values = list(dict.fromkeys(float(np.float32(v)) for v in candidates))
    for value in np.linspace(lo, hi, 65):
        if len(values) >= 9:
            break
        value = float(np.float32(value))
        if value not in values:
            values.append(value)
    return sorted(values[:9])


def dominant(values, lo, hi):
    """返回 (lo,hi) 内出现次数最多的数值（按 4 位小数聚类），无有效值返回 None。"""
    v = values[np.isfinite(values) & (values > lo) & (values < hi)]
    if not len(v):
        return None
    unique, count = np.unique(np.round(v, 4), return_counts=True)
    return float(unique[np.argmax(count)])


def condition_banks(teacher, seed, ranges, region, training):
    """生成两个条件 bank 及 nominal/单参数 profile 列，供切片标签生成使用。"""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 44401]))
    profiles = [
        dict(id=name, condition=c)
        for name, c in zip(["nominal", "low_stand", "strict_stand", "relaxed"], FIXED)
    ]
    banks = []
    grad = teacher.grad_sq.get()
    maximum = teacher.grad_max.get()
    valid = teacher.valid_ground.get() & (teacher.interval.get() >= 0.5) & region
    pairs = [list(combinations(range(4 + 9 * p, 13 + 9 * p), 2)) for p in range(4)]
    for bank in range(2):
        anchor = np.array([0.5, 0.6, 0.4, 0.05]) if bank == 0 else rng.uniform(ranges[:, 0], ranges[:, 1])
        flat = (1.2 * teacher.resolution * np.tan(anchor[1])) ** 2
        step_values = np.sqrt(maximum[valid & (grad > flat)])
        slope_values = np.arctan(np.sqrt(grad[valid]) / (1.2 * teacher.resolution))
        counts = np.rint(
            uniform_filter((grad < flat).astype(np.float32), size=(1, 7, 7), mode="constant") * 49
        )
        step_region = valid & (grad > flat) & (maximum <= anchor[2] ** 2)
        thresholds = (counts[step_region] + 2) / 49
        centers = [
            None,
            dominant(slope_values, *ranges[1]),
            dominant(step_values, *ranges[2]),
            dominant(thresholds, *ranges[3]),
        ]
        offsets = [
            None,
            (
                [-0.08, -0.04, -0.01, 0, 0.01, 0.04, 0.08]
                if training
                else [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
            ),
            (
                [-0.04, -0.02, -0.005, 0, 0.005, 0.02, 0.04]
                if training
                else [-0.06, -0.03, -0.01, 0, 0.01, 0.03, 0.06]
            ),
            (
                np.array([-1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25]) / 49
                if training
                else np.array([-1.1, -0.6, -0.2, 0, 0.2, 0.6, 1.1]) / 49
            ),
        ]
        columns = list(range(4))
        for p, column in enumerate([0, 1, 2, 4]):
            lo, hi = ranges[p]
            values = np.linspace(lo, hi, 9) if p == 0 else nine_values(lo, hi, centers[p], offsets[p])
            for i, value in enumerate(values):
                c = np.array([anchor[0], anchor[1], anchor[2], 0.5, anchor[3]])
                c[column] = float(value)
                columns.append(len(profiles))
                profiles.append(
                    dict(
                        id=f"bank{bank}_{NAMES[p]}_{i}",
                        condition=c.tolist(),
                        bank=bank,
                        parameter=p,
                        value=float(value),
                        critical_value=centers[p],
                    )
                )
        banks.append(columns)
    return profiles, banks, pairs
