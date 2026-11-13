"""Seeded robot profiles and explicit single-parameter pairs for FiLM supervision."""
from copy import deepcopy
import numpy as np

STUDENT_COLUMNS = [0, 1, 2, 4]
RANGES = np.array([[.2, 1.], [.18, .65], [.08, .4], [.05, .5]])
FIXED = [[.5, .4, .3, .5, .1], [.5, .4, .3, .5, .05],
         [.5, .4, .3, .5, .3], [.5, .6, .4, .5, .05]]


def pairs_for(conditions):
    values = np.asarray(conditions)[:, STUDENT_COLUMNS]
    pairs = [[] for _ in STUDENT_COLUMNS]
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            changed = np.flatnonzero(values[a] != values[b])
            if len(changed) == 1:
                pairs[changed[0]].append([a, b])
    return pairs


def legacy_profiles(robots, seed):
    profiles = deepcopy(robots["profiles"])
    baseline = np.asarray(profiles[0]["condition"])
    count = robots["random_profiles"]
    rng = np.random.default_rng(seed)
    ranges = np.asarray(robots["random_ranges"])
    samples = np.column_stack([(rng.permutation(count) + rng.random(count)) / count for _ in range(4)])
    for i, sample in enumerate(samples):
        c = baseline.copy()
        c[STUDENT_COLUMNS] = ranges[STUDENT_COLUMNS, 0] + sample * np.diff(ranges[STUDENT_COLUMNS], axis=1)[:, 0]
        profiles.append(dict(id=f"joint_random_{i:02d}", condition=c.tolist()))
    anchors = profiles[-count:]
    for i, anchor in enumerate(anchors[:robots["paired_anchors"]]):
        for column in STUDENT_COLUMNS:
            c = np.asarray(anchor["condition"]).copy()
            lo, hi = ranges[column]
            c[column] = lo + (((c[column] - lo) / (hi - lo) + .5) % 1.) * (hi - lo)
            profiles.append(dict(id=f"paired_{i:02d}_{column}", condition=c.tolist()))
    return [p for p in profiles if p["condition"][3] == .5]


def condition_banks(teacher, seed, banks=3, critical=True, evaluation=False,
                    ranges=None, focus_anchor=None, region_mask=None):
    """Values vary by scene. Pair slots remain identical across scenes and banks."""
    from scipy.ndimage import uniform_filter
    bounds = RANGES if ranges is None else np.asarray(ranges, dtype=float)
    profiles = [dict(id=name, condition=c) for name, c in zip(
        ["nominal", "low_stand", "strict_stand", "relaxed"], FIXED)]
    groups, bank_columns = [[] for _ in STUDENT_COLUMNS], []
    grad, maximum = teacher.grad_sq.get(), teacher.grad_max.get()
    valid = teacher.valid_ground.get() & (teacher.interval.get() >= .5)
    if region_mask is not None:
        # 仅限定临界参数的采样来源，标签仍对全部原生查询点生成。
        valid &= region_mask
    for bank in range(banks):
        rng = np.random.default_rng(np.random.SeedSequence([seed, bank, 913733]))
        unit = np.column_stack([(rng.permutation(4) + rng.random(4)) / 4 for _ in range(4)])
        anchors = bounds[:, 0] + unit * np.diff(bounds, axis=1)[:, 0]
        if focus_anchor is not None:
            # 两个全范围锚点，两个目标能力附近的锚点，条件列布局保持一致。
            jitter = np.array([.1, .06, .025, .025])
            anchors[2:] = np.clip(np.asarray(focus_anchor) + rng.uniform(-1, 1, (2, 4))*jitter,
                                   bounds[:, 0], bounds[:, 1])
        columns = list(range(4))
        for a, values in enumerate(anchors):
            anchor = np.array([values[0], values[1], values[2], .5, values[3]])
            columns.append(len(profiles))
            profiles.append(dict(id=f"bank{bank}_anchor{a}", condition=anchor.tolist(), bank=bank, anchor=a))
            flat = (1.2 * teacher.resolution * np.tan(anchor[1])) ** 2
            counts = np.rint(uniform_filter((grad < flat).astype(np.float32), size=(1, 7, 7),
                                           mode="constant") * 49)
            step_region = valid & (grad > flat) & (maximum <= anchor[2] ** 2)
            stand_thresholds = (np.unique(counts[step_region]) + 2) / 49
            step_thresholds = np.unique(np.sqrt(maximum[valid & (grad > flat)]))
            slope_thresholds = np.unique(np.arctan(np.sqrt(grad[valid]) / (1.2 * teacher.resolution)))
            for p, column in enumerate(STUDENT_COLUMNS):
                lo, hi = bounds[p]
                # Wide pairs cover the full range even where the teacher stays stable.
                low = rng.uniform(lo, lo + .4 * (hi - lo))
                high = rng.uniform(lo + .6 * (hi - lo), hi)
                if critical and a >= 2:
                    delta = [.07, .035, .02, .25 / 49][p]
                    candidates = (np.array([anchor[0]]) if p == 0 else
                                  slope_thresholds if p == 1 else step_thresholds if p == 2 else stand_thresholds)
                    candidates = candidates[(candidates > lo + delta) & (candidates < hi - delta)]
                    if len(candidates):
                        threshold = float(rng.choice(candidates) if focus_anchor is None else
                                          candidates[np.argmin(abs(candidates-anchor[column]))])
                        low, high = threshold - delta, threshold + delta
                local = 4 + 9 * a + 1 + 2 * p
                if bank == 0:
                    groups[p].append([local, local + 1])
                for side, value in [("low", low), ("high", high)]:
                    c = anchor.copy()
                    c[column] = value
                    columns.append(len(profiles))
                    profiles.append(dict(id=f"bank{bank}_anchor{a}_{p}_{side}", condition=c.tolist(),
                                         bank=bank, anchor=a, paired_parameter=p))
        bank_columns.append(columns)
    if evaluation:
        # A fixed slope/step context with a dense stand sweep, separate from random pairs.
        sweep = [3]
        for value in [.08, .1, .12, .2, .3, .4, .5]:
            c = FIXED[3].copy()
            c[4] = value
            sweep.append(len(profiles))
            profiles.append(dict(id=f"stand_sweep_{value:g}", condition=c))
            bank_columns[0].append(len(profiles) - 1)
        groups[3].extend([[sweep[i], sweep[i+1]] for i in range(len(sweep)-1)])
    return profiles, bank_columns, groups
