"""Stair connectivity metrics and checkpoint selection used by training."""

from collections import defaultdict
import numpy as np


def components(indptr, indices, allowed):
    """标记允许节点集合中的连通分量。"""
    labels = np.full(len(allowed), -1, dtype=np.int32)
    component = 0
    for root in np.flatnonzero(allowed):
        if labels[root] >= 0:
            continue
        labels[root] = component
        stack = [int(root)]
        while stack:
            node = stack.pop()
            neighbours = indices[indptr[node] : indptr[node + 1]]
            neighbours = neighbours[allowed[neighbours] & (labels[neighbours] < 0)]
            labels[neighbours] = component
            stack.extend(neighbours.tolist())
        component += 1
    return labels


def connected(labels, pairs):
    """判断边两端是否连通。"""
    a, b = pairs.T
    return (labels[a] >= 0) & (labels[a] == labels[b])


class StairMetrics:
    """累计楼梯连通性指标。"""

    def __init__(self, dataset):
        self.scenes = {
            s["id"]: s for s in dataset.metadata["scenes"] if any(r["scene"] == s["id"] for r in dataset.rows)
        }
        self.prediction = {
            name: np.empty((s["query_points"], 4), np.float32) for name, s in self.scenes.items()
        }
        self.truth = {
            name: np.load(s["arrays"]["cost"], mmap_mode="r")[:, s["profile_banks"][0][:4]]
            for name, s in self.scenes.items()
        }
        self.mask = {name: np.load(s["arrays"]["stair_mask"]) for name, s in self.scenes.items()}
        self.non_stair = np.zeros(2)

    def add(self, batch, prediction, target):
        """把模型预测写入对应场景的查询位置。"""
        qb = batch["query_batch"].cpu().numpy()
        qi = batch["query_index"].cpu().numpy()
        pred, truth = prediction.cpu().numpy(), target.cpu().numpy()
        for b, name in enumerate(batch["scene"]):
            selected = qb == b
            indices = qi[selected]
            self.prediction[name][indices] = pred[selected, :4]
            outside = ~self.mask[name][indices]
            delta = abs(pred[selected][outside] - truth[selected][outside])
            self.non_stair += [delta.sum(), delta.size]

    def report(self):
        """生成楼梯连通性报告。"""
        rows = []
        for name, scene in self.scenes.items():
            for graph in scene["connectivity"]:
                with np.load(graph["file"]) as g:
                    nodes, ptr, ind = g["nodes"], g["indptr"], g["indices"]
                    endpoints, geometry = g["endpoints"], g["geometry_connected"]
                    for condition in range(4):
                        selected = g["profile_ids"] == condition
                        pairs, geom = endpoints[selected], geometry[selected]
                        gt = connected(components(ptr, ind, self.truth[name][nodes, condition] <= 0.3), pairs)
                        pred = connected(
                            components(ptr, ind, self.prediction[name][nodes, condition] <= 0.3), pairs
                        )
                        for k in range(len(pairs)):
                            rows.append(
                                dict(
                                    scene=name,
                                    family=scene["family"],
                                    radius=graph["radius"],
                                    profile=condition,
                                    pair=k,
                                    geometry=bool(geom[k]),
                                    gt=bool(gt[k]),
                                    prediction=bool(pred[k]),
                                )
                            )
        summary = {}
        for radius in [0.3, 0.45]:
            families = defaultdict(list)
            for r in rows:
                if r["radius"] == radius and r["geometry"]:
                    families[r["family"]].append(r)
            breaks, false_pass = [], []
            counts = dict(
                gt_connected=0,
                gt_blocked=0,
                geometry_disconnected=sum(r["radius"] == radius and not r["geometry"] for r in rows),
            )
            for values in families.values():
                positive = [r for r in values if r["gt"]]
                negative = [r for r in values if not r["gt"]]
                counts["gt_connected"] += len(positive)
                counts["gt_blocked"] += len(negative)
                if positive:
                    breaks.append(np.mean([not r["prediction"] for r in positive]))
                if negative:
                    false_pass.append(np.mean([r["prediction"] for r in negative]))
            summary[str(radius)] = dict(
                break_pct=float(100 * np.mean(breaks)) if breaks else None,
                false_pass_pct=float(100 * np.mean(false_pass)) if false_pass else None,
                **counts
            )
        return dict(radii=summary, non_stair_mae=float(self.non_stair[0] / self.non_stair[1]), pairs=rows)


def checkpoint_key(metrics, baseline):
    """Fixed validation criterion: retention first, then stair breaks and stand response."""
    value, before = metrics["stair_metrics"], baseline["stair_metrics"]
    current, initial = value["radii"]["0.45"], before["radii"]["0.45"]
    eligible = (
        current["false_pass_pct"] <= initial["false_pass_pct"] + 2
        and value["non_stair_mae"] <= before["non_stair_mae"] * 1.05
    )
    return (
        not eligible,
        current["break_pct"],
        metrics["parameter_response"]["stand_ratio"]["changed_delta_mae"],
        metrics["cost_mae"],
    )
