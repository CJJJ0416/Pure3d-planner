"""Fixed-XYZ stair connectivity metrics. This does not modify the planning graph."""
from collections import defaultdict
from pathlib import Path
import numpy as np

from utils.pointcloud import read_json, write_json


def components(indptr, indices, allowed):
    labels = np.full(len(allowed), -1, dtype=np.int32)
    component = 0
    for root in np.flatnonzero(allowed):
        if labels[root] >= 0:
            continue
        labels[root] = component
        stack = [int(root)]
        while stack:
            node = stack.pop()
            neighbours = indices[indptr[node]:indptr[node + 1]]
            neighbours = neighbours[allowed[neighbours] & (labels[neighbours] < 0)]
            labels[neighbours] = component
            stack.extend(neighbours.tolist())
        component += 1
    return labels


def connected(labels, pairs):
    a, b = pairs.T
    return (labels[a] >= 0) & (labels[a] == labels[b])


def prepare_evaluation(root):
    """Cache radius adjacency and endpoints from geometry, independent of predictions."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    root = Path(root)
    meta = read_json(root / "blocks.json")
    for scene in meta["scenes"]:
        xyz = np.load(scene["arrays"]["query_coord"])
        nodes = np.flatnonzero(np.load(scene["arrays"]["stair_mask"]))
        points = xyz[nodes]
        low = np.flatnonzero(points[:, 2] < points[:, 2].min() + .06)
        high = np.flatnonzero(points[:, 2] > points[:, 2].max() - .06)
        rng = np.random.default_rng(scene["seed"])
        geometric_endpoints = np.column_stack((rng.choice(low, 3), rng.choice(high, 3)))
        edges = cKDTree(points).query_pairs(.45, output_type="ndarray")
        distance = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)
        ii, jj = np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]]
        full = coo_matrix((np.ones(len(ii),bool),(ii,jj)),shape=(len(nodes),len(nodes))).tocsr()
        truth = np.load(scene["arrays"]["cost"],mmap_mode="r")[nodes,:4]
        endpoints, profile_ids = [], []
        for profile in range(4):
            selected = geometric_endpoints.tolist()
            labels = components(full.indptr,full.indices,truth[:,profile] <= .3)
            candidates = []
            for label in np.unique(labels):
                if label < 0:
                    continue
                component = np.flatnonzero(labels==label)
                span = np.ptp(points[component,2])
                if span >= .6 * np.ptp(points[:,2]):
                    candidates.append((span,component))
            if candidates:
                component = max(candidates,key=lambda item:item[0])[1]
                z = points[component,2]
                start = component[z < z.min()+.05]
                goal = component[z > z.max()-.05]
                selected.extend(np.column_stack((rng.choice(start,3),rng.choice(goal,3))).tolist())
            endpoints.extend(selected)
            profile_ids.extend([profile]*len(selected))
        endpoints,profile_ids = np.asarray(endpoints),np.asarray(profile_ids)
        directory = Path(scene["arrays"]["query_coord"]).parent
        graphs = []
        for radius in [.3, .45]:
            e = edges[distance <= radius]
            i, j = np.r_[e[:,0], e[:,1]], np.r_[e[:,1], e[:,0]]
            csr = coo_matrix((np.ones(len(i), bool), (i, j)), shape=(len(nodes),len(nodes))).tocsr()
            geometry = connected(components(csr.indptr, csr.indices, np.ones(len(nodes),bool)), endpoints)
            path = directory / f"connectivity_{radius:g}.npz"
            np.savez(path, nodes=nodes, indptr=csr.indptr, indices=csr.indices, endpoints=endpoints,
                     geometry_connected=geometry,profile_ids=profile_ids)
            graphs.append(dict(radius=radius, file=str(path)))
        scene["connectivity"] = graphs
    meta["stair_evaluation"] = True
    write_json(root / "blocks.json", meta)


class StairMetrics:
    def __init__(self, dataset):
        self.scenes = {s["id"]:s for s in dataset.metadata["scenes"]
                       if any(r["scene"] == s["id"] for r in dataset.rows)}
        self.prediction = {name:np.empty((s["query_points"],4),np.float32) for name,s in self.scenes.items()}
        self.truth = {name:np.load(s["arrays"]["cost"],mmap_mode="r")[:,s["profile_banks"][0][:4]]
                      for name,s in self.scenes.items()}
        self.mask = {name:np.load(s["arrays"]["stair_mask"]) for name,s in self.scenes.items()}
        self.non_stair = np.zeros(2)

    def add(self, batch, prediction, target):
        qb = batch["query_batch"].cpu().numpy()
        qi = batch["query_index"].cpu().numpy()
        pred, truth = prediction.cpu().numpy(), target.cpu().numpy()
        for b,name in enumerate(batch["scene"]):
            selected = qb == b
            indices = qi[selected]
            self.prediction[name][indices] = pred[selected,:4]
            outside = ~self.mask[name][indices]
            delta = abs(pred[selected][outside] - truth[selected][outside])
            self.non_stair += [delta.sum(), delta.size]

    def report(self):
        rows = []
        for name, scene in self.scenes.items():
            for graph in scene["connectivity"]:
                with np.load(graph["file"]) as g:
                    nodes, ptr, ind = g["nodes"],g["indptr"],g["indices"]
                    endpoints, geometry = g["endpoints"],g["geometry_connected"]
                    for condition in range(4):
                        selected = g["profile_ids"] == condition
                        pairs,geom = endpoints[selected],geometry[selected]
                        gt = connected(components(ptr, ind, self.truth[name][nodes,condition] <= .3), pairs)
                        pred = connected(components(ptr, ind, self.prediction[name][nodes,condition] <= .3), pairs)
                        for k in range(len(pairs)):
                            rows.append(dict(scene=name, family=scene["family"], radius=graph["radius"],
                                             profile=condition, pair=k, geometry=bool(geom[k]),
                                             gt=bool(gt[k]), prediction=bool(pred[k])))
        summary = {}
        for radius in [.3,.45]:
            families = defaultdict(list)
            for r in rows:
                if r["radius"] == radius and r["geometry"]:
                    families[r["family"]].append(r)
            breaks, false_pass = [], []
            counts = dict(gt_connected=0,gt_blocked=0,geometry_disconnected=sum(
                r["radius"] == radius and not r["geometry"] for r in rows))
            for values in families.values():
                positive = [r for r in values if r["gt"]]
                negative = [r for r in values if not r["gt"]]
                counts["gt_connected"] += len(positive)
                counts["gt_blocked"] += len(negative)
                if positive:
                    breaks.append(np.mean([not r["prediction"] for r in positive]))
                if negative:
                    false_pass.append(np.mean([r["prediction"] for r in negative]))
            summary[str(radius)] = dict(break_pct=float(100*np.mean(breaks)) if breaks else None,
                                       false_pass_pct=float(100*np.mean(false_pass)) if false_pass else None, **counts)
        return dict(radii=summary, non_stair_mae=float(self.non_stair[0]/self.non_stair[1]), pairs=rows)


def checkpoint_key(metrics, baseline):
    """Fixed validation criterion: retention first, then stair breaks and stand response."""
    value, before = metrics["stair_metrics"], baseline["stair_metrics"]
    current, initial = value["radii"]["0.45"], before["radii"]["0.45"]
    eligible = (current["false_pass_pct"] <= initial["false_pass_pct"] + 2
                and value["non_stair_mae"] <= before["non_stair_mae"] * 1.05)
    return (not eligible, current["break_pct"],
            metrics["parameter_response"]["stand_ratio"]["changed_delta_mae"], metrics["cost_mae"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    prepare_evaluation(parser.parse_args().prepare)
