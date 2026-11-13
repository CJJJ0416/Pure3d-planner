"""Continuous retention targets from a static triangle reference and provenance."""
from time import perf_counter

import numpy as np
import open3d as o3d

from Filter.data.augment import SOURCE_ROLES


class StaticSurfaceTargets:
    """Build the CPU triangle acceleration structure once per WFC scene."""
    def __init__(self, mesh, config):
        started = perf_counter()
        self.config = config
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=config["query_threads"])
        vertices = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
        triangles = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
        self.scene.add_triangles(vertices, triangles)
        self.build_s = perf_counter() - started

    def score(self, rough):
        started = perf_counter()
        coord = rough["coord"]
        distance = np.empty(len(coord), np.float32)
        batch = self.config["query_batch_points"]
        for start in range(0, len(coord), batch):
            distance[start:start + batch] = self.scene.compute_distance(
                o3d.core.Tensor(np.ascontiguousarray(coord[start:start + batch])),
                nthreads=self.config["query_threads"]).numpy()
        target = np.exp(-.5 * (distance / self.config["sigma_m"]) ** 2)
        target[rough["source_role"] == SOURCE_ROLES["dynamic_object_trail"]] = 0.
        result = dict(keep_target=target, surface_distance_m=distance)
        groups = {}
        for name, role in SOURCE_ROLES.items():
            selected = rough["source_role"] == role
            if selected.any():
                values = target[selected]
                groups[name] = dict(points=int(selected.sum()), mean=float(values.mean()),
                    min=float(values.min()), max=float(values.max()),
                    retained_at_0_5=int((values >= .5).sum()),
                    quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist())
        disconnected=rough['is_disconnected_control']
        report = dict(points=len(coord), sigma_m=self.config["sigma_m"],
            distance_query_and_score_s=perf_counter() - started, groups=groups,
            histogram_edges=np.linspace(0, 1, 11).tolist(),
            histogram_counts=np.histogram(target, np.linspace(0, 1, 11))[0].tolist(),
            disconnected_control_points=int(disconnected.sum()),
            disconnected_control_min=float(target[disconnected].min()) if disconnected.any() else None)
        return result, report
