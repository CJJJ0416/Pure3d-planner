"""Native merged ground Tomogram as normalized XYZ + trav_cost points."""
import numpy as np


def tomogram_points(payload):
    """Export /tomogram geometry and costs, without querying the input PCD.

    Adjacent ground layers closer than slice_dh merge into the upper layer,
    carrying their minimum cost, as in the planner's publishTomogram logic.
    Integer XY centres match the tomography CUDA kernel, including odd grids.
    Zero-cost and barrier points are both retained. Ceiling is not a cost cloud.
    """
    cost = np.asarray(payload["data"][0], dtype=np.float32).copy()
    ground = np.asarray(payload["data"][3], dtype=np.float32).copy()
    count, nx, ny = ground.shape
    resolution, center = payload["resolution"], payload["center"]
    clouds = []
    for layer in range(count):
        if layer + 1 < count:
            merge = ground[layer + 1] - ground[layer] < payload["slice_dh"]
            ground[layer, merge] = np.nan
            cost[layer + 1, merge] = np.minimum(cost[layer, merge], cost[layer + 1, merge])
        x, y = np.nonzero(np.isfinite(ground[layer]) & np.isfinite(cost[layer]))
        clouds.append(np.column_stack((
            (x - nx // 2) * resolution + center[0],
            (y - ny // 2) * resolution + center[1],
            ground[layer, x, y], cost[layer, x, y] / payload["cost_scale"],
        )))
    return np.concatenate(clouds).astype(np.float32)
