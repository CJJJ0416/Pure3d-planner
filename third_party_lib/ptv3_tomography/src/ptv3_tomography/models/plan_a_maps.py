"""Plan A's explicit map interface. No teacher values/masks enter this adapter."""
import torch


def pack_plan_a_maps(prediction, points_sensor_m, up_sensor, map_config):
    """One cloud -> [B=1,C=2,L,X,Y] maps and a separate input-only hit mask.

    Channels are normalized cost and gravity-relative ground height in metres.
    The hit mask indicates endpoint observations, not support or traversability.
    """
    cost, height = prediction["cost"], prediction["height"]
    expected = (len(map_config["height_bin_edges_m"]) - 1, *map_config["grid_size_xy"])
    nx, ny = expected[1:]
    xy = torch.round(points_sensor_m[:, :2] / map_config["resolution_m"]).long()
    xy = xy + xy.new_tensor([nx // 2, ny // 2])
    edges = up_sensor.new_tensor(map_config["height_bin_edges_m"])
    band = torch.bucketize(points_sensor_m @ up_sensor, edges, right=True) - 1
    valid = ((xy >= 0) & (xy < xy.new_tensor([nx, ny]))).all(1) & (band >= 0) & (band < expected[0])
    observed = torch.zeros_like(cost, dtype=torch.bool)
    observed[band[valid], xy[valid, 0], xy[valid, 1]] = True
    return {"maps": torch.stack((cost, height), dim=0).unsqueeze(0),
            "input_observed_hits": observed.unsqueeze(0).unsqueeze(0),
            "cost_raw": cost * map_config["cost_scale"]}


def surface_points_from_height(height, up_sensor, map_config):
    """Reconstruct predicted 3D surfaces from fixed LiDAR XY and predicted gravity height."""
    nx, ny = map_config["grid_size_xy"]
    x = (torch.arange(nx, device=height.device) - nx // 2) * map_config["resolution_m"]
    y = (torch.arange(ny, device=height.device) - ny // 2) * map_config["resolution_m"]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx, yy = xx.expand_as(height), yy.expand_as(height)
    zz = (height - up_sensor[0] * xx - up_sensor[1] * yy) / up_sensor[2]
    return torch.stack((xx, yy, zz), dim=-1)
