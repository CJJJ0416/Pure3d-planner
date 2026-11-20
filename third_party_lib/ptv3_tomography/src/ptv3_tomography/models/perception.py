"""Spatial PTv3 tokens -> fixed LiDAR-XY / gravity-height queries -> cost and height."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialMapDecoder(nn.Module):
    """Three-neighbour feature interpolation, then a shared per-query MLP.

    Queries depend only on map geometry and measured up direction. Teacher surface
    coordinates, availability and supervision masks never enter the forward pass.
    """
    def __init__(self, channels, map_config, hidden_channels=32):
        super().__init__()
        self.shape = (len(map_config["height_bin_edges_m"]) - 1, *map_config["grid_size_xy"])
        edges = torch.tensor(map_config["height_bin_edges_m"], dtype=torch.float32)
        centers, half_widths = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        nx, ny = self.shape[1:]
        xx, yy = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
        xy = torch.stack((xx - nx // 2, yy - ny // 2), dim=-1).float() * map_config["resolution_m"]
        self.register_buffer("query_xy", xy.reshape(-1, 2).repeat(len(centers), 1))
        self.register_buffer("query_height", centers.repeat_interleave(nx * ny))
        self.register_buffer("height_half_width", half_widths.repeat_interleave(nx * ny))
        self.mlp = nn.Sequential(nn.Linear(channels + 4, hidden_channels), nn.LayerNorm(hidden_channels),
                                 nn.GELU(), nn.Linear(hidden_channels, 2))

    def query_coordinates(self, up):
        z = (self.query_height - (self.query_xy * up[:2]).sum(-1)) / up[2]
        return torch.cat((self.query_xy, z[:, None]), dim=-1)

    def forward(self, coordinates, features, up):
        queries = self.query_coordinates(up)
        distances, indices = torch.cdist(queries, coordinates).topk(min(3, len(coordinates)), largest=False)
        weights = distances.clamp_min(1e-4).reciprocal()
        weights = weights / weights.sum(-1, keepdim=True)
        local = (features[indices] * weights[..., None]).sum(1)
        raw = self.mlp(torch.cat((local, queries, self.query_height[:, None]), dim=-1))
        return {"cost": raw[:, 0].sigmoid().reshape(self.shape),
                "height": (self.query_height + self.height_half_width * raw[:, 1].tanh()).reshape(self.shape)}


# 任务适配：官方 ScanNet 配方是 CrossEntropy+Lovasz，这里监督连续 cost 与高度。
def batched_perception_loss(prediction, targets, config):
    """Average each frame's masked regression loss before averaging the batch."""
    losses = {}
    for name in ("cost", "height"):
        # 回归和归约使用 FP32，骨干/MLP 的主要矩阵计算仍使用外层 BF16。
        pred = prediction[name].float()
        mask = targets["mask_" + name]
        # 先替换未知目标，避免 NaN 先参与减法。该 detach 只作用于未监督位置。
        target = torch.where(mask, targets[name].float(), pred.detach())
        error = F.smooth_l1_loss(pred, target, beta=config[name + "_beta"], reduction="none")
        # 每帧先按有效格归一化，再平均有该目标监督的帧，避免密集帧支配 loss。
        # cost/height 各用自己的掩码，support 目标不进入这个损失。
        counts = mask.flatten(1).sum(1)
        per_frame = (error * mask).flatten(1).sum(1) / counts.clamp_min(1)
        active = counts > 0
        losses[name] = (per_frame * active).sum() / active.sum().clamp_min(1)
    return sum(config[name + "_weight"] * losses[name] for name in losses), losses
