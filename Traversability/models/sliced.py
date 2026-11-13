"""Point-aligned adapter, FiLM and cost head for a frozen Tiny encoder."""
import torch
from torch import nn


class SlicedCostHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conditioning = config['conditioning']
        channels = config.get('encoder', {}).get('feature_channels', 486)
        self.adapter = nn.Sequential(nn.Linear(channels, 256), nn.LayerNorm(256), nn.GELU())
        self.condition_head = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 512))
        inputs = 260 if self.conditioning == 'concat' else 256
        self.cost_head = nn.Sequential(nn.Linear(inputs, 128), nn.GELU(), nn.Linear(128, 64),
                                       nn.GELU(), nn.Linear(64, 1))
        self.register_buffer('condition_scale', torch.tensor(config['condition_scale']))

    def forward(self, features, conditions, voxel_batch):
        geometry = self.adapter(features.float())
        scaled = conditions / self.condition_scale
        gamma, beta = self.condition_head(scaled).chunk(2, dim=-1)
        chunks = []
        for start in range(0, len(geometry), 2048):
            end = start + 2048
            batch = voxel_batch[start:end]
            if self.conditioning == 'concat':
                value = torch.cat((geometry[start:end, None].expand(-1, conditions.shape[1], -1),
                                   scaled[batch]), dim=-1)
            else:
                value = (1 + gamma[batch]) * geometry[start:end, None] + beta[batch]
            chunks.append(self.cost_head(value).squeeze(-1).sigmoid())
        return torch.cat(chunks)
