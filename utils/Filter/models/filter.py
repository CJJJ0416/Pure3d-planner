"""Pointwise retention scorer. Inputs are frozen XYZ features and XYZ statistics."""
import torch
from torch import nn


def geometry_mode(config):
    """The explicit mode extends the two historical use_knn checkpoint formats."""
    return config.get('geometry_mode','knn' if config.get('use_knn',True) else 'relative')


class FilterHead(nn.Module):
    def __init__(self, mean=None, std=None, dropout=.1, use_knn=True, geometry_mode=None):
        super().__init__()
        self.geometry_mode = geometry_mode or ('knn' if use_knn else 'relative')
        self.use_knn = self.geometry_mode=='knn'
        self.has_statistics = self.geometry_mode!='relative'
        self.register_buffer('geometry_mean', torch.as_tensor(mean, dtype=torch.float32) if self.has_statistics else None)
        self.register_buffer('geometry_std', torch.as_tensor(std, dtype=torch.float32) if self.has_statistics else None)
        self.projection = nn.Sequential(nn.Linear(486,128),nn.LayerNorm(128),nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(9 if self.has_statistics else 3,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,32),nn.LayerNorm(32),nn.GELU())
        self.score = nn.Sequential(nn.Linear(160,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(128,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1))

    def forward(self, batch):
        feature = self.projection(batch['features'])[batch['inverse']]
        relative = batch['relative'].float()
        if self.has_statistics:
            statistics = (batch['geometry'].float()-self.geometry_mean)/self.geometry_std
            relative = torch.cat([relative,statistics],1)
        geometry = self.geometry(relative)
        return self.score(torch.cat([feature,geometry],1)).squeeze(1).sigmoid().float()
