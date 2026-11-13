"""Geometry autoencoder and conditional latent velocity field.

Dense tensors use [B,C,X,Y,Z]. XYZ remains in map metres outside the PTv3
preprocessor. Grid metadata, never target occupancy, defines latent support.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sample_grid(grid, xyz, origin, spacing):
    shape = torch.tensor(grid.shape[2:], device=xyz.device)
    normalized = 2*(xyz-origin)/ (shape*spacing)-1
    query = normalized[:, [2,1,0]].view(1,1,1,-1,3)
    return F.grid_sample(grid.float(), query.float(), align_corners=False, padding_mode='border').reshape(grid.shape[1],-1).T


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(486,64), nn.LayerNorm(64), nn.GELU())
        self.position = nn.Linear(67,32)
        self.compress = nn.Sequential(nn.Conv3d(33,64,4,2,1), nn.GroupNorm(8,64), nn.SiLU(),
                                      nn.Conv3d(64,32,4,2,1), nn.GroupNorm(8,32), nn.SiLU())

    def forward(self, xyz, features, inverse, origin, shape):
        fine_shape = tuple(int(n)*4 for n in shape)
        cells = torch.floor((xyz-origin)/.06).long()
        local = (xyz-origin)/.06-cells-.5
        feat = self.position(torch.cat([self.project(features.float())[inverse], local],1))
        flat = (cells[:,0]*fine_shape[1]+cells[:,1])*fine_shape[2]+cells[:,2]
        total = math.prod(fine_shape)
        values = torch.zeros((total,32),device=xyz.device,dtype=torch.float32)
        count = torch.zeros((total,1),device=xyz.device)
        values.index_add_(0,flat,feat.float())
        count.index_add_(0,flat,torch.ones((len(flat),1),device=xyz.device))
        values = torch.cat([values/count.clamp_min(1),count.gt(0).float()],1)
        grid = values.T.reshape(1,33,*fine_shape)
        return self.compress(grid)


class GeometryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Sequential(nn.ConvTranspose3d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
                                    nn.ConvTranspose3d(64,32,4,2,1),nn.GroupNorm(8,32),nn.SiLU())
        self.query = nn.Sequential(nn.Linear(85,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.surface = nn.Sequential(nn.Linear(129,64),nn.SiLU(),nn.Linear(64,1))
        self.offset = nn.Sequential(nn.Linear(129,64),nn.SiLU(),nn.Linear(64,3))
        nn.init.constant_(self.surface[-1].bias,-2.)

    def forward(self, latent, origin):
        return GeometryField(self, latent, self.expand(latent), origin)


class GeometryField:
    def __init__(self, decoder, latent, fine, origin):
        self.decoder,self.latent,self.fine,self.origin = decoder,latent,fine,origin

    def query_features(self, xyz):
        """Internal geometry feature query. Downstream Cost integration is deferred."""
        local = (xyz-self.origin)/.24
        phase = local[:,:,None]*torch.tensor([1,2,4],device=xyz.device)*torch.pi
        positional = torch.cat([local/32, phase.sin().flatten(1), phase.cos().flatten(1)],1)
        return self.decoder.query(torch.cat([sample_grid(self.latent,xyz,self.origin,.24),
            sample_grid(self.fine,xyz,self.origin,.06),positional],1))

    def predict_cells(self, xyz, size):
        feature = self.query_features(xyz)
        scale = torch.full((len(xyz),1), math.log2(size/.015),device=xyz.device)
        h = torch.cat([feature,scale],1)
        return self.decoder.surface(h).squeeze(1), self.decoder.offset(h).tanh()*(size/2)

    @torch.no_grad()
    def generate(self, core_lower, core_upper, kept_xyz, levels=(.24,.12,.06,.03,.015),
                 expansion_threshold=.35, threshold=.5, chunk=32768):
        from Reconstruction.data.geometry import grid_between, CHILDREN
        import numpy as np
        device = self.latent.device
        grid = torch.as_tensor(grid_between(core_lower,core_upper,levels[0]),device=device)
        child = torch.as_tensor(CHILDREN,device=device)
        for level,size in enumerate(levels):
            centers=(grid.float()+.5)*size
            probs,offsets=[],[]
            for q in centers.split(chunk):
                logits,offset=self.predict_cells(q,size)
                probs.append(logits.float().sigmoid());offsets.append(offset.float())
            if not probs:
                return dict(xyz=np.empty((0,3),np.float32), probability=np.empty(0,np.float32), cells=np.empty((0,3),np.int64))
            prob=torch.cat(probs);offset=torch.cat(offsets)
            selected=prob>=(threshold if level==len(levels)-1 else expansion_threshold)
            if level<len(levels)-1:
                grid=(grid[selected,None]*2+child).reshape(-1,3)
        xyz=(centers+offset)[selected].cpu().numpy();cells=grid[selected].cpu().numpy();prob=prob[selected].cpu().numpy()
        inside=((xyz>=core_lower)&(xyz<core_upper)).all(1)
        xyz,cells,prob=xyz[inside],cells[inside],prob[inside]
        # BF16 offsets can round onto a cell face. Deduplicate by the actual
        # exported XYZ cell, including collisions with retained original points.
        cells=np.floor(xyz.astype(np.float64)/levels[-1]).astype(np.int64)
        _,first=np.unique(cells,axis=0,return_index=True)
        first.sort();xyz,cells,prob=xyz[first],cells[first],prob[first]
        # Retained original rows have priority over synthesis in their cells.
        keep_cells=np.floor(np.asarray(kept_xyz,dtype=np.float64)/levels[-1]).astype(np.int64)
        occupied={tuple(c) for c in keep_cells}
        fresh=np.array([tuple(c) not in occupied for c in cells],bool)
        return dict(xyz=xyz[fresh],probability=prob[fresh],cells=cells[fresh])


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=SpatialEncoder();self.decoder=GeometryDecoder()

    def encode(self,sample):
        return self.encoder(sample['xyz'],sample['features'],sample['inverse'],sample['origin'],sample['shape'])

    def forward(self,sample):
        return self.decoder(self.encode(sample),sample['origin'])


class TimeBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.first=nn.Conv3d(cin,cout,3,padding=1)
        self.norm1=nn.GroupNorm(8,cout);self.norm2=nn.GroupNorm(8,cout)
        self.time=nn.Linear(64,cout)
        self.second=nn.Conv3d(cout,cout,3,padding=1)
        self.skip=nn.Conv3d(cin,cout,1) if cin!=cout else nn.Identity()

    def forward(self,x,t):
        h=F.silu(self.norm1(self.first(x))+self.time(t)[:,:,None,None,None])
        return F.silu(self.norm2(self.second(h))+self.skip(x))


class VelocityUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,64))
        self.enc0=TimeBlock(64,64);self.enc1=TimeBlock(64,128);self.mid=TimeBlock(128,256)
        self.dec1=TimeBlock(384,128);self.dec0=TimeBlock(192,64)
        self.out=nn.Conv3d(64,32,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,z,t,condition):
        frequency=torch.exp(torch.arange(32,device=z.device)*(-math.log(10000)/31))
        phase=t.reshape(-1,1)*frequency[None]*1000
        time=self.time(torch.cat([phase.sin(),phase.cos()],1))
        a=self.enc0(torch.cat([z,condition],1),time)
        b=self.enc1(F.avg_pool3d(a,2,ceil_mode=True),time)
        c=self.mid(F.avg_pool3d(b,2,ceil_mode=True),time)
        h=self.dec1(torch.cat([F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b],1),time)
        h=self.dec0(torch.cat([F.interpolate(h,size=a.shape[2:],mode='trilinear',align_corners=False),a],1),time)
        return self.out(h)


class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition=SpatialEncoder();self.velocity=VelocityUNet()
        self.register_buffer('latent_mean',torch.zeros(1,32,1,1,1))
        self.register_buffer('latent_std',torch.ones(1,32,1,1,1))

    def encode_condition(self,sample):
        return self.condition(sample['xyz'],sample['features'],sample['inverse'],sample['origin'],sample['shape'])

    def normalized(self,z):
        return (z-self.latent_mean)/self.latent_std

    @torch.no_grad()
    def sample(self,condition,steps=16,seed=42):
        generator=torch.Generator(device=condition.device).manual_seed(seed)
        z=torch.randn(condition.shape,device=condition.device,generator=generator)
        for i in range(steps):
            z=z+self.velocity(z,torch.tensor([i/steps],device=z.device),condition).float()/steps
        return z*self.latent_std+self.latent_mean
