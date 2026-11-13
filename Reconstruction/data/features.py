"""Frozen shared PTv3 pass, followed by the existing frozen Filter head."""
import numpy as np
import torch
import point_transformer_v3 as ptv3
from Filter.data.features import geometry_features
from Filter.training.evaluate import load_head


class FrozenFrontend:
    def __init__(self, checkpoint, device='cuda:0', threshold=.2):
        self.device=torch.device(device);self.threshold=threshold
        self.head,self.checkpoint=load_head(checkpoint,self.device)
        self.head.requires_grad_(False)
        self.config=self.checkpoint['config'];e=self.config['encoder']
        # The Filter checkpoint already owns the exact frozen backbone. No
        # machine-specific Hugging Face cache path or second download is needed.
        self.encoder=ptv3.PointTransformerV3.from_preset(e['name']).to(self.device).eval()
        self.encoder.precision='fp32'
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.encoder.requires_grad_(False)

    @torch.inference_mode()
    def encode(self,xyz,filter_points):
        e=self.config['encoder'];zeros=np.zeros_like(xyz)
        point,inverse=ptv3.prepare(xyz,zeros,zeros,preset=e['name'],grid_size=e['grid_size'],scale=e['scale'],device=self.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=ptv3.upcast(self.encoder(point)).feat
        score=torch.ones(len(xyz),device=self.device)
        if filter_points:
            scaled=(torch.as_tensor(xyz,device=self.device).double()*e['scale']).float()
            lo,hi=scaled.amin(0),scaled.amax(0)
            shift=torch.stack([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
            relative=(scaled-shift-point.coord[inverse])/e['grid_size']
            statistics=geometry_features(xyz,np.arange(len(xyz)),voxel=None,workers=4)
            score=self.head(dict(features=features.float(),inverse=inverse,relative=relative,
                geometry=torch.as_tensor(statistics,device=self.device)))
        keep=torch.nonzero(score>=self.threshold,as_tuple=True)[0] if filter_points else torch.arange(len(xyz),device=self.device)
        used,remapped=torch.unique(inverse[keep],sorted=True,return_inverse=True)
        ids=keep.cpu().numpy()
        return dict(xyz=xyz[ids],features=features[used].half().cpu().numpy(),inverse=remapped.cpu().numpy(),
                    keep_index=ids,keep_score=score.cpu().numpy())
