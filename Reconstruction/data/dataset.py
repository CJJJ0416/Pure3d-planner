"""Fixed-grid blocks extracted from shared, globally corrupted terrain versions."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.pointcloud import read_json
from Reconstruction.data.supervision import geometry_batch


def load_sample(root,row,stage):
    path=Path(root)/row['folder']
    feature=Path(root)/(row['complete_feature_file'] if stage=='ae' else row['feature_file'])
    with np.load(feature) as data:
        sample={k:np.array(data[k]) for k in ['xyz','features','inverse','keep_index','global_index']}
    sample.update(origin=np.array(row['origin'],np.float32),shape=row['shape'],row=row,folder=path)
    return sample


def on_device(sample,device):
    model_fields={'xyz','features','inverse','origin'}
    return {k:torch.as_tensor(v,device=device) if k in model_fields else v for k,v in sample.items()}


class Windows(Dataset):
    """Each item owns one core with context, not an independent geometry."""
    def __init__(self,root,split,stage):
        self.root=Path(root);self.stage=stage
        manifest=read_json(self.root/'manifest.json');rows=manifest['samples']
        self.stage_id=manifest.get('stage_id');self.data_version=manifest.get('data_version')
        self.rows=[r for r in rows if r['split']==split]
        if stage=='ae':self.rows=[r for r in self.rows if r['mask_id']=='complete']
        elif split in ['val','test']:self.rows=[r for r in self.rows if r['difficulty']!='outside_validation']

    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return load_sample(self.root,self.rows[i],self.stage)


def identity(sample):return sample
