"""Variable-size cached cores, real batch collation and source-balanced sampling."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from utils.pointcloud import read_json


@lru_cache(maxsize=64)
def cached_arrays(folder,use_knn=True,statistics_folder=None):
    names=['features','inverse','relative','point_index','target','group','semantic']
    arrays={name:np.load(Path(folder)/f'{name}.npy',mmap_mode='r') for name in names}
    if use_knn: arrays['geometry']=np.load(Path(statistics_folder or folder)/'geometry.npy',mmap_mode='r')
    return arrays


class CachedCores(Dataset):
    def __init__(self,cache,split,suite='nominal',use_knn=True,geometry_mode=None):
        self.root=Path(cache);self.metadata=read_json(self.root/'encoding.json')
        self.geometry_mode=geometry_mode or ('knn' if use_knn else 'relative')
        self.has_statistics=self.geometry_mode!='relative'
        self.feature_root=Path(self.metadata['feature_cache']) if 'feature_cache' in self.metadata else self.root
        self.records=[r for r in self.metadata['records'] if r['split']==split and r['suite']==suite]
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        a=cached_arrays(str(self.feature_root/row['cache_key']),self.has_statistics,str(self.root/row['cache_key']))
        names=['features','inverse','relative','target','group','semantic']
        if self.has_statistics: names.append('geometry')
        return dict(**{k:torch.from_numpy(np.array(a[k])) for k in names},sample_index=index)


def collate_cores(samples):
    prefix=np.cumsum([0]+[len(s['features']) for s in samples[:-1]])
    names=['relative','target','group','semantic']
    if 'geometry' in samples[0]: names.append('geometry')
    return dict(features=torch.cat([s['features'] for s in samples]),
        inverse=torch.cat([s['inverse']+int(o) for s,o in zip(samples,prefix)]),
        **{k:torch.cat([s[k] for s in samples]) for k in names},
        sample_indices=[s['sample_index'] for s in samples],counts=[len(s['target']) for s in samples])


class BalancedBatches(Sampler):
    """Two general, two dynamic, two registration, one static and one disconnected box."""
    def __init__(self,dataset,seed,epoch=0,start_batch=0):
        self.dataset,self.seed,self.epoch,self.start_batch=dataset,seed,epoch,start_batch
        self.steps=math.ceil(len(dataset)/8)
        self.scenes=sorted({r['scene'] for r in dataset.records})
        self.pools={}
        for scene in self.scenes:
            for group in range(5):
                self.pools[scene,group]=np.array([i for i,r in enumerate(dataset.records)
                    if r['scene']==scene and (group==0 or r['group_counts'][group]>0)],np.int64)
    def __len__(self): return self.steps-self.start_batch
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        for step in range(self.steps):
            batch=[]
            for group in [0,0,4,4,3,3,1,2]:
                scene=self.scenes[int(rng.integers(len(self.scenes)))]
                batch.append(int(rng.choice(self.pools[scene,group])))
            if step>=self.start_batch: yield batch


class SceneBalancedBatches(Sampler):
    """Uniform scenes and variable-size boxes, without requiring absent sources."""
    def __init__(self,dataset,seed,batch_size,epoch=0,start_batch=0):
        self.seed,self.batch_size,self.epoch,self.start_batch=seed,batch_size,epoch,start_batch
        self.steps=math.ceil(len(dataset)/batch_size)
        self.scenes=sorted({r['scene'] for r in dataset.records})
        self.pools=[np.array([i for i,r in enumerate(dataset.records) if r['scene']==scene]) for scene in self.scenes]
    def __len__(self): return self.steps-self.start_batch
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        for step in range(self.steps):
            batch=[int(rng.choice(self.pools[int(rng.integers(len(self.pools)))])) for _ in range(self.batch_size)]
            if step>=self.start_batch: yield batch
