"""Exact Mesh targets and grouped query sampling. Missing surfaces stay positive."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from Reconstruction.training.loss import OBSERVED,MISSING,NEAR_EMPTY,GLOBAL_EMPTY


def cell_keys(c):
    return np.ascontiguousarray(c,dtype=np.int32).view(np.dtype((np.void,12))).ravel()


def build_targets(query,complete,global_indices,origin,shape,levels,seed,near_distance=.24):
    from Reconstruction.data.geometry import surface_targets
    upper=np.asarray(origin,dtype=np.float64)+np.asarray(shape)*levels[0]
    targets=surface_targets(query,origin,upper,levels)
    tree=cKDTree(complete);rng=np.random.default_rng(seed)
    for i,size in enumerate(levels):
        mask=targets[f'label_{i}'];grid=targets[f'grid_{i}'][mask];points=targets[f'point_{i}'][mask]
        targets[f'source_index_{i}']=global_indices[tree.query(points,workers=4)[1]]
        budget=65536
        centers=(grid[rng.integers(len(grid),size=budget)]+.5)*size
        jitter=rng.uniform(-near_distance,near_distance,(budget,3))
        candidates=np.unique(np.floor((centers+jitter)/size).astype(np.int32),axis=0)
        xyz=(candidates.astype(np.float64)+.5)*size
        within=((xyz>=origin)&(xyz<upper)).all(1);candidates=candidates[within];xyz=xyz[within]
        labels,_=query.cells(candidates,size)
        targets[f'near_empty_{i}']=candidates[(~labels)&(query.distances(xyz)<=near_distance)]
    return targets


@lru_cache(maxsize=2)
def target_arrays(folder):
    with np.load(Path(folder)/'targets.npz') as data:
        result=[]
        for i in range(5):
            positive=data[f'label_{i}'];grid=data[f'grid_{i}'][positive]
            result.append((grid,data[f'point_{i}'][positive],data[f'source_index_{i}'],
                           data[f'near_empty_{i}'],cell_keys(grid)))
    return result


def pick(pool,n,rng):
    return rng.choice(pool,n,replace=len(pool)<n)


def geometry_batch(folder,levels,queries_per_level,rng,device,bounds,stage='ae',kept_global_indices=None):
    result=[];n=queries_per_level//4
    for size,(grid,point,source,near,occupied) in zip(levels,target_arrays(str(folder))):
        centers=(grid.astype(np.float64)+.5)*size
        eligible=((centers>=bounds[0])&(centers<bounds[1])).all(1)
        ids=np.flatnonzero(eligible)
        missing=np.zeros(len(grid),bool) if stage=='ae' else ~np.isin(source,kept_global_indices)
        if stage=='flow':
            a=ids[missing[ids]];b=ids[~missing[ids]]
            selected=np.r_[pick(a,n,rng),pick(b,n,rng)] if len(a) and len(b) else pick(ids,2*n,rng)
        else:selected=pick(ids,2*n,rng)
        near_centers=(near.astype(np.float64)+.5)*size
        near=near[((near_centers>=bounds[0])&(near_centers<bounds[1])).all(1)]
        near=near[pick(np.arange(len(near)),n,rng)] if len(near) else near
        global_count=2*n-len(near)
        lower=np.ceil(np.asarray(bounds[0])/size-.5).astype(np.int64)
        upper=np.ceil(np.asarray(bounds[1])/size-.5).astype(np.int64)
        chunks=[];count=0
        while count<global_count:
            proposed=rng.integers(lower,upper,size=(4*n,3))
            empty=proposed[~np.isin(cell_keys(proposed),occupied)]
            chunks.append(empty);count+=len(empty)
        cells=np.concatenate([grid[selected],near,np.concatenate(chunks)[:global_count]])
        xyz=(cells.astype(np.float64)+.5)*size
        offset=np.zeros_like(xyz);offset[:2*n]=point[selected]-xyz[:2*n]
        groups=np.r_[np.where(missing[selected],MISSING,OBSERVED),np.full(len(near),NEAR_EMPTY),np.full(global_count,GLOBAL_EMPTY)]
        result.append(dict(xyz=torch.as_tensor(xyz,dtype=torch.float32,device=device),
             label=torch.as_tensor(np.r_[np.ones(2*n),np.zeros(2*n)],dtype=torch.float32,device=device),
             offset=torch.as_tensor(offset,dtype=torch.float32,device=device),size=size,
             group=torch.as_tensor(groups,dtype=torch.long,device=device)))
    return result
