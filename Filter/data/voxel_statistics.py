"""World-aligned, full-PCD CUDA counts and 26-cell occupancy statistics."""
import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from utils.pointcloud import read_json,write_json,workspace_path


@torch.inference_mode()
def voxel_statistics(xyz_cuda,voxel_sizes=(.05,.15,.5)):
    """[N,3] map XYZ -> [N,6], interleaved log1p(count) and occupancy/26.

    Counts include repeated observations and the query point. Occupancy excludes
    the central cell. No point is removed or moved. Float64 cell division fixes
    the lattice before integer encoding, including for negative coordinates.
    """
    offsets=torch.tensor([v for v in product((-1,0,1),repeat=3) if v!=(0,0,0)],
                         device=xyz_cuda.device,dtype=torch.int64)
    result=torch.empty((len(xyz_cuda),2*len(voxel_sizes)),device=xyz_cuda.device,dtype=torch.float32)
    for scale,size in enumerate(voxel_sizes):
        grid=torch.floor(xyz_cuda.double()/size).long()
        grid-=grid.amin(0)
        shape=grid.amax(0)+3
        grid+=1
        stride_y,stride_x=shape[2],shape[1]*shape[2]
        keys=grid[:,0]*stride_x+grid[:,1]*stride_y+grid[:,2]
        del grid
        occupied,inverse,counts=torch.unique(keys,sorted=True,return_inverse=True,return_counts=True)
        del keys
        delta=offsets[:,0]*stride_x+offsets[:,1]*stride_y+offsets[:,2]
        occupancy=torch.empty(len(occupied),device=xyz_cuda.device,dtype=torch.float32)
        for start in range(0,len(occupied),65536):
            query=occupied[start:start+65536,None]+delta[None]
            position=torch.searchsorted(occupied,query)
            position.clamp_max_(len(occupied)-1)
            occupancy[start:start+65536]=(occupied[position]==query).sum(1).float()/26.
        result[:,2*scale]=torch.log1p(counts[inverse].float())
        result[:,2*scale+1]=occupancy[inverse]
    return result


def build_cache(config):
    """Cache only new statistics. Frozen features and labels stay in their base cache."""
    torch.set_num_threads(config['cpu_threads'])
    base=workspace_path(config['feature_cache']);root=workspace_path(config['cache'])
    root.mkdir(parents=True,exist_ok=True)
    metadata=deepcopy(read_json(base/'encoding.json'))
    metadata.update(config=config,geometry_mode='voxel',use_knn=False,feature_cache=str(base),
        geometry_mean=None,geometry_std=None,complete=False,feature_names=config['geometry']['feature_names'])
    groups=defaultdict(list)
    for row in metadata['records']:
        folder=workspace_path(config['data'])/row['scene']
        if row['suite']!='nominal': folder=folder/'suites'/row['suite']
        row['folder']=str(folder)
        groups[(row['scene'],row['suite'],row['variant'])].append(row)
    write_json(root/'encoding.json',metadata)
    device=torch.device(config['device']);started=perf_counter()
    count=0;sums=np.zeros(6);squares=np.zeros(6);cuda_s=0.;timings=[]
    for (scene,suite,variant),rows in groups.items():
        tick=perf_counter();folder=Path(rows[0]['folder'])/f'arrays_{variant:03d}'
        xyz=np.load(folder/'coord.npy')
        tensor=torch.as_tensor(xyz,device=device)
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();values=voxel_statistics(tensor,config['geometry']['voxel_sizes_m']);end.record()
        values=values.cpu().numpy();elapsed=begin.elapsed_time(end)/1000.;cuda_s+=elapsed
        for row in rows:
            ids=np.load(base/row['cache_key']/'point_index.npy')
            output=root/row['cache_key'];output.mkdir(parents=True,exist_ok=True)
            np.save(output/'geometry.npy',values[ids])
        if rows[0]['split']=='train':
            v=values.astype(np.float64);count+=len(v);sums+=v.sum(0);squares+=(v*v).sum(0)
        timings.append(dict(scene=scene,suite=suite,variant=variant,points=len(xyz),cuda_s=elapsed,wall_s=perf_counter()-tick))
        print(f'voxel statistics {scene} v{variant}: points={len(xyz)} cuda_s={elapsed:.4f}',flush=True)
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0))
    metadata.update(geometry_mean=mean.tolist(),geometry_std=np.where(std==0,1,std).tolist(),
        wall_s=perf_counter()-started,statistics_cuda_s=cuda_s,sample_timings=timings,complete=True)
    write_json(root/'encoding.json',metadata)
    print(f"Voxel cache complete: {len(metadata['records'])} boxes, CUDA {cuda_s:.3f}s, wall {metadata['wall_s']:.3f}s",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('Filter/configs/train_scoring_voxel.json'))
    args=parser.parse_args();build_cache(read_json(workspace_path(args.config)))


if __name__=='__main__': main()
