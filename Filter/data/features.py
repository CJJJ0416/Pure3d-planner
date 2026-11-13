"""Encode context once, cache only the core's unprojected 486-channel features."""
import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset,DataLoader
import point_transformer_v3 as ptv3
from Filter.models.filter import geometry_mode

from utils.pointcloud import read_json,write_json,workspace_path


@lru_cache(maxsize=2)
def variant_arrays(folder, variant):
    folder=Path(folder)
    root=folder/f'arrays_{variant:03d}'
    return ({name:np.load(root/f'{name}.npy',mmap_mode='r') for name in ['coord','keep_target','loss_group','semantic_id']},
        np.load(folder/f'windows_{variant:03d}_index.npy',mmap_mode='r'),
        np.load(folder/f'windows_{variant:03d}_core.npy',mmap_mode='r'),
        np.load(folder/f'windows_{variant:03d}_offsets.npy',mmap_mode='r'))


def geometry_features(context, core, voxel=.03, workers=1):
    if voxel is None:
        representatives=context
        selected=np.asarray(core);back=np.arange(len(core))
    else:
        cells=np.floor(context.astype(np.float64)/voxel).astype(np.int64)
        _,first,inverse=np.unique(cells,axis=0,return_index=True,return_inverse=True)
        representatives=context[first]
        selected,back=np.unique(inverse[core],return_inverse=True)
    if len(representatives)==1:
        return np.zeros((len(core),6),np.float32)
    tree=cKDTree(representatives)
    q=representatives[selected]
    distances=tree.query(q,k=min(65,len(representatives)),workers=workers)[0][:,1:]
    d16=distances[:,min(15,distances.shape[1]-1)]
    d64=distances[:,-1]
    mean=distances[:,:16].mean(1)
    counts=[tree.query_ball_point(q,r,return_length=True,workers=workers)-1 for r in [.15,.5]]
    values=np.column_stack([mean,d16,d64,*counts,mean/d64]).astype(np.float32)
    return values[back]


class ContextWindows(Dataset):
    def __init__(self, records, voxel=.03, use_knn=True):
        self.records,self.voxel,self.use_knn=records,voxel,use_knn
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        arrays,indices,core,offsets=variant_arrays(row['folder'],row['variant'])
        a,b=offsets[row['id']:row['id']+2]
        ids=indices[a:b]
        local_core=np.flatnonzero(core[a:b])
        xyz=np.array(arrays['coord'][ids])
        sample=dict(index=index,coord=xyz,core=local_core,point_index=np.array(ids[local_core]))
        if self.use_knn: sample['geometry']=geometry_features(xyz,local_core,self.voxel)
        return sample


def identity(value): return value


def encode_core_tensors(encoder,coord,core,config):
    """Encode resident GPU XYZ, retaining feature tensors for the score head."""
    e=config['encoder']
    device=coord.device
    zero=torch.zeros_like(coord)
    point,inverse=ptv3.prepare(coord,zero,zero,preset=e['name'],grid_size=e['grid_size'],scale=e['scale'],device=device)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[e['precision']]
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype,enabled=dtype!=torch.float32):
        features=ptv3.upcast(encoder(point)).feat
    core_t=torch.as_tensor(core,device=device)
    original_inverse=inverse[core_t]
    used,core_inverse=torch.unique(original_inverse,sorted=True,return_inverse=True)
    scaled=(coord.double()*e['scale']).float()
    low,high=scaled.amin(0),scaled.amax(0)
    shift=torch.stack([(low[0]+high[0])/2,(low[1]+high[1])/2,low[2]])
    relative=(scaled[core_t]-shift-point.coord[original_inverse])/e['grid_size']
    return features[used].to(torch.float16),core_inverse,relative


def encode_core(encoder,coord,core,config,device):
    values=encode_core_tensors(encoder,torch.as_tensor(coord,device=device),core,config)
    return tuple(value.cpu().numpy() for value in values)


def scene_records(data):
    result=[]
    manifest=read_json(data/'manifest.json')
    for scene in manifest['scenes']:
        folders=[('nominal',data/scene['scene_id'])]
        if scene['split']=='test':
            folders += [(name,data/scene['scene_id']/'suites'/name)
                for name in manifest.get('test_suites',['nominal','sparse','few_positions']) if name!='nominal']
        for suite,folder in folders:
            for sample in read_json(folder/'boxes.json')['samples']:
                row=read_json(folder/sample['path']/'box.json')
                result.append({**row,'scene':scene['scene_id'],'suite':suite,'folder':str(folder.resolve()),'split':scene['split'],
                    'key':f"{scene['scene_id']}/{suite}/v{sample['variant']:03d}/b{row['id']:05d}"})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('Filter/configs/train_scoring_no_knn.json'))
    args=parser.parse_args();config=read_json(workspace_path(args.config))
    if config.get('geometry_mode')=='voxel':
        from Filter.data.voxel_statistics import build_cache
        build_cache(config)
        return
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    data,cache=workspace_path(config['data']),workspace_path(config['cache']);cache.mkdir(parents=True,exist_ok=True)
    records=scene_records(data)
    # Workers load arrays and optionally compute the explicit CPU reference.
    use_knn=geometry_mode(config)=='knn'
    knn_backend=config.get('knn_backend','cuda')
    loader=DataLoader(ContextWindows(records,config['geometry']['representative_voxel_m'],use_knn and knn_backend=='cpu'),batch_size=None,num_workers=config['num_workers'],collate_fn=identity,
        prefetch_factor=2,persistent_workers=True)
    iterator=iter(loader)
    device=torch.device(config['device'])
    encoder=ptv3.load(config['encoder']['name'],weights=workspace_path(config['encoder']['weights']),
        device=device,head=False,precision='fp32',warmup=False).eval()
    encoder.requires_grad_(False)
    torch.save(encoder.state_dict(),cache/'encoder.pth')
    if use_knn and knn_backend=='cuda':
        from Filter.data.gpu_knn import geometry_features_cuda
    metadata=dict(config=config,use_knn=use_knn,knn_backend=knn_backend if use_knn else None,encoder_info=encoder.weights_info,feature_channels=486,
        core_only=True,encoder_passes_per_context=1,records=[],geometry_mean=None,geometry_std=None)
    write_json(cache/'encoding.json',metadata)
    started=perf_counter();total_count=0;sums=np.zeros(6);squares=np.zeros(6)
    for sample in iterator:
        row=records[sample['index']]
        tick=perf_counter()
        if use_knn and knn_backend=='cuda':
            context=torch.as_tensor(sample['coord'],device=device)
            core=torch.as_tensor(sample['core'],device=device)
            sample['geometry']=geometry_features_cuda(context,core,config['geometry']['representative_voxel_m']).cpu().numpy()
            feature,inverse,relative=(v.cpu().numpy() for v in encode_core_tensors(encoder,context,core,config))
        else:
            feature,inverse,relative=encode_core(encoder,sample['coord'],sample['core'],config,device)
        output=cache/row['key'];output.mkdir(parents=True,exist_ok=True)
        arrays,*_=variant_arrays(row['folder'],row['variant'])
        ids=sample['point_index']
        values=dict(features=feature,inverse=inverse,relative=relative,point_index=ids,
            target=np.array(arrays['keep_target'][ids]),group=np.array(arrays['loss_group'][ids]),
            semantic=np.array(arrays['semantic_id'][ids]))
        if use_knn: values['geometry']=sample['geometry']
        for name,value in values.items(): np.save(output/f'{name}.npy',value)
        if use_knn and row['split']=='train':
            v=sample['geometry'].astype(np.float64);total_count+=len(v);sums+=v.sum(0);squares+=(v*v).sum(0)
        row.update(cache_key=row['key'],core_voxels=len(feature),encode_and_save_s=perf_counter()-tick)
        metadata['records'].append(row)
        if len(metadata['records'])%50==0:
            write_json(cache/'encoding.json',metadata)
            print(f"features {len(metadata['records'])}/{len(records)} | elapsed={perf_counter()-started:.1f}s",flush=True)
    if use_knn:
        mean=sums/total_count
        std=np.sqrt(np.maximum(squares/total_count-mean**2,0))
        metadata.update(geometry_mean=mean.tolist(),geometry_std=np.where(std==0,1,std).tolist())
    metadata.update(wall_s=perf_counter()-started,core_points=sum(r['core_points'] for r in records),complete=True)
    write_json(cache/'encoding.json',metadata)
    print(f"Feature cache complete | {len(records)} boxes | {metadata['wall_s']:.1f}s",flush=True)


if __name__=='__main__': main()
