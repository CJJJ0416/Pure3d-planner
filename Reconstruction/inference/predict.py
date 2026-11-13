"""One bounded XYZ PCD -> frozen frontend -> AE probe or conditional completion."""
import argparse
import gc
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from common.pointcloud_io import read_pcd_xyz,write_pcd
from utils.pointcloud import workspace_path,write_json,read_json
from Reconstruction.data.features import FrozenFrontend
from Reconstruction.data.dataset import on_device
from Reconstruction.data.grid import input_grid
from Reconstruction.training.evaluate import load_models


@torch.no_grad()
def predict(checkpoint,pcd,output,steps=16,seed=42,block_meta=None):
    started=perf_counter();device=torch.device('cuda:0')
    ae,flow,ckpt=load_models(checkpoint,device);config=ckpt['config']
    torch.set_num_threads(config['cpu_threads'])
    frontend=FrozenFrontend(workspace_path(config['filter_checkpoint']),threshold=config['filter_threshold'])
    tick=perf_counter();xyz=read_pcd_xyz(pcd);timing=dict(read_pcd_s=perf_counter()-tick)
    if block_meta is None:
        origin,shape=input_grid(xyz);core_lower=origin;core_upper=origin.astype(np.float64)+shape*.24
    else:
        grid=read_json(block_meta);origin=np.array(grid['origin'],np.float32);shape=np.array(grid['shape'],int)
        core_lower=np.array(grid['core_lower']);core_upper=np.array(grid['core_upper'])
    tick=perf_counter();features=frontend.encode(xyz,True);torch.cuda.synchronize()
    timing['ptv3_filter_s']=perf_counter()-tick
    kept_ids=features['keep_index'];features.update(origin=origin,shape=shape.tolist())
    del frontend
    gc.collect();torch.cuda.empty_cache()
    sample=on_device(features,device);tick=perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if flow is None:
            latent=ae.encode(sample)
            torch.cuda.synchronize();timing['ae_encode_s']=perf_counter()-tick
        else:
            condition=flow.encode_condition(sample)
            torch.cuda.synchronize();timing['condition_s']=perf_counter()-tick;tick=perf_counter()
            latent=flow.sample(condition,steps,seed)
            torch.cuda.synchronize();timing['flow_s']=perf_counter()-tick
    tick=perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        field=ae.decoder(latent,sample['origin'])
        pred=field.generate(core_lower,core_upper,features['xyz'],
                            config['levels_m'],config['expansion_threshold'],config['surface_threshold'])
    torch.cuda.synchronize();timing['decode_generate_s']=perf_counter()-tick;tick=perf_counter()
    new=pred['xyz'];repaired=np.concatenate([xyz[kept_ids],new])
    source=np.r_[np.zeros(len(kept_ids),np.uint8),np.ones(len(new),np.uint8)]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_pcd(output/'repaired.pcd',repaired,source,field_name='source')
    write_pcd(output/'generated.pcd',new,pred['probability'],field_name='surface_probability')
    write_pcd(output/'retained.pcd',xyz[kept_ids])
    np.savez(output/'point_attributes.npz',source=source,input_index=np.r_[kept_ids,np.full(len(new),-1)],
             generated_cells=pred['cells'],surface_probability=pred['probability'])
    timing['merge_write_s']=perf_counter()-tick;timing['total_s']=perf_counter()-started
    report=dict(checkpoint=str(checkpoint),input=str(pcd),input_points=len(xyz),retained_points=len(kept_ids),
                generated_points=len(new),scope='single_input_block',mode='ae_reconstruction_probe' if flow is None else 'conditional_flow_completion',
                checkpoint_epoch=ckpt['epoch'],checkpoint_step=ckpt['step'],encoder_passes=1,steps=0 if flow is None else steps,seed=seed,
                coordinate_frame='map',origin=origin.tolist(),shape=shape.tolist(),core_lower=core_lower.tolist(),core_upper=core_upper.tolist(),timings=timing,
                retained_xyz_exact=bool(np.array_equal(repaired[:len(kept_ids)],xyz[kept_ids])),
                cuda_allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(output/'report.json',report);print(report,flush=True)
    return repaired


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--pcd',required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--block-meta',help='Prepared block scene.json, preserving the training grid.')
    a=p.parse_args();predict(workspace_path(a.checkpoint),workspace_path(a.pcd),workspace_path(a.output),a.steps,a.seed,workspace_path(a.block_meta) if a.block_meta else None)

if __name__=='__main__':main()
