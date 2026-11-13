"""Single XYZ PCD -> continuous scores at every original point, without deletion."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import point_transformer_v3 as ptv3

from common.pointcloud_io import read_pcd_xyz,write_pcd
from Filter.data.features import encode_core_tensors,geometry_features
from Filter.data.voxel_statistics import voxel_statistics
from Filter.data.voxel import representatives
from Filter.models.filter import geometry_mode
from Filter.training.evaluate import load_head
from utils.pointcloud import spatial_blocks,write_json,workspace_path


@torch.inference_mode()
def predict(checkpoint,pcd,output,knn_backend='cuda'):
    started=perf_counter();device=torch.device('cuda:0')
    head,ckpt=load_head(checkpoint,device);c=ckpt['config']
    mode=geometry_mode(c)
    if mode=='knn' and knn_backend=='cuda':
        from Filter.data.gpu_knn import geometry_features_cuda
    encoder=ptv3.PointTransformerV3.from_preset(c['encoder']['name']);encoder.precision='fp32'
    encoder.load_state_dict(ckpt['encoder']);encoder.to(device).eval();encoder.requires_grad_(False)
    tick=perf_counter();xyz=read_pcd_xyz(pcd);read_s=perf_counter()-tick
    original=xyz
    tick=perf_counter();original_inverse=None
    if c['input_voxel_m'] is not None:
        selected,original_inverse=representatives(original,c['input_voxel_m']);xyz=original[selected]
    downsample_s=perf_counter()-tick
    timing=dict(read_pcd_s=read_s,downsample_s=downsample_s,geometry_s=0.,geometry_cuda_s=0.,encode_s=0.,head_s=0.,encode_wall_s=0.,head_wall_s=0.);windows=0
    torch.cuda.reset_peak_memory_stats()
    tick=perf_counter();xyz_cuda=torch.as_tensor(xyz,device=device);torch.cuda.synchronize()
    timing['upload_s']=perf_counter()-tick
    score_cuda=torch.empty(len(xyz),device=device,dtype=torch.float32)
    if mode=='voxel':
        tick=perf_counter();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();statistics=voxel_statistics(xyz_cuda,c['geometry']['voxel_sizes_m']);end.record();end.synchronize()
        timing['geometry_s']=perf_counter()-tick;timing['geometry_cuda_s']=begin.elapsed_time(end)/1000.
    events=[];geometry_events=[]
    for ids,core,_ in spatial_blocks(xyz,c['window']['core_size_m'],c['window']['context_margin_m']):
        local=np.flatnonzero(core)
        if mode=='knn' and knn_backend=='cpu':
            tick=perf_counter();geometry=geometry_features(xyz[ids],local,c['geometry']['representative_voxel_m']);timing['geometry_s']+=perf_counter()-tick
        ids_cuda=torch.as_tensor(ids,device=device);local_cuda=torch.as_tensor(local,device=device)
        point_ids=ids_cuda[local_cuda];context=xyz_cuda[ids_cuda]
        if mode=='knn' and knn_backend=='cuda':
            tick=perf_counter();geometry_begin,geometry_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            geometry_begin.record();geometry=geometry_features_cuda(context,local_cuda,c['geometry']['representative_voxel_m']);geometry_end.record()
            timing['geometry_s']+=perf_counter()-tick;geometry_events.append((geometry_begin,geometry_end))
        encode_begin,encode_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        head_begin,head_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=perf_counter();encode_begin.record()
        feat,inverse,relative=encode_core_tensors(encoder,context,local_cuda,c)
        encode_end.record();timing['encode_wall_s']+=perf_counter()-tick
        tick=perf_counter()
        head_begin.record();batch=dict(features=feat,inverse=inverse,relative=relative)
        if mode=='knn': batch['geometry']=torch.as_tensor(geometry,device=device)
        elif mode=='voxel': batch['geometry']=statistics[point_ids]
        with torch.autocast('cuda',dtype=torch.bfloat16): score_cuda[point_ids]=head(batch)
        head_end.record();timing['head_wall_s']+=perf_counter()-tick;windows+=1
        events.append((encode_begin,encode_end,head_begin,head_end))
        if windows%50==0: print(f'predict windows={windows} elapsed={perf_counter()-started:.1f}s',flush=True)
    tick=perf_counter();score=score_cuda.cpu().numpy();timing['download_s']=perf_counter()-tick
    timing['encode_s']=sum(a.elapsed_time(b) for a,b,_,_ in events)/1000.
    timing['head_s']=sum(a.elapsed_time(b) for _,_,a,b in events)/1000.
    if geometry_events: timing['geometry_cuda_s']=sum(a.elapsed_time(b) for a,b in geometry_events)/1000.
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    network_points=len(xyz)
    if original_inverse is not None: score=score[original_inverse]
    xyz=original;tick=perf_counter()
    write_pcd(output/'prediction.pcd',xyz,score,field_name='keep_score')
    np.savez(output/'prediction.npz',coord=xyz,keep_score=score,point_index=np.arange(len(xyz)))
    timing['save_s']=perf_counter()-tick;timing['total_s']=perf_counter()-started
    report=dict(input=str(pcd),checkpoint=str(checkpoint),points=len(xyz),output_points=len(xyz),hard_filtering=False,use_knn=mode=='knn',geometry_mode=mode,
        knn_backend=knn_backend if mode=='knn' else None,
        statistics_passes_per_pcd=1 if mode=='voxel' else 0,cuda_event_encode_and_head_timing=True,
        network_points=network_points,input_voxel_m=c['input_voxel_m'],voxel_scores_restored_to_original_points=True,threshold=None,gt_used=False,coordinate_frame='map',xyz_and_order_preserved=True,windows=windows,
        timing=timing,points_per_s=len(xyz)/timing['total_s'],cuda_peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    if mode=='knn' and knn_backend=='cuda':
        import cupy as cp
        report['cupy_version']=cp.__version__
        report['cupy_pool_reserved_GiB']=cp.get_default_memory_pool().total_bytes()/2**30
        free,total=torch.cuda.mem_get_info();report['device_used_GiB_including_other_processes']=(total-free)/2**30
    write_json(output/'prediction.json',report);print(report,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--pcd',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--knn-backend',choices=['cuda','cpu'],default='cuda',help='Original six KNN features: exact CUDA KDTree or explicit CPU reference')
    a=p.parse_args();predict(workspace_path(a.checkpoint),workspace_path(a.pcd),workspace_path(a.output),a.knn_backend)
if __name__=='__main__': main()
