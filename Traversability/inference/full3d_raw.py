"""Complete input PCD XYZ -> complete per-input-point conditional cost.

The pretrained encoder still consumes 5 cm real representatives in complete
3D contexts. Each original point queries its nearest representative through
the existing GPU hash, then uses the encoder inverse and the unchanged head.
No Tomogram positions, slices or teacher costs are used by this entrypoint.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import point_transformer_v3 as ptv3
import torch

from common.pointcloud_io import read_pcd_xyz, write_pcd
from Traversability.data.voxel_hash import representatives, VoxelHash
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.full3d_cache import clock, encode_context
from utils.pointcloud import workspace_path, write_json


@torch.inference_mode()
def predict(checkpoint, pcd, condition_values, output, device='cuda:0'):
    started=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=deepcopy(state['config']);config['device']=device
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    output=workspace_path(output);output.mkdir(parents=True,exist_ok=True)
    pcd=workspace_path(pcd);write_json(output/'config.json',config)
    tick=time.perf_counter();coord=read_pcd_xyz(pcd);reading_s=time.perf_counter()-tick
    raw=torch.from_numpy(coord).to(device)
    tick=clock()
    points,original,keys,low,shape=representatives(raw,config['geometry_voxel_size_m'])
    voxel_s=clock()-tick;tick=clock()
    table=VoxelHash(points,original,keys,low,shape,config['geometry_voxel_size_m'])
    source=torch.empty(len(raw),device=device,dtype=torch.long)
    maximum_match=0.
    for first in range(0,len(raw),262144):
        last=min(first+262144,len(raw))
        matched,distance,radius=table.query(raw[first:last])
        source[first:last]=matched
        maximum_match=max(maximum_match,float(distance.max()))
    matching_s=clock()-tick
    np.save(output/'query_original_index.npy',original[source].cpu().numpy())
    del table,keys,original,matched,distance,radius
    tick=clock();size=torch.tensor(config['window']['core_size_m'],device=device)
    origin=torch.floor(raw.amin(0)/size)*size
    cells=torch.floor((raw-origin)/size).long();grid_shape=cells.amax(0)+1
    keys=(cells[:,0]*grid_shape[1]+cells[:,1])*grid_shape[2]+cells[:,2]
    order=torch.argsort(keys,stable=True)
    unique,counts=torch.unique_consecutive(keys[order],return_counts=True)
    core_keys=unique.cpu().tolist();core_counts=counts.cpu().tolist();shape_yz=grid_shape.cpu().tolist()[1:]
    del cells,keys,unique,counts,raw
    grouping_s=clock()-tick
    encoder=ptv3.PointTransformerV3.from_preset(config['encoder']['name']).to(device).eval()
    encoder.precision='fp32';encoder.load_state_dict(state['encoder']);encoder.requires_grad_(False)
    head=SlicedCostHead(config).to(device).eval();head.load_state_dict(state['model']);head.requires_grad_(False)
    condition=torch.tensor(condition_values,device=device,dtype=torch.float32)[None,None]
    prediction=np.empty(len(coord),np.float32);encoding_s=head_s=0.;first=0
    context=config['window']['context_m'];torch.cuda.reset_peak_memory_stats()
    for number,(key,count) in enumerate(zip(core_keys,core_counts),1):
        tick=clock()
        cell=[key//(shape_yz[0]*shape_yz[1]),(key//shape_yz[1])%shape_yz[0],key%shape_yz[1]]
        lower=origin+torch.tensor(cell,device=device)*size
        indices=torch.nonzero(((points>=lower-context)&(points<lower+size+context)).all(1)).flatten()
        query_indices=order[first:first+count];selected=source[query_indices]
        local=torch.searchsorted(indices,selected)
        assert bool((local<len(indices)).all()) and torch.equal(indices[local],selected)
        grouping_s+=clock()-tick;tick=clock()
        feature,back,_=encode_context(encoder,points[indices],local,config)
        encoding_s+=clock()-tick;tick=clock()
        batch=torch.zeros(len(feature),device=device,dtype=torch.long)
        values=head(feature,condition,batch)[:,0]
        prediction[query_indices.cpu().numpy()]=values[back.long()].cpu().numpy()
        head_s+=clock()-tick;first+=count
        if number%20==0 or number==len(core_keys):
            row=dict(stage='predicting',blocks_completed=number,blocks_total=len(core_keys),
                points_completed=first,points_total=len(coord),elapsed_s=time.perf_counter()-started)
            write_json(output/'progress.json',row)
            print(f'RAW COST {number}/{len(core_keys)} | {first:,}/{len(coord):,} points',flush=True)
    assert first==len(coord) and np.isfinite(prediction).all()
    tick=clock();write_pcd(output/'prediction.pcd',coord,prediction)
    np.save(output/'prediction.npy',prediction);saving_s=time.perf_counter()-tick
    actual=read_pcd_xyz(output/'prediction.pcd');same_xyz=np.array_equal(coord,actual)
    assert same_xyz
    report=dict(checkpoint=str(Path(checkpoint).resolve()),selected_epoch=state['epoch'],
        source_pcd=str(pcd),points=len(coord),raw_points=len(coord),geometry_points=len(points),
        encoder=config['encoder']['name'],condition=condition_values,coordinate_frame='map',
        output_coordinates='all original input PCD XYZ in original row order, map metres',
        original_xyz_and_order_equal=same_xyz,output_query_source='input_pcd_xyz',
        slices_computed=False,tomogram_used=False,teacher_cost_computed=False,external_queries_used=False,
        cpu_kdtree_used=False,encoder_calls=len(core_keys),maximum_representative_match_m=maximum_match,
        cost_range=[float(prediction.min()),float(prediction.max())],peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        timing=dict(pcd_read_s=reading_s,gpu_voxel_s=voxel_s,gpu_correspondence_s=matching_s,
            grouping_s=grouping_s,encoder_and_feature_mapping_s=encoding_s,head_s=head_s,output_save_s=saving_s),
        total_s=time.perf_counter()-started,
        supervision_scope='Checkpoint trained on support-query costs. Other raw surfaces are direct model predictions.')
    write_json(output/'prediction.json',report);write_json(output/'progress.json',dict(status='complete',**report))
    print(output/'prediction.pcd',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--pcd',type=Path,required=True)
    parser.add_argument('--condition',type=float,nargs=4,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    predict(args.checkpoint,args.pcd,args.condition,args.output,args.device)


if __name__=='__main__':main()
