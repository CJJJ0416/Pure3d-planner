"""PCD geometry queries with complete 3D pretrained-encoder contexts and conditional cost."""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np
import point_transformer_v3 as ptv3
import torch

from common.pointcloud_io import read_pcd_xyz, write_pcd
from Traversability.data.voxel_hash import representatives, VoxelHash
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.full3d_cache import clock, encode_context
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


@torch.inference_mode()
def predict(checkpoint, pcd, condition_values, output, state=None):
    started=time.perf_counter()
    state=state or torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=state['config']
    torch.set_num_threads(config['cpu_threads'])
    output=workspace_path(output);output.mkdir(parents=True,exist_ok=True)
    pcd=workspace_path(pcd)
    write_json(output/'config.json',config)
    tick=time.perf_counter()
    subprocess.run([config['geometry_python'],'-m','Traversability.data.sliced',
        '--config',str(output/'config.json'),'--pcd',str(pcd),'--output',str(output/'geometry')],cwd=ROOT,check=True)
    geometry_s=time.perf_counter()-tick
    metadata=read_json(output/'geometry/geometry.json')
    coord=np.load(output/'geometry/coord.npy')
    tick=clock()
    raw=read_pcd_xyz(pcd)
    reading_s=clock()-tick
    raw_count=len(raw)
    tick=clock()
    points,original,keys,low,shape=representatives(torch.from_numpy(raw).to(config['device']),config['geometry_voxel_size_m'])
    voxel_s=clock()-tick
    del raw
    query=torch.from_numpy(coord).to(config['device'])
    tick=clock()
    table=VoxelHash(points,original,keys,low,shape,config['geometry_voxel_size_m'])
    source,distance,radius=table.query(query)
    matching_s=clock()-tick
    np.save(output/'geometry/query_original_index.npy',original[source].cpu().numpy())
    np.save(output/'geometry/query_match_distance.npy',distance.cpu().numpy())
    del table,keys,distance,radius,original
    encoder=ptv3.PointTransformerV3.from_preset(config['encoder']['name']).to(config['device']).eval()
    encoder.precision='fp32'
    encoder.load_state_dict(state['encoder']);encoder.requires_grad_(False)
    head=SlicedCostHead(config).to(config['device']).eval();head.load_state_dict(state['model'])
    condition=torch.tensor(condition_values,device=config['device'],dtype=torch.float32)[None,None]
    prediction=np.empty(len(coord),np.float32)
    size=torch.tensor(config['window']['core_size_m'],device=config['device'])
    context=config['window']['context_m']
    grouping_s=encoding_s=head_s=0.
    for row in metadata['blocks']:
        tick=clock()
        lower=torch.tensor(row['lower'],device=config['device'])
        index=torch.nonzero(((points>=lower-context)&(points<lower+size+context)).all(1)).flatten()
        with np.load(output/'geometry'/(row['id']+'.npz')) as block:
            point_index=block['point_index'].copy()
        q=torch.from_numpy(point_index).to(config['device'])
        local=torch.searchsorted(index,source[q])
        assert bool((local<len(index)).all()) and torch.equal(index[local],source[q])
        grouping_s+=clock()-tick
        tick=clock()
        feature,back,_=encode_context(encoder,points[index],local,config)
        encoding_s+=clock()-tick
        tick=clock()
        batch=torch.zeros(len(feature),device=config['device'],dtype=torch.long)
        values=head(feature,condition,batch)[:,0]
        prediction[point_index]=values[back.long()].cpu().numpy()
        head_s+=clock()-tick
    write_pcd(output/'prediction.pcd',coord,prediction)
    np.save(output/'prediction.npy',prediction)
    report=dict(checkpoint=str(Path(checkpoint).resolve()),source_pcd=str(pcd),condition=condition_values,
        points=len(coord),raw_points=raw_count,geometry_points=len(points),architecture=state['architecture'],
        selected_epoch=state['epoch'],output_coordinates='native geometry support XYZ, map metres',
        encoder=config['encoder']['name'],
        encoder_input='complete 3D PCD representative context, not Tomogram points',encoder_calls=len(metadata['blocks']),
        teacher_cost_computed=False,external_queries_used=False,cpu_kdtree_used=False,
        timing=dict(native_geometry_s=geometry_s,pcd_read_s=reading_s,gpu_voxel_s=voxel_s,
            gpu_correspondence_s=matching_s,grouping_s=grouping_s,tiny_encoding_s=encoding_s,head_s=head_s),
        total_s=time.perf_counter()-started)
    write_json(output/'prediction.json',report)
    print(output/'prediction.pcd',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--pcd',type=Path,required=True)
    parser.add_argument('--condition',type=float,nargs=4,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    predict(args.checkpoint,args.pcd,args.condition,args.output)


if __name__=='__main__':
    main()
