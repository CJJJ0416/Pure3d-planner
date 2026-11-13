"""Direct R4 support-query inference with the full original building context."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from Traversability.data.dataset import prepare_geometry_sample,collate_queries,to_device
from Traversability.models.model import ConditionedCost
from Traversability.training.train import autocast,make_loader
from utils.pointcloud import ROOT,read_json,write_json

BASE=ROOT/'runs/planner/large_terrain_six_vst_20261111/prediction/geometry'
QUERY=ROOT/'runs/planner/r4_stand_ratio_sweep_20261111/query_geometry'
SWEEP=ROOT/'runs/planner/r4_step_slope_response_20261111'


class R4Queries(Dataset):
    def __init__(self,config,query,conditions):
        meta=read_json(BASE/'blocks.json');self.rows=meta['blocks'];scene=meta['scenes'][0]
        self.input=np.load(BASE/scene['arrays']['input_coord'],mmap_mode='r')
        self.coarse=np.load(BASE/scene['arrays']['global_coord'],mmap_mode='r')
        self.query=query;self.conditions=conditions.astype(np.float32);self.config=config
        self.global_neighbors=np.load(QUERY/'global_neighbors.npy')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with np.load(BASE/row['file']) as a:ii=a['input_indices']
        with np.load(QUERY/f'block_{i:05d}.npz') as a:
            qi=a['query_indices'];knn=a['query_neighbors'];mask=a['query_neighbor_mask']
        origin=(np.array(row['core_lower_m'],np.float32)+np.array(row['core_upper_m'],np.float32))/2
        sample=dict(input_coord=self.input[ii]-origin,query_coord=self.query[qi]-origin,global_coord=self.coarse[:]-origin,
            query_neighbors=knn,global_neighbors=self.global_neighbors[qi],query_neighbor_mask=mask,query_index=qi,
            origin=origin,scene='large_terrain_R4',block=row['id'])
        result=prepare_geometry_sample(sample,self.config['model'],self.config['seed']+i)
        result['conditions']=torch.from_numpy(self.conditions)
        return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=cp['config']
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    a=np.load(SWEEP/'teacher.npz');xyz=np.r_[a['query_xyz'],a['watch_xyz']]
    dataset=R4Queries(config,xyz,a['conditions']);device=torch.device(config['device'])
    model=ConditionedCost(config['model'],cp['condition_scale'],pretrained=False).to(device)
    model.load_state_dict(cp['model']);model.eval()
    result=np.empty((len(xyz),len(a['conditions'])),np.float32)
    for batch in make_loader(dataset,config,device,collate_fn=collate_queries):
        batch=to_device(batch,device)
        with autocast(device,config['amp']):
            fine=model.encode(batch['point']);coarse=model.encode(batch['global_point'])
            if len(batch['query_coord']):
                pieces=[]
                for first in range(0,len(batch['query_coord']),config['model']['point_chunk_size']):
                    last=first+config['model']['point_chunk_size']
                    local=model.local_query(fine,batch['local_index'][first:last],batch['local_relative'][first:last],batch['local_neighbor_mask'][first:last])
                    global_query=model.global_query(coarse,batch['global_index'][first:last],batch['global_relative'][first:last])
                    pieces.append(model.fusion(torch.cat([local,global_query],-1)))
                values=model.decode_cost(torch.cat(pieces),batch['conditions'],batch['query_batch'])
                result[batch['query_index'].cpu().numpy()]=values.cpu().numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,query_xyz=a['query_xyz'],prediction=result[:len(a['query_xyz'])],watch_prediction=result[len(a['query_xyz']):],conditions=a['conditions'])
    print('R4_COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
