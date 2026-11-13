"""Legacy QueryPool on the unchanged full3D held-out queries and condition banks."""
import argparse
from pathlib import Path
import time

import numpy as np

from utils.pointcloud import read_json, write_json, workspace_path


def prepare(run):
    from scipy.spatial import cKDTree
    from common.pointcloud_io import read_pcd_xyz
    from Traversability.data.blocks import query_blocks
    from utils.pointcloud import voxel_downsample

    config=read_json(run/'config.json');data=workspace_path(config['data'])
    scene=next(s for s in read_json(data/'manifest.json')['scenes'] if s['split']=='test')
    settings=read_json(workspace_path('runs/traversability/building_SUST_comparison_20261113/legacy_geometry_config.json'))
    output=run/'legacy_test/geometry';output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();inputs=read_pcd_xyz(scene['source_pcd']);query=np.load(data/scene['id']/'coord.npy')
    coarse=voxel_downsample(inputs,settings['global_voxel_m'])
    _,neighbors=cKDTree(coarse).query(query,k=settings['query_neighbors']['global_k'])
    arrays=dict(input_coord=inputs,query_coord=query,global_coord=coarse,global_neighbors=neighbors)
    for name,value in arrays.items():np.save(output/(name+'.npy'),value)
    rows=[]
    for index,(ii,core,qi,lower,upper) in enumerate(query_blocks(inputs,query,settings['core_size_m'],settings['context_m'])):
        _,neighbors=cKDTree(inputs[ii]).query(query[qi],k=settings['query_neighbors']['local_k'])
        mask=neighbors<len(ii);neighbors=np.where(mask,neighbors,0);name=f'block_{index:05d}'
        np.savez(output/(name+'.npz'),input_indices=ii,query_indices=qi,query_neighbors=neighbors,query_neighbor_mask=mask)
        rows.append(dict(id=name,scene='inference',file=name+'.npz',core_lower_m=lower.tolist(),
            core_upper_m=upper.tolist(),input_points=len(ii),query_points=len(qi)))
    record=dict(id='inference',arrays={k:k+'.npy' for k in arrays},input_points=len(inputs),query_points=len(query),global_points=len(coarse))
    write_json(output/'blocks.json',dict(scenes=[record],blocks=rows,preparation_s=time.perf_counter()-started,
        settings=settings,source_pcd=scene['source_pcd'],fixed_scene=scene['id'],same_xyz=True))
    print('Legacy held-out geometry ready',len(rows),flush=True)


def evaluate(run):
    import torch
    from common.pointcloud_io import write_pcd
    from Traversability.data.dataset import collate_queries, to_device
    from Traversability.inference.inference import QueryBlocks
    from Traversability.models.model import ConditionedCost
    from Traversability.training.condition_response import response_totals
    from Traversability.training.decoupling import scene_result, macro
    from Traversability.training.train import autocast, make_loader

    output=run/'legacy_test';config=read_json(run/'config.json');data=workspace_path(config['data'])
    scene=next(s for s in read_json(data/'manifest.json')['scenes'] if s['split']=='test')
    folder=data/scene['id'];columns=scene['banks'][0]+scene['banks'][1]
    conditions=np.load(folder/'conditions.npy')[columns];cost=np.load(folder/'cost.npy',mmap_mode='r')
    facility=np.load(folder/'facility.npy');xyz=np.load(folder/'coord.npy')
    checkpoint=workspace_path('runs/cost_all_stairs_dr_v3/best.pt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);original=state['config']
    geometry=read_json(output/'geometry/blocks.json')
    assert geometry['settings']==state['blocks_config']
    torch.set_num_threads(original['cpu_threads']);torch.manual_seed(original['seed'])
    device=torch.device('cuda');dataset=QueryBlocks(output/'geometry','inference',original['model'],original['seed'],conditions[3])
    dataset.condition=conditions
    model=ConditionedCost(original['model'],state['condition_scale'],pretrained=False).to(device).eval()
    model.load_state_dict(state['model']);model.requires_grad_(False)
    record=dict(absolute=0.,count=0,response=np.zeros((4,10),np.float64))
    regional=np.zeros((2,2),np.float64);threshold=np.zeros(4,np.float64)
    prediction=np.empty(len(xyz),np.float32);response_prediction=np.zeros(40);response_truth=np.zeros(40);response_count=0
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():
        for batch in make_loader(dataset,original,device,collate_fn=collate_queries):
            batch=to_device(batch,device);qi=batch['query_index'].cpu().numpy()
            with autocast(device,original['amp']):values=model(batch)
            truth=torch.from_numpy(np.array(cost[qi][:,columns],np.float32)).to(device)
            region=torch.from_numpy(facility[qi]).to(device);error=abs(values-truth)
            record['absolute']+=float(error.sum());record['count']+=truth.numel()
            for bank in range(2):
                part=slice(40*bank,40*(bank+1))
                record['response']+=response_totals(values[:,part],truth[:,part],scene['pairs'],config['change_threshold']).cpu().numpy()
            low=truth<=.5;high=~low
            threshold += [int(((values>.5)&low).sum()),int(low.sum()),int(((values<=.5)&high).sum()),int(high.sum())]
            for index,mask in enumerate([region,~region]):regional[index]+=[float(error[mask].sum()),int(mask.sum())*truth.shape[1]]
            prediction[qi]=values[:,3].cpu().numpy()
            response_prediction+=values[region,:40].sum(0).cpu().numpy();response_truth+=truth[region,:40].sum(0).cpu().numpy()
            response_count+=int(region.sum())
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    metrics=macro([scene_result(record)])
    metrics.update(facility_mae_raw=float(50*regional[0,0]/regional[0,1]),background_mae_raw=float(50*regional[1,0]/regional[1,1]),
        false_block_pct=float(100*threshold[0]/threshold[1]),false_pass_pct=float(100*threshold[2]/threshold[3]),
        false_block_denominator=int(threshold[1]),false_pass_denominator=int(threshold[3]))
    write_json(output/'metrics.json',dict(checkpoint=str(checkpoint),selected_epoch=state['epoch'],scene=scene['id'],
        same_xyz=bool(np.array_equal(xyz,np.load(output/'geometry/query_coord.npy'))),condition_slots=len(columns),metrics=metrics,
        preparation_s=geometry['preparation_s'],all_conditions_prediction_s=seconds,blocks=len(dataset),
        note='Original QueryPool precision, preprocessing and model. Both original condition banks, no training or selection.'))
    write_json(output/'responses.json',{scene['id']:dict(conditions=conditions[:40].tolist(),
        prediction=(response_prediction/response_count).tolist(),truth=(response_truth/response_count).tolist(),region='facility support points')})
    np.save(output/'prediction.npy',prediction);write_pcd(output/'prediction.pcd',xyz,prediction)
    print('Legacy held-out result',metrics['cost_mae_raw'],metrics['mean_response_error'],flush=True)
    # The historical frozen report did not record threshold errors. Evaluate
    # its unchanged cached features and head, without touching the new run.
    from Traversability.models.sliced import SlicedCostHead
    from Traversability.training.sliced import CachedSlices, loader, device_batch
    frozen=workspace_path(config['frozen_reference_run'])
    frozen_state=torch.load(frozen/'best.pt',map_location='cpu',weights_only=False)
    frozen_config=frozen_state['config'];head=SlicedCostHead(frozen_config).to(device).eval()
    head.load_state_dict(frozen_state['model']);baseline=read_json(frozen/'summary.json');threshold_metrics={}
    del model
    with torch.inference_mode():
        for split,key in [('val','best_validation'),('test','test')]:
            sums=np.zeros(6,np.float64)
            for bank in range(2):
                dataset=CachedSlices(frozen_config,split,bank)
                for batch in loader(dataset,frozen_config):
                    b=device_batch(batch,frozen_config)
                    values=head(b['features'],b['conditions'],b['voxel_batch'])[b['inverse']];truth=b['target']
                    low=truth<=.5;high=~low
                    sums += [int(((values>.5)&low).sum()),int(low.sum()),int(((values<=.5)&high).sum()),int(high.sum()),float(abs(values-truth).sum()),truth.numel()]
            result=dict(false_block_pct=float(100*sums[0]/sums[1]),false_pass_pct=float(100*sums[2]/sums[3]),
                false_block_denominator=int(sums[1]),false_pass_denominator=int(sums[3]),cost_mae_raw=float(50*sums[4]/sums[5]))
            assert abs(result['cost_mae_raw']-baseline[key]['cost_mae_raw'])<1e-5
            threshold_metrics[key]=result
    write_json(run/'frozen_thresholds.json',threshold_metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    (prepare if args.prepare else evaluate)(args.run.resolve())


if __name__=='__main__':main()
