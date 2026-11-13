"""Fixed-seed reconstruction evaluation and map-frame PCD exports."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import trimesh
from common.pointcloud_io import write_pcd
from utils.pointcloud import read_json,write_json,workspace_path
from Reconstruction.data.dataset import Windows,on_device,geometry_batch
from Reconstruction.data.geometry import MeshQueries
from Reconstruction.models.model import Autoencoder,FlowModel
from Reconstruction.training.metrics import geometry_metrics,summarize
from Reconstruction.training.loss import geometry_loss
from Reconstruction.data.prepare import input_indices
from Reconstruction.visualization.plots import comparison


def load_models(checkpoint,device):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ae=Autoencoder().to(device);ae.load_state_dict(ckpt['ae']);ae.eval()
    flow=None
    if ckpt['stage']=='flow':
        flow=FlowModel().to(device);flow.load_state_dict(ckpt['model']);flow.eval()
    return ae,flow,ckpt


@torch.no_grad()
def evaluate(ae,flow,dataset,config,output=None,steps=16,seed=42,vst=False):
    device=next(ae.parameters()).device;rows=[]
    for i in range(len(dataset)):
        raw=dataset[i];s=on_device(raw,device);row=s['row'];folder=s['folder']
        torch.cuda.synchronize();start=perf_counter();timings={}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']=='bf16'):
            if flow is None:
                z=ae.encode(s)
                torch.cuda.synchronize();timings['encoding_s']=perf_counter()-start
            else:
                condition=flow.encode_condition(s)
                torch.cuda.synchronize();timings['condition_s']=perf_counter()-start;tick=perf_counter()
                z=flow.sample(condition,steps,seed)
                torch.cuda.synchronize();timings['flow_s']=perf_counter()-tick
            tick=perf_counter()
            field=ae.decoder(z,s['origin'])
            kept=raw['xyz'] if flow is not None else np.empty((0,3),np.float32)
            pred=field.generate(row['core_lower'],row['core_upper'],kept,
                levels=config['levels_m'],threshold=config['surface_threshold'],expansion_threshold=config['expansion_threshold'])
            torch.cuda.synchronize();timings['decode_generate_s']=perf_counter()-tick
            duration=perf_counter()-start
            # GT queries/loss are evaluation work, outside inference timing.
            stage='ae' if flow is None else 'flow'
            loss_config=config['loss_settings'] if 'loss_settings' in config else read_json(workspace_path(config['loss_config']))
            heldout_queries=geometry_batch(folder,config['levels_m'],config['queries_per_level'],np.random.default_rng(seed),device,(row['core_lower'],row['core_upper']),stage,raw['global_index'])
            heldout_loss,_=geometry_loss(field,heldout_queries,stage,loss_config)
        core=((kept>=row['core_lower'])&(kept<row['core_upper'])).all(1);kept=kept[core]
        kept_ids=raw['global_index'][core] if flow is not None else np.empty(0,np.int64)
        reference=np.load(folder/'reference.npy')
        terrain_xyz=np.load(workspace_path(row['terrain_folder'])/'terrain.npy',mmap_mode='r')
        ideal=terrain_xyz[input_indices(row)] if flow is not None else np.empty((0,3))
        scene=read_json(folder/'scene.json');mesh=trimesh.load(workspace_path(row['terrain_folder'])/'terrain.obj',force='mesh',process=False)
        semantic=np.load(folder/'reference_semantic.npy') if (folder/'reference_semantic.npy').exists() else None
        metrics=geometry_metrics(reference,pred['xyz'],raw['xyz'] if flow is not None else kept,ideal,MeshQueries(mesh),scene,semantic=semantic)
        metrics.update(mode='completion' if flow is not None else 'autoencoder',scene=row['scene'],kind=row['kind'],sample_id=row['sample_id'],recipe='global_subtraction',severity=row['core_removed_fraction'],mask_id=row['mask_id'],difficulty=row['difficulty'],geometry_loss=float(heldout_loss),forward_and_generation_s=duration,timings=timings,timing_scope='cached PTv3/Filter features; excludes GT queries, metrics and plotting',steps=steps if flow is not None else 0,seed=seed)
        if vst:
            from Reconstruction.training.vst_metrics import compare_vst
            metrics['vst']=compare_vst(np.load(folder/'complete.npy'),ideal if flow is not None else raw['xyz'],np.concatenate([raw['xyz'] if flow is not None else np.empty((0,3),np.float32),pred['xyz']]),core_bounds=[row['core_lower'],row['core_upper']])
        rows.append(metrics)
        if output is not None:
            out=Path(output)/row['sample_id'];out.mkdir(parents=True,exist_ok=True)
            xyz=np.concatenate([kept,pred['xyz']]);source=np.r_[np.zeros(len(kept)),np.ones(len(pred['xyz']))]
            tick=perf_counter()
            write_pcd(out/'repaired.pcd',xyz,source,field_name='source')
            np.savez(out/'point_attributes.npz',source=source.astype(np.uint8),
                     input_index=np.r_[kept_ids,np.full(len(pred['xyz']),-1)],
                     generated_cells=pred['cells'])
            metrics['timings']['pcd_write_s']=perf_counter()-tick
            metrics['retained_xyz_exact']=bool(np.array_equal(xyz[:len(kept)],raw['xyz'][core])) if flow is not None else None
            write_pcd(out/'generated.pcd',pred['xyz'],pred['probability'],field_name='surface_probability')
            np.savez(out/'generated.npz',**pred)
            write_json(out/'metrics.json',metrics)
            shown_input=kept if flow is not None else raw['xyz'][((raw['xyz']>=row['core_lower'])&(raw['xyz']<row['core_upper'])).all(1)]
            label='Input retained' if flow is not None else 'Complete input (AE)'
            caption=f"{row['mask_id']} / core missing {row['core_removed_fraction']:.1%}" if flow is not None else 'Geometry autoencoder'
            comparison(out/'comparison.png',[(label,shown_input,'#4466dd'),('Repaired: blue observed / green new',xyz,dict(observed_count=len(kept))),('Complete GT',reference,'#e09035')],f"{row['kind']} | {caption} | F1 {metrics['f1_2cm']:.3f}")
        print(f"eval {row['sample_id']} | generated={len(pred['xyz'])} F1={metrics['f1_2cm']:.4f} recall={metrics['recall_2cm']:.4f}",flush=True)
    report=summarize(rows);report['rows']=rows
    report['geometry_loss']=float(np.mean([r['geometry_loss'] for r in rows]))
    report['by_kind']={name:summarize([r for r in rows if r['kind']==name]) for name in sorted({r['kind'] for r in rows})}
    for key in ['difficulty','mask_id']:
        report[f'by_{key}']={n:summarize([r for r in rows if r[key]==n]) for n in sorted({r[key] for r in rows})}
    if output is not None:write_json(Path(output)/'metrics.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--steps',type=int,nargs='+',default=[16]);p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--output',required=True)
    p.add_argument('--vst',action='store_true')
    a=p.parse_args();ae,flow,ckpt=load_models(workspace_path(a.checkpoint),'cuda:0');config=ckpt['config']
    dataset=Windows(workspace_path(config['data']),a.split,ckpt['stage'])
    for steps in a.steps:
        for seed in a.seeds:evaluate(ae,flow,dataset,config,workspace_path(a.output)/f'steps_{steps}_seed_{seed}',steps,seed,vst=a.vst)

if __name__=='__main__':main()
