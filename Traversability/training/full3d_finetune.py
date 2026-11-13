"""Online complete-3D cost training with Tiny gradients from the first update."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import point_transformer_v3 as ptv3
import torch
from torch.utils.data import Dataset,DataLoader

from common.pointcloud_io import write_pcd
from Traversability.data.full3d_online import prepare_fixed_maps,prepare_maps
from Traversability.data.rotating import fixed_manifest,disk_bytes,generation_seeds
from Traversability.data.sliced import progress
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.decoupling import decoupling_loss,scene_result,macro
from Traversability.training.condition_response import response_totals
from Traversability.training.rotating import random_state,restore_random,phase_learning_rate
from Traversability.training.train import save_checkpoint
from utils.pointcloud import ROOT,read_json,write_json,workspace_path


class OnlineBlocks(Dataset):
    def __init__(self,config,split,bank):
        self.root=workspace_path(config['data']);self.meta=read_json(self.root/'manifest.json')
        self.scenes={s['id']:s for s in self.meta['scenes']}
        self.rows=[r for r in self.meta['blocks'] if r['split']==split]
        self.bank=bank;self.arrays={}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];name=row['scene'];folder=self.root/name
        if name not in self.arrays:
            self.arrays[name]={key:np.load(folder/(key+'.npy'),mmap_mode='r') for key in
                              ['geometry_coord','cost','conditions','facility']}
        arrays=self.arrays[name]
        with np.load(folder/'maps'/(row['id']+'.npz')) as a:
            context=a['context_index'];local=a['query_local'].astype(np.int64);point_index=a['point_index'].copy()
        banks=self.scenes[name]['banks']
        columns=banks[self.bank] if self.bank is not None else banks[0]+banks[1]
        return dict(coord=torch.from_numpy(np.array(arrays['geometry_coord'][context])),
            query_local=torch.from_numpy(local),target=torch.from_numpy(np.array(arrays['cost'][point_index][:,columns],np.float32)),
            conditions=torch.from_numpy(np.array(arrays['conditions'][columns],np.float32)),
            facility=torch.from_numpy(np.array(arrays['facility'][point_index])),point_index=point_index,scene=name,block=row['id'])


def keep_samples(samples):return samples


def loader(data,config,order=None):
    return DataLoader(data,batch_size=config['batch_size'],sampler=order,num_workers=config['num_workers'],
        pin_memory=True,collate_fn=keep_samples)


def encode_online(encoder,coord,local,config):
    """Gradients cross feature gathers/casts. Integer geometry maps remain fixed."""
    coord=coord.to(config['device'],non_blocking=True);local=local.to(config['device'],non_blocking=True)
    zero=torch.zeros_like(coord);spec=config['encoder']
    point,inverse=ptv3.prepare(coord,zero,zero,preset=spec['name'],grid_size=spec['grid_size'],
        scale=spec['scale'],pick='first',device=config['device'])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        feature=ptv3.upcast(encoder(point)).feat
    used,back=torch.unique(inverse[local],sorted=True,return_inverse=True)
    return feature[used].half(),back


def forward_samples(encoder,head,samples,config):
    features=[];inverse=[];voxel_batch=[];point_batch=[];offset=0
    for i,sample in enumerate(samples):
        feature,back=encode_online(encoder,sample['coord'],sample['query_local'],config)
        features.append(feature);inverse.append(back+offset)
        voxel_batch.append(torch.full((len(feature),),i,dtype=torch.long,device=config['device']))
        point_batch.append(torch.full((len(back),),i,dtype=torch.long,device=config['device']))
        offset+=len(feature)
    conditions=torch.stack([s['conditions'] for s in samples]).to(config['device'],non_blocking=True)
    predicted=head(torch.cat(features),conditions,torch.cat(voxel_batch))[torch.cat(inverse)]
    target=torch.cat([s['target'] for s in samples]).to(config['device'],non_blocking=True)
    facility=torch.cat([s['facility'] for s in samples]).to(config['device'],non_blocking=True)
    return predicted,target,facility,torch.cat(point_batch)


@torch.inference_mode()
def evaluate(encoder,head,config,split,export=None,response_path=None):
    encoder.eval();head.eval()
    data=OnlineBlocks(config,split,None);pairs=next(iter(data.scenes.values()))['pairs']
    records=defaultdict(lambda:dict(absolute=0.,count=0,response=np.zeros((4,10),np.float64)))
    regional=np.zeros((2,2),np.float64);threshold=np.zeros(4,np.float64)
    predictions={};curves={}
    for samples in loader(data,config):
        predicted,target,facility,batch=forward_samples(encoder,head,samples,config)
        error=abs(predicted-target)
        for group,mask in enumerate([facility,~facility]):
            regional[group]+=[float(error[mask].sum()),int(mask.sum())*target.shape[1]]
        low=target<=.5;high=~low
        threshold += [int(((predicted>.5)&low).sum()),int(low.sum()),int(((predicted<=.5)&high).sum()),int(high.sum())]
        for i,sample in enumerate(samples):
            name=sample['scene'];take=batch==i;rec=records[name]
            rec['absolute']+=float(error[take].sum());rec['count']+=int(take.sum())*target.shape[1]
            for bank in range(2):
                columns=slice(40*bank,40*(bank+1))
                rec['response']+=response_totals(predicted[take,columns],target[take,columns],pairs,config['change_threshold']).cpu().numpy()
            if export is not None:
                if name not in predictions:predictions[name]=np.empty(data.scenes[name]['points'],np.float32)
                predictions[name][sample['point_index']]=predicted[take,3].cpu().numpy()
            if response_path is not None:
                if name not in curves:curves[name]=dict(prediction=np.zeros(40),truth=np.zeros(40),points=0,
                    conditions=sample['conditions'][:40].numpy().tolist())
                selected=take&facility
                curves[name]['prediction']+=predicted[selected,:40].sum(0).cpu().numpy()
                curves[name]['truth']+=target[selected,:40].sum(0).cpu().numpy()
                curves[name]['points']+=int(selected.sum())
    per_scene={name:scene_result(r) for name,r in records.items()};result=macro(list(per_scene.values()))
    result.update(per_scene=per_scene,facility_mae_raw=float(50*regional[0,0]/regional[0,1]),
        background_mae_raw=float(50*regional[1,0]/regional[1,1]),
        false_block_pct=float(100*threshold[0]/threshold[1]),false_pass_pct=float(100*threshold[2]/threshold[3]),
        false_block_denominator=int(threshold[1]),false_pass_denominator=int(threshold[3]))
    if export is not None:
        export.mkdir(parents=True,exist_ok=True)
        for name,values in predictions.items():
            xyz=np.load(data.root/name/'coord.npy');np.save(export/f'{name}_prediction.npy',values)
            write_pcd(export/f'{name}_prediction.pcd',xyz,values)
    if response_path is not None:
        for row in curves.values():
            row['prediction']=(row['prediction']/row['points']).tolist();row['truth']=(row['truth']/row['points']).tolist()
            row['region']='facility support points'
        write_json(response_path,curves)
    return result


def prepare_pool(config,generation,completed,validation):
    root=workspace_path(config['data']);output=workspace_path(config['output']);path=root/'generation.json'
    local=deepcopy(config);local['progress_context']=dict(completed_epochs=completed,generation=generation,
        generations_total=5,generation_seeds=generation_seeds(config,generation),last_validation=validation)
    write_json(output/'active_config.json',local)
    state=read_json(path) if path.exists() else None
    if state is None or state['index']!=generation or not state['prepared']:
        torch.cuda.empty_cache()
        with (output/f'generation_{generation:02d}.log').open('a') as stream:
            subprocess.run([config['geometry_python'],'-u','-m','Traversability.data.full3d_online',
                '--config',str(output/'active_config.json'),'--generation',str(generation)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        state=read_json(path)
    if not state.get('geometry_mapped',False):
        reports=prepare_maps(local)
        meta=read_json(root/'manifest.json')
        state.update(geometry_mapped=True,training_blocks=sum(r['split']=='train' for r in meta['blocks']),
            temporary_bytes=disk_bytes(root/'active'),maps=reports,features_cached=False)
        write_json(path,state);write_json(output/'generations'/f'{generation:02d}.json',state)
    return state


def gradient_norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt())


def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def train_epoch(encoder,head,optimizer,config,epoch,step,pool,validation):
    data=OnlineBlocks(config,'train',(epoch-1)%2)
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()
    if len(order)%2:order+=order[:1]
    updates=len(order)//2;pairs=next(iter(data.scenes.values()))['pairs'];output=workspace_path(config['output'])
    encoder.train();head.train();encoder.requires_grad_(True)
    totals=defaultdict(float);started=time.perf_counter();torch.cuda.reset_peak_memory_stats();gradients={}
    for number,samples in enumerate(loader(data,config,order),1):
        first=not (output/'first_update.json').exists()
        before={'encoder':cpu_state(encoder),'head':cpu_state(head)} if first else None
        optimizer.zero_grad(set_to_none=True)
        predicted,target,facility,_=forward_samples(encoder,head,samples,config)
        loss,parts=decoupling_loss(predicted,target,facility,pairs,config)
        assert bool(torch.isfinite(loss))
        loss.backward()
        if first or (step+1)%config['log_every_steps']==0:
            gradients={name:gradient_norm(module) for name,module in
                [('encoder',encoder),('adapter',head.adapter),('condition_head',head.condition_head),('cost_head',head.cost_head)]}
            assert all(math.isfinite(v) for v in gradients.values()),gradients
            if first:assert all(v>0 for v in gradients.values()),gradients
        rate=phase_learning_rate(config,epoch-1+number/updates)
        optimizer.param_groups[0]['lr']=rate*config['encoder_learning_rate']/config['learning_rate']
        optimizer.param_groups[1]['lr']=rate
        optimizer.step();step+=1
        if first:
            changes={'encoder':max(float(abs(v-before['encoder'][k]).max()) for k,v in cpu_state(encoder).items())}
            current=cpu_state(head)
            for name in ['adapter','condition_head','cost_head']:
                changes[name]=max(float(abs(v-before['head'][k]).max()) for k,v in current.items() if k.startswith(name+'.'))
            assert all(v>0 for v in changes.values())
            write_json(output/'first_update.json',dict(epoch=epoch,step=step,true_batch=len(samples),conditions=target.shape[1],
                loss=float(loss.detach()),gradients=gradients,max_parameter_change=changes,
                encoder_trainable_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                head_trainable_parameters=sum(p.numel() for p in head.parameters()),frozen_features_used=False))
        for key,value in parts.items():totals[key]+=float(value.detach())
        if step%config['log_every_steps']==0:
            progress(config,'training',completed_epochs=epoch-1,epoch=epoch,step=step,batch=number,batches_total=updates,
                generation=pool['index'],generations_total=5,generation_seeds=pool['seeds'],
                loss={k:v/number for k,v in totals.items()},gradients=gradients,last_validation=validation,
                learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,temporary_bytes=pool['temporary_bytes'])
            print(f'EPOCH {epoch}/50 BATCH {number}/{updates} STEP {step} LOSS {float(loss.detach()):.6f}',flush=True)
    torch.cuda.synchronize()
    return dict(epoch=epoch,step=step,bank=(epoch-1)%2,generation=pool['index'],seeds=pool['seeds'],
        training_blocks=len(data),updates=updates,loss={k:v/updates for k,v in totals.items()},
        elapsed_s=time.perf_counter()-started,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,gradients=gradients,encoder_frozen=False),step


def train(config,resume=False,stop_after_epoch=None):
    root=workspace_path(config['data']);output=workspace_path(config['output']);output.mkdir(parents=True,exist_ok=True)
    reference=workspace_path(config['frozen_reference_run']);source=workspace_path(config['source_run'])/'best.pt'
    write_json(output/'config.json',config)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    initial_state=torch.load(source,map_location='cpu',weights_only=False);assert initial_state['epoch']==90
    encoder=ptv3.PointTransformerV3.from_preset(config['encoder']['name']).to(config['device']);encoder.precision='fp32'
    encoder.load_state_dict(initial_state['encoder']);encoder.requires_grad_(True)
    head=SlicedCostHead(config).to(config['device']);head.load_state_dict(initial_state['model'])
    optimizer=torch.optim.AdamW([dict(params=encoder.parameters(),lr=config['encoder_minimum_learning_rate'],name='encoder'),
        dict(params=head.parameters(),lr=config['minimum_learning_rate'],name='heads')],weight_decay=config['weight_decay'])
    if resume:
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        encoder.load_state_dict(state['encoder']);head.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        restore_random(state['random_state']);start=state['epoch']+1;step=state['step'];best=state['best_score']
        history=read_json(output/'history.json')[:state['epoch']];initial=read_json(output/'initial_val.json')
        check=read_json(output/'resume_verification.json')
        check['continuations'].append(dict(from_epoch=state['epoch'],from_step=step,
            optimizer_steps=sorted({int(v['step']) for v in optimizer.state.values()}),scheduler=state['scheduler'],
            learned_encoder_restored=all(torch.equal(v.cpu(),state['encoder'][k]) for k,v in encoder.state_dict().items()),random_state_restored=True))
        write_json(output/'resume_verification.json',check)
    else:
        fixed=deepcopy(config);fixed['fixed_data']=str(prepare_fixed_maps(config))
        if not (root/'manifest.json').exists():write_json(root/'manifest.json',fixed_manifest(fixed))
        initial=evaluate(encoder,head,config,'val',output/'initial_validation_predictions',output/'baseline_responses.json')
        frozen_initial=read_json(reference/'initial_val.json')
        differences=[]
        for p in (output/'initial_validation_predictions').glob('*.npy'):
            differences.append(float(abs(np.load(p)-np.load(reference/'initial_validation_predictions'/p.name)).max()))
        initial_check=dict(score_difference=abs(initial['score']-frozen_initial['score']),
            raw_mae_difference=abs(initial['cost_mae_raw']-frozen_initial['cost_mae_raw']),
            maximum_prediction_difference=max(differences),encoder_weights_equal=all(torch.equal(v.cpu(),initial_state['encoder'][k]) for k,v in encoder.state_dict().items()))
        assert initial_check['score_difference']<1e-5 and initial_check['maximum_prediction_difference']<1e-5,initial_check
        write_json(output/'initialization_verification.json',initial_check)
        write_json(output/'initial_val.json',initial);write_json(output/'history.json',[])
        write_json(output/'resume_verification.json',dict(source_epoch=90,fresh_optimizer=True,continuations=[]))
        history=[];start=1;step=0;best=initial['score']
        print('UNFROZEN INITIAL VALIDATION',initial['cost_mae_raw'],initial['score'],initial_check,flush=True)
    def payload(epoch,pool):
        return dict(architecture=config['architecture'],config=config,encoder=cpu_state(encoder),model=cpu_state(head),
            optimizer=optimizer.state_dict(),scheduler=dict(kind='epoch_fraction_warmup_cosine',phase_start=0,
                phase_end=config['epochs'],epoch_progress=epoch,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups}),
            epoch=epoch,step=step,best_score=best,random_state=random_state(),generation=pool,
            conditions=config['condition_names'],condition_scale=config['condition_scale'],source_checkpoint=str(source),
            source_epoch=90,encoder_trainable=True,frozen_features_used=False,
            geometry_contract='Full 3D contexts and native support queries, map metres, trainable Tiny')
    if not resume:
        save_checkpoint(output/'best.pt',payload(0,None));write_json(output/'selection.json',dict(epoch=0,score=best,source='initialization',validation=initial))
    target=config['stop_epoch']
    for epoch in range(start,target+1):
        previous=history[-1]['validation'] if history else initial
        pool=prepare_pool(config,(epoch-1)//10,epoch-1,previous)
        row,step=train_epoch(encoder,head,optimizer,config,epoch,step,pool,previous)
        validation=evaluate(encoder,head,config,'val');row['validation']=validation;history.append(row)
        if validation['score']<best:
            best=validation['score'];save_checkpoint(output/'best.pt',payload(epoch,pool))
            write_json(output/'selection.json',dict(epoch=epoch,score=best,source='online_finetuning',validation=validation))
        save_checkpoint(output/'last.pt',payload(epoch,pool));write_json(output/'history.json',history)
        progress(config,'training',completed_epochs=epoch,epoch=epoch,step=step,generation=pool['index'],generations_total=5,
            loss=row['loss'],last_validation=validation,best_score=best,gradients=row['gradients'],
            learning_rates=row['learning_rates'],peak_cuda_gib=row['peak_cuda_gib'],temporary_bytes=pool['temporary_bytes'])
        print(f'VALIDATION epoch={epoch} raw_MAE={validation["cost_mae_raw"]:.5f} score={validation["score"]:.6f}',flush=True)
        gc.collect()
        if stop_after_epoch is not None and epoch==stop_after_epoch:
            print('CONTROLLED CHECKPOINT STOP',epoch,flush=True);return
    selected=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    encoder.load_state_dict(selected['encoder']);head.load_state_dict(selected['model'])
    validation=evaluate(encoder,head,config,'val',output/'validation_predictions',output/'final_responses.json')
    test=evaluate(encoder,head,config,'test',output/'predictions')
    frozen=read_json(reference/'summary.json')
    write_json(output/'summary.json',dict(selected_epoch=selected['epoch'],completed_epochs=target,initial_validation=initial,
        best_validation=validation,test=test,frozen_validation=frozen['best_validation'],frozen_test=frozen['test'],
        new_scenes=20,generations=5,scheduler_budget_epochs=100,best_checkpoint=str(output/'best.pt'),last_checkpoint=str(output/'last.pt')))
    last=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    changed=[k for k,v in last['encoder'].items() if not torch.equal(v,initial_state['encoder'][k])]
    check=dict(completed_epoch=last['epoch'],selected_epoch=selected['epoch'],last_step=last['step'],
        expected_epochs_complete=[r['epoch'] for r in history]==list(range(1,51)),expected_step=sum(r['updates'] for r in history),
        same_update_count_as_frozen=last['step']==read_json(reference/'training_verification.json')['last_step'],
        encoder_updated_keys=len(changed),encoder_total_state_keys=len(last['encoder']),encoder_is_trained=bool(changed),
        best_encoder_is_trained=any(not torch.equal(v,initial_state['encoder'][k]) for k,v in selected['encoder'].items()),
        maximum_encoder_change=max(float(abs(last['encoder'][k]-initial_state['encoder'][k]).max()) for k in changed),
        optimizer_steps=sorted({int(v['step']) for v in last['optimizer']['state'].values()}),
        final_learning_rates={g['name']:g['lr'] for g in last['optimizer']['param_groups']},
        scenes_archived=len(list((output/'scene_archive').glob('scene_*'))),rotations_verified=len(list((output/'rotations').glob('*.json'))))
    assert check['expected_epochs_complete'] and check['same_update_count_as_frozen'] and check['encoder_is_trained']
    write_json(output/'training_verification.json',check)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args();config=read_json(args.config);torch.set_num_threads(config['cpu_threads'])
    train(config,args.resume,args.stop_after_epoch)


if __name__=='__main__':main()
