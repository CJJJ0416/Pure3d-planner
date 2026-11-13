"""A/B controlled fine-tuning with one continuous 30-epoch cosine schedule."""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch

from Traversability.data.dataset import PointBlocks,EpochSampler,to_device
from Traversability.models.model import ConditionedCost
from Traversability.training.train import autocast,make_loader,save_checkpoint
from Traversability.training.decoupling import decoupling_loss,evaluate_decoupling
from utils.pointcloud import read_json,write_json as _write_json


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    _write_json(temporary,value);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after-epoch',type=int,default=30)
    args=parser.parse_args()
    restored=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    config=restored['config'] if restored else read_json(args.config)
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    np.random.seed(config['seed']);random.seed(config['seed'])
    device=torch.device(config['device'])
    initial=torch.load(config['initialization_checkpoint'],map_location='cpu',weights_only=False)
    dataset=PointBlocks(config['data'],'train',config['model'],config['seed'],augment=True)
    sampler=EpochSampler(dataset,config['seed'],config['blocks_per_epoch'],
                         dict(stair_blocks=336,source='new',balance_by='augmentation_target'))
    loader=make_loader(dataset,config,device,sampler)
    model=ConditionedCost(config['model'],dataset.condition_scale,pretrained=False).to(device)
    model.load_state_dict(initial['model'])
    encoder=list(model.encoder.parameters());encoder_ids={id(p) for p in encoder}
    optimizer=torch.optim.AdamW([dict(params=encoder,lr=config['encoder_learning_rate']),
        dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],lr=config['learning_rate'])],
        weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
    baseline=read_json(config['baseline_metrics']);best=baseline['selection_score']
    history=[];start_epoch=0;next_batch=0;step=0;epoch_sums={};epoch_batches=0;epoch_elapsed=0.
    if restored:
        model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
        history=restored['history'];start_epoch=restored['epoch'];next_batch=restored['next_batch']
        step=restored['step'];best=restored['best_score'];epoch_sums=restored['epoch_sums']
        epoch_batches=restored['epoch_batches'];epoch_elapsed=restored['epoch_elapsed_s']
        torch.set_rng_state(restored['torch_rng']);np.random.set_state(restored['numpy_rng']);random.setstate(restored['python_rng'])
        torch.cuda.set_rng_state_all(restored['cuda_rng'])
    write_json(out/'config.json',config)

    def payload(epoch,batch):
        return dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            condition_names=dataset.condition_names,condition_scale=dataset.condition_scale,profile_ids=dataset.profile_ids,
            blocks_config=dataset.metadata['blocks_config'],coordinate_frame='map',inference_query_source='input_pcd_xyz',cost_scale=50.,
            epoch=epoch,next_batch=batch,step=step,best_score=best,history=history,epoch_sums=epoch_sums,
            epoch_batches=epoch_batches,epoch_elapsed_s=epoch_elapsed,torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all())

    if not restored:save_checkpoint(out/'best.pt',payload(0,0))
    for epoch in range(start_epoch,min(30,args.stop_after_epoch)):
        sampler.epoch=epoch;model.train();frozen=epoch<2
        model.encoder.requires_grad_(not frozen)
        if frozen:model.encoder.eval()
        torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
        for batch_index,batch in enumerate(loader):
            if batch_index<next_batch:continue
            batch=to_device(batch,device);optimizer.zero_grad(set_to_none=True)
            with autocast(device,config['amp']):prediction=model(batch)
            loss,parts=decoupling_loss(prediction,batch['target'],batch['facility_mask'],dataset.condition_pairs,config)
            loss.backward();optimizer.step();step+=1;epoch_batches+=1
            numeric=dict(zip(parts,torch.stack([v.detach() for v in parts.values()]).cpu().tolist()))
            for name,value in numeric.items():epoch_sums[name]=epoch_sums.get(name,0.)+value
            if step%50==0:
                print('TRAIN',epoch+1,step,numeric,flush=True)
            next_batch=batch_index+1
            progress=dict(stage='train',variant=config['variant'],epoch=epoch+1,total_epochs=30,
                step=step,step_in_epoch=next_batch,updates_per_epoch=240,condition_bank=epoch%2,
                losses=numeric)
            write_json(out/'progress.json',progress)
            if step%100==0:
                epoch_elapsed+=time.perf_counter()-began;began=time.perf_counter()
                save_checkpoint(out/'last.pt',payload(epoch,next_batch))
        torch.cuda.synchronize();wall=epoch_elapsed+time.perf_counter()-began
        validation=evaluate_decoupling(model,config,'val')
        row=dict(epoch=epoch+1,step=step,loss=epoch_sums['total']/epoch_batches,
            losses={k:v/epoch_batches for k,v in epoch_sums.items()},validation=validation,
            train_wall_s=wall,encoder_frozen=frozen,learning_rates=scheduler.get_last_lr(),
            true_batch_size=2,blocks=480,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        history.append(row);improved=validation['selection_score']<best
        if improved:best=validation['selection_score']
        scheduler.step();next_batch=0;epoch_sums={};epoch_batches=0;epoch_elapsed=0.
        state=payload(epoch+1,0);save_checkpoint(out/'last.pt',state)
        if improved:save_checkpoint(out/'best.pt',state)
        write_json(out/'history.json',history)
        print('EPOCH',epoch+1,'S',validation['selection_score'],'best',best,'train_s',wall,flush=True)
    write_json(out/'progress.json',dict(stage='pilot_complete' if args.stop_after_epoch==5 else 'train_complete',
        variant=config['variant'],epoch=history[-1]['epoch'],step=step,history=history))


if __name__=='__main__':main()
