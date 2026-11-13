"""Staged geometry AE / conditional Flow Matching with resumable best/last."""
import argparse
import hashlib
import math
import random
import signal
import gc
import subprocess
import sys
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.pointcloud import read_json,write_json,workspace_path
from Reconstruction.data.dataset import Windows,on_device,geometry_batch,identity
from Reconstruction.models.model import Autoencoder,FlowModel
from Reconstruction.training.loss import geometry_loss,flow_matching_loss
from Reconstruction.data.curriculum import CurriculumSampler,band_for_epoch
from Reconstruction.training.evaluate import evaluate
from Reconstruction.visualization.plots import curves


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def prepare_data_stage(config,stage_id,log):
    """A child process owns preparation RNG and the temporary dataset version."""
    with log.open('a') as stream:
        subprocess.run([sys.executable,'-m','Reconstruction.data.stages','prepare-stage',
                        '--config',str(workspace_path(config['dataset_config'])),
                        '--stage-id',str(stage_id),'--device',config['device']],
                       cwd=workspace_path('.'),stdout=stream,stderr=subprocess.STDOUT,check=True)


def close_loader(loader):
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:iterator._shutdown_workers()


def make_loader(ds,stage,config,curriculum):
    sampler=CurriculumSampler(ds.rows,curriculum,config['seed']) if stage=='flow' else None
    loader=DataLoader(ds,batch_size=None,shuffle=sampler is None,sampler=sampler,
                      num_workers=config['workers'],collate_fn=identity,persistent_workers=config['workers']>0,
                      **({'prefetch_factor':2} if config['workers'] else {}))
    return loader,sampler


def train(config,stage,epochs=None,resume=None,ae_checkpoint=None):
    config=dict(config)
    resumed=torch.load(workspace_path(resume),map_location='cpu',weights_only=False) if resume else None
    device=torch.device(config['device']);torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    root=workspace_path(config['output'])/stage;root.mkdir(parents=True,exist_ok=True)
    loss_config=resumed['config']['loss_settings'] if resumed else read_json(workspace_path(config['loss_config']))
    curriculum=resumed['config']['curriculum_settings'] if resumed else read_json(workspace_path(config['curriculum_config']))
    config.update(loss_settings=loss_config,curriculum_settings=curriculum)
    if stage=='flow':
        next_epoch=(resumed['epoch'] if resumed else 0)+1
        expected=band_for_epoch(curriculum,min(next_epoch,config['epochs']))['stage_id']
        active=read_json(workspace_path(config['data'])/'manifest.json')
        if resumed and resumed.get('data_version')!=active['data_version'] and not resumed.get('pending_stage'):
            raise ValueError('Checkpoint data version differs from the published training data.')
        prepare_data_stage(config,expected,root/'preparation.log')
    ds=Windows(workspace_path(config['data']),'train',stage);val=Windows(workspace_path(config['data']),'val',stage)
    loader,curriculum_sampler=make_loader(ds,stage,config,curriculum)
    ae=Autoencoder().to(device);flow=None;ae_validation=None
    if stage=='flow':
        prior=resumed if resumed is not None else torch.load(workspace_path(ae_checkpoint),map_location='cpu',weights_only=False)
        ae_validation=prior['ae_validation'] if prior['stage']=='flow' else next((r['validation'] for r in reversed(prior['history']) if 'validation' in r),{})
        if (epochs or config['epochs'])>5 and not all(ae_validation.get(k,0)>=v for k,v in config['ae_gate'].items()):
            raise RuntimeError(f'Stage A validation gate not met: {ae_validation}. Stage B main training is not started.')
        ae.load_state_dict(prior['ae']);ae.eval().requires_grad_(False)
        flow=FlowModel().to(device);flow.condition.load_state_dict(ae.encoder.state_dict())
        # Target latents and normalization come exclusively from training geometry.
        zsum=torch.zeros(32,device=device);zsq=torch.zeros_like(zsum);count=0
        encoder_digest=hashlib.sha256()
        for value in ae.encoder.state_dict().values():encoder_digest.update(value.detach().cpu().numpy().tobytes())
        cache=root/'latents'/encoder_digest.hexdigest()[:16];cache.mkdir(parents=True,exist_ok=True)
        latent_ds=Windows(workspace_path(config['data']),'train','ae')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']=='bf16'):
            for raw in latent_ds:
                file=cache/f"{raw['row']['frontend_id']}_{raw['row']['grid_id']}.npz"
                if file.exists():
                    with np.load(file) as data:z=torch.as_tensor(data['latent'],device=device)
                else:
                    z=ae.encode(on_device(raw,device)).float()
                    np.savez(file,latent=z.cpu().numpy(),origin=raw['origin'],shape=raw['shape'])
                zsum+=z.sum((0,2,3,4));zsq+=(z*z).sum((0,2,3,4));count+=math.prod(z.shape[2:])
        mean=zsum/count;std=(zsq/count-mean**2).clamp_min(1.e-8).sqrt()
        flow.latent_mean.copy_(mean.view(1,32,1,1,1));flow.latent_std.copy_(std.view(1,32,1,1,1))
    model=ae if stage=='ae' else flow
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    budget=config['epochs']*len(loader);warmup=max(1,int(.05*budget))
    def schedule(step):
        if step<warmup:return (step+1)/warmup
        return .01+.99*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,budget-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    history=[];best=-1.;best_loss=float('inf');step=0;start_epoch=0;start_index=0;running_losses=[]
    rng=np.random.default_rng(config['seed'])
    if resume:
        ckpt=resumed
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
        history=ckpt['history'];best=ckpt['best'];best_loss=ckpt.get('best_loss',float('inf'));step=ckpt['step'];start_epoch=ckpt['epoch'];start_index=ckpt['next_index']
        rng.bit_generator.state=ckpt['numpy_generator'];torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        random.setstate(ckpt['python_rng'])
        running_losses=ckpt.get('running_losses',[])
    write_json(root/'config.json',dict(config,stage=stage,ae_checkpoint=ae_checkpoint))
    stop_epoch=epochs or config['epochs'];train_wall=0.
    pending_stage=None
    def checkpoint(epoch,next_index):
        return dict(version=config['version'],stage=stage,config=config,model=model.state_dict(),ae=ae.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,next_index=next_index,
                    step=step,best=best,best_loss=best_loss,history=history,ae_validation=ae_validation,running_losses=running_losses if next_index else [],numpy_generator=rng.bit_generator.state,
                    curriculum_stage=ds.stage_id,data_version=ds.data_version,pending_stage=pending_stage,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate())
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    old_signals={sig:signal.signal(sig,request_stop) for sig in [signal.SIGINT,signal.SIGTERM]}
    progress_epoch,progress_index=start_epoch,start_index
    try:
        for epoch in range(start_epoch,stop_epoch):
            model.train();losses=running_losses if epoch==start_epoch else [];parts={};tick=perf_counter()
            progress_epoch=epoch
            # Deterministic epoch ordering also makes partial-epoch restoration exact.
            loader.generator=torch.Generator().manual_seed(config['seed']+epoch)
            # Worker startup consumes its generator once. Keep permutation RNG
            # independent so a fresh resumed loader has the same epoch order.
            if curriculum_sampler is None:loader.sampler.generator=torch.Generator().manual_seed(config['seed']+epoch)
            else:curriculum_sampler.set_epoch(epoch+1)
            for index,raw in enumerate(loader):
                if epoch==start_epoch and index<start_index:continue
                sample=on_device(raw,device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']=='bf16'):
                    if stage=='ae':
                        field=ae(sample)
                        queries=geometry_batch(sample['folder'],config['levels_m'],config['queries_per_level'],rng,device,(sample['row']['core_lower'],sample['row']['core_upper']))
                        loss,parts=geometry_loss(field,queries,'ae',loss_config)
                    else:
                        with np.load(cache/f"{sample['row']['frontend_id']}_{sample['row']['grid_id']}.npz") as cached:
                            target=torch.as_tensor(cached['latent'],device=device)
                        queries=geometry_batch(sample['folder'],config['levels_m'],config['queries_per_level'],rng,device,
                            (sample['row']['core_lower'],sample['row']['core_upper']),'flow',raw['global_index'])
                        loss,parts=flow_matching_loss(flow,ae,sample,target,queries,epoch+index/len(loader),loss_config)
                loss.backward();optimizer.step();scheduler.step();step+=1;losses.append(float(loss.detach()))
                running_losses=losses;progress_index=index+1
                if step%config['save_every_steps']==0:atomic_save(checkpoint(epoch,index+1),root/'last.pt')
                if config.get('max_steps') and step>=config['max_steps']:
                    atomic_save(checkpoint(epoch,index+1),root/'last.pt')
                    return root/'last.pt'
                if stopping:raise KeyboardInterrupt
            torch.cuda.synchronize();elapsed=perf_counter()-tick;train_wall+=elapsed
            row=dict(epoch=epoch+1,step=step,loss=float(np.mean(losses)),train_s=elapsed,**parts)
            if (epoch+1)%config['evaluate_every']==0 or epoch+1==stop_epoch:
                model.eval()
                report=evaluate(ae,flow,val,config,root/'validation_latest',config['sample_steps'],config['seed'])
                row['validation']={k:v for k,v in report.items() if k not in ['rows','by_kind','by_difficulty','by_mask_id']}
                score=min(report['precision_3cm'],report['complete_recall_3cm']) if stage=='ae' else report['f1_2cm']
                improved=(score,-report['geometry_loss'])>(best,-best_loss)
                if improved:best,best_loss=score,report['geometry_loss']
            else:improved=False
            history.append(row);write_json(root/'history.json',history)
            progress_epoch,progress_index=epoch+1,0
            pending_stage=(band_for_epoch(curriculum,epoch+2)['stage_id']
                           if stage=='flow' and epoch+1<config['epochs'] and
                           band_for_epoch(curriculum,epoch+2)['stage_id']!=ds.stage_id else None)
            atomic_save(checkpoint(epoch+1,0),root/'last.pt')
            if improved:atomic_save(checkpoint(epoch+1,0),root/'best.pt')
            curves(history,root/'curves.png')
            print(f"{stage} epoch {epoch+1}/{stop_epoch} step {step} loss {row['loss']:.6f} train {elapsed:.2f}s best_score {best:.4f}",flush=True)
            if stopping:raise KeyboardInterrupt
            if pending_stage is not None and epoch+1<stop_epoch:
                close_loader(loader)
                del raw,sample,queries,loss
                gc.collect();torch.cuda.empty_cache()
                prepare_data_stage(config,pending_stage,root/'preparation.log')
                if stopping:raise KeyboardInterrupt
                ds=Windows(workspace_path(config['data']),'train',stage)
                loader,curriculum_sampler=make_loader(ds,stage,config,curriculum)
                pending_stage=None
                atomic_save(checkpoint(epoch+1,0),root/'last.pt')
    except KeyboardInterrupt:
        atomic_save(checkpoint(progress_epoch,progress_index),root/'last.pt');raise
    finally:
        close_loader(loader)
        for sig,handler in old_signals.items():signal.signal(sig,handler)
    write_json(root/'run_metrics.json',dict(stage=stage,epochs=stop_epoch,step=step,train_loop_s=train_wall,
        cuda_allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name(),best_validation_score=best))
    return root/'best.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='Reconstruction/configs/training.json')
    p.add_argument('--stage',required=True,choices=['ae','flow']);p.add_argument('--epochs',type=int);p.add_argument('--resume');p.add_argument('--ae-checkpoint')
    a=p.parse_args();train(read_json(workspace_path(a.config)),a.stage,a.epochs,a.resume,a.ae_checkpoint)

if __name__=='__main__':main()
