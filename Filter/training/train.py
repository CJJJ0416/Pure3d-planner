"""Frozen-feature score-head training with scene-balanced batches and resumable updates."""
import argparse
from pathlib import Path
import signal
from time import perf_counter
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from Filter.data.dataset import CachedCores,BalancedBatches,SceneBalancedBatches,collate_cores
from Filter.models.filter import FilterHead,geometry_mode
from Filter.training.loss import regression_loss
from Filter.training.metrics import Metrics
from utils.pointcloud import read_json,write_json,workspace_path


def loader(dataset,config,sampler=None):
    args=dict(num_workers=config['num_workers'],pin_memory=True,persistent_workers=True,
              prefetch_factor=2,collate_fn=collate_cores,generator=torch.Generator().manual_seed(config['seed']))
    return DataLoader(dataset,batch_sampler=sampler,**args) if sampler is not None else DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,**args)


def move(batch,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def save_checkpoint(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


@torch.inference_mode()
def evaluate(model,dataset,config,device,thresholds=(.5,),predictions=None):
    model.eval();metric=Metrics(thresholds,device);writers={};entries={};started=perf_counter()
    # Disjoint cores cover each variant once. Export needs only the cache,
    # including when the original geometry files have been archived separately.
    sample_sizes={}
    if predictions is not None:
        for row in dataset.records:
            key=(row['scene'],row['suite'],row['variant'])
            sample_sizes[key]=sample_sizes.get(key,0)+row['core_points']
    for batch in loader(dataset,config):
        batch=move(batch,device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']=='bf16'):
            prediction=model(batch)
        metric.update(prediction,batch['target'],batch['group'],batch['semantic'])
        if predictions is not None:
            values=prediction.cpu().numpy();offset=0
            for index,count in zip(batch['sample_indices'],batch['counts']):
                row=dataset.records[index];key=(row['scene'],row['suite'],row['variant'])
                if key not in writers:
                    folder=Path(predictions)/row['scene']/row['suite'];folder.mkdir(parents=True,exist_ok=True)
                    n=sample_sizes[key]
                    path=folder/f"prediction_{row['variant']:03d}.npy"
                    writers[key]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,));writers[key][:]=np.nan
                    entries[key]=dict(scene=row['scene'],suite=row['suite'],variant=row['variant'],
                        prediction=str(path.resolve()),source_folder=row['folder'])
                ids=np.load(dataset.feature_root/row['cache_key']/'point_index.npy',mmap_mode='r')
                writers[key][ids]=values[offset:offset+count];offset+=count
    for value in writers.values(): value.flush()
    if predictions is not None: write_json(Path(predictions)/'index.json',dict(samples=list(entries.values())))
    report=metric.report();report.update(wall_s=perf_counter()-started)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('Filter/configs/train_scoring_no_knn.json'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after-epoch',type=int,help='Save after this epoch, retaining the configured 50-epoch scheduler.')
    args=parser.parse_args()
    old=torch.load(workspace_path(args.resume),map_location='cpu',weights_only=False) if args.resume else None
    c=old['config'] if old else read_json(workspace_path(args.config))
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    device=torch.device(c['device']);output=workspace_path(c['output']);output.mkdir(parents=True,exist_ok=True)
    mode=geometry_mode(c);use_knn=mode=='knn'
    training=CachedCores(workspace_path(c['cache']),'train',geometry_mode=mode)
    validation=CachedCores(workspace_path(c['cache']),'val',geometry_mode=mode)
    meta=training.metadata
    mean,std=(meta['geometry_mean'],meta['geometry_std']) if mode!='relative' else (None,None)
    model=FilterHead(mean,std,c['dropout'],geometry_mode=mode).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    sampler=(SceneBalancedBatches(training,c['seed'],c['batch_size']) if c.get('sampler')=='scene_balanced'
             else BalancedBatches(training,c['seed']))
    total=sampler.steps*c['epochs'];warm=max(1,int(total*c['warmup_fraction']))
    def rate(step):
        if step<warm: return (step+1)/warm
        p=(step-warm)/(total-warm)
        return c['minimum_learning_rate']/c['learning_rate']+(1-c['minimum_learning_rate']/c['learning_rate'])*(.5+.5*math.cos(math.pi*p))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    epoch,next_batch,step=0,0,0;best_key=[float('inf'),float('inf')];history=[];running=torch.zeros(5,device=device,dtype=torch.float64)
    encoder_state=torch.load(training.feature_root/'encoder.pth',map_location='cpu',weights_only=True)
    if old:
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
        epoch,next_batch,step=old['epoch'],old['next_batch'],old['step'];best_key=old['best_key'];history=old['history']
        running.copy_(old['running'].to(device));torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng']);np.random.set_state(old['numpy_rng'])
    stop=[False];signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    write_json(output/'config.json',c)
    if not old and c.get('train_metrics_each_epoch',False):
        initial=dict(training=evaluate(model,training,c,device),validation=evaluate(model,validation,c,device))
        write_json(output/'initial_metrics.json',initial)
        print(f"initial train_macro_mae={initial['training']['macro_mae']:.6f} val_macro_mae={initial['validation']['macro_mae']:.6f}",flush=True)
    def payload():
        return dict(config=c,use_knn=use_knn,geometry_mode=mode,model=model.state_dict(),encoder=encoder_state,encoder_info=meta['encoder_info'],
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,next_batch=next_batch,step=step,
            best_key=best_key,history=history,running=running.cpu(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            geometry_mean=mean,geometry_std=std)
    data_loader=loader(training,c,sampler)
    while epoch<c['epochs']:
        sampler.epoch,sampler.start_batch=epoch,next_batch
        model.train();started=perf_counter();torch.cuda.reset_peak_memory_stats()
        for batch_number,batch in enumerate(data_loader,start=next_batch):
            batch=move(batch,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['amp']=='bf16'):
                prediction=model(batch)
            loss=regression_loss(prediction,batch['target'],batch['group'],c['smooth_l1_beta'],
                batch['semantic'] if c.get('structure_balanced_loss',False) else None)
            loss.backward();optimizer.step();scheduler.step();step+=1;next_batch=batch_number+1
            running+=torch.stack([loss.detach().double(),torch.tensor(1.,device=device),
                torch.tensor(float(len(batch['sample_indices'])),device=device),
                torch.tensor(float(len(prediction)),device=device),torch.tensor(0.,device=device)])
            if step%c['log_every_steps']==0:
                print(f'epoch={epoch+1} step={step} loss={loss.item():.6f}',flush=True)
            if step%c['save_every_steps']==0 or stop[0]: save_checkpoint(output/'last.pt',payload())
            if stop[0]:
                print('Stopped after a complete update. last.pt saved.',flush=True);return
        torch.cuda.synchronize();wall=perf_counter()-started
        totals=running.cpu().tolist();metric=None;improved=False
        if epoch<5 or (epoch+1)%c.get('validation_every_epochs',5)==0 or epoch+1==c['epochs']:
            metric=evaluate(model,validation,c,device)
            key=[metric['macro_mae'],metric['mae']]
            if tuple(key)<tuple(best_key): best_key=key;improved=True
        peak=torch.cuda.max_memory_allocated()/2**30
        train_metric=evaluate(model,training,c,device) if c.get('train_metrics_each_epoch',False) else None
        row=dict(epoch=epoch+1,step=step,loss=totals[0]/totals[1],train_wall_s=wall,
            boxes_per_s=totals[2]/wall,points_per_s=totals[3]/wall,
            cuda_peak_GiB=peak,learning_rate=optimizer.param_groups[0]['lr'],validation=metric,training=train_metric)
        history.append(row);epoch+=1;next_batch=0;running.zero_()
        save_checkpoint(output/'last.pt',payload())
        if improved: save_checkpoint(output/'best.pt',payload())
        write_json(output/'history.json',history)
        with (output/'train.log').open('a') as log: log.write(str(row)+'\n')
        write_json(output/'progress.json',dict(epoch=epoch,total_epochs=c['epochs'],step=step,best_macro_mae=best_key[0],state='training'))
        print(f"epoch={epoch} loss={row['loss']:.6f} val_macro_mae={metric['macro_mae'] if metric else '-'} train_s={wall:.2f}",flush=True)
        if args.stop_after_epoch and epoch>=args.stop_after_epoch: return
    write_json(output/'progress.json',dict(epoch=epoch,total_epochs=c['epochs'],step=step,best_macro_mae=best_key[0],state='complete'))

if __name__=='__main__': main()
