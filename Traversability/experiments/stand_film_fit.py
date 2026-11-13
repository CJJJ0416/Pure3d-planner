"""Fixed R4 features and teachers: update existing FiLM only, for 1,000 steps.

The four raw bottleneck points are observation-only. Held-out conditions share
the same geometry, so this is a learnability experiment, not unseen-map testing.
"""
import argparse
import copy
import hashlib
from itertools import combinations
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from Traversability.models.model import ConditionedCost
from Traversability.training.train import autocast, regression_loss
from Traversability.training.condition_response import balanced_difference_loss
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda:stream.read(8*1024*1024),b''):h.update(piece)
    return h.hexdigest()


def loss_parts(prediction,target,config,pairs):
    absolute,_=regression_loss(prediction,target,config['smooth_l1_beta'],0.)
    difference=balanced_difference_loss(prediction,target,[pairs],config['smooth_l1_beta'],
        config['change_threshold'],[1.])
    total=absolute+config['stand_difference_weight']*difference
    return dict(total=total,absolute=absolute,stand=difference)


def scalar_parts(parts):return {name:float(value.detach()) for name,value in parts.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'Traversability/configs/stand_ratio_film_r4_fit.json')
    args=parser.parse_args();config=read_json(args.config)
    output=workspace_path(config['output']);output.mkdir(parents=True,exist_ok=True)
    cache=workspace_path(config['cache']);source_path=workspace_path(config['source_checkpoint'])
    teacher=np.load(cache/'teacher.npz');cached=np.load(cache/'student.npz')
    cache_protocol=read_json(cache/'protocol.json');source_sha=sha(source_path)
    assert source_sha==cache_protocol['checkpoint_sha256']
    assert np.array_equal(cached['query_xyz'],teacher['query_xyz'])
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    np.random.seed(config['seed']);device=torch.device(config['device'])
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    model=ConditionedCost(source['config']['model'],source['condition_scale'],pretrained=False).to(device)
    model.load_state_dict(source['model']);model.requires_grad_(False);model.eval()
    model.condition_head.requires_grad_(True)
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert all(name.startswith('condition_head.') for name in trainable)
    # Keep the cached fusion output's float32 dtype. AMP still uses BF16 in the
    # linear layers, exactly as the original mixed-precision inference did.
    features=torch.from_numpy(cached['query_features']).to(device=device).detach()
    conditions=torch.from_numpy(teacher['conditions'].astype(np.float32))[None].to(device)
    target=torch.from_numpy(teacher['target']).to(device);n=len(target)
    query_batch=torch.zeros(len(features),device=device,dtype=torch.long)
    ratios=teacher['ratios'];baseline=int(np.flatnonzero(ratios==config['baseline_ratio'])[0])
    train_cols=[int(np.flatnonzero(ratios==r)[0]) for r in config['train_ratios']]
    val_cols=[int(np.flatnonzero(ratios==r)[0]) for r in config['validation_ratios']]
    train_pairs=list(combinations(range(len(train_cols)),2))
    val_response_cols=[baseline,*val_cols];val_pairs=[(0,k+1) for k in range(len(val_cols))]
    optimizer=torch.optim.AdamW(model.condition_head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['steps'])
    history=[];evaluations=[];best=float('inf');best_step=0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
    model_inference_config=copy.deepcopy(source['config'])
    model_inference_config['output']=str(output)
    model_inference_config['diagnostic_training']=config
    protocol=dict(experiment='R4 fixed-feature FiLM controlled fit',configuration=config,
        source_checkpoint=str(source_path),source_checkpoint_sha256=source_sha,source_epoch=source['epoch'],
        cache_feature_sha256=sha(cache/'student.npz'),teacher_sha256=sha(cache/'teacher.npz'),
        conditions=teacher['conditions'].tolist(),train_columns=train_cols,validation_columns=val_cols,
        train_pairs=train_pairs,validation_pairs_in_baseline_and_heldout_columns=val_pairs,
        train_points=n,watch_points=len(teacher['watch_xyz']),features_shape=list(features.shape),
        training_targets_per_update=n*len(train_cols),trainable_names=trainable,trainable_parameters=trainable_count,
        precision=config['amp'],device=torch.cuda.get_device_name(device),
        frozen_geometry_and_CostHead=True,gradient_through_frozen_CostHead=True,
        validation='Same XYZ and geometry, held-out stand conditions only. No unseen-building claim.',
        checkpoint_selection='Heldout absolute SmoothL1 + heldout-vs-0.05 balanced stand loss, include step zero',
        shared_FiLM='All four inputs share the head. Other input values fixed, other contexts not evaluated.',
        optimizer_fresh=True,astar_run=False,gpmp_run=False)
    write_json(output/'protocol.json',protocol);write_json(output/'config.json',config)

    def checkpoint(step):
        return dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            config=model_inference_config,condition_names=source['condition_names'],condition_scale=source['condition_scale'],
            blocks_config=source['blocks_config'],coordinate_frame=source['coordinate_frame'],
            inference_query_source=source['inference_query_source'],block_size_m=source['block_size_m'],
            context_m=source['context_m'],cost_scale=source['cost_scale'],
            profile_ids=[f'stand_{r:.2f}' for r in ratios],source_epoch=source['epoch'],epoch=source['epoch'],
            experiment_type='stand_film_fixed_features',diagnostic_step=step,step=step,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            diagnostic_config=config,best_score=best,best_step=best_step,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())

    def save_prediction(filename,prediction,step):
        values=prediction.float().cpu().numpy()
        np.savez(output/filename,coord=teacher['query_xyz'],prediction=values[:n],target=teacher['target'],
            conditions=teacher['conditions'],ratios=ratios,watch_xyz=teacher['watch_xyz'],
            watch_prediction=values[n:],watch_target=teacher['watch_target'],
            part=teacher['part'],step_index=teacher['step'],across=teacher['across'],display=teacher['display'],
            diagnostic_step=step)

    @torch.no_grad()
    def evaluate(step):
        nonlocal best,best_step
        model.eval()
        with autocast(device,config['amp']):prediction=model.decode_cost(features,conditions,query_batch)
        train_loss=loss_parts(prediction[:n,train_cols],target[:,train_cols],config,train_pairs)
        absolute,_=regression_loss(prediction[:n,val_cols],target[:,val_cols],config['smooth_l1_beta'],0.)
        delta=balanced_difference_loss(prediction[:n,val_response_cols],target[:,val_response_cols],
            [val_pairs],config['smooth_l1_beta'],config['change_threshold'],[1.])
        validation=dict(total=float(absolute+config['stand_difference_weight']*delta),absolute=float(absolute),stand=float(delta))
        row=dict(step=step,train=scalar_parts(train_loss),validation=validation,per_condition=[],
                 elapsed_s=time.perf_counter()-started)
        for i,r in enumerate(ratios):
            error=prediction[:n,i]-target[:,i]
            absolute,_=regression_loss(prediction[:n,i:i+1],target[:,i:i+1],config['smooth_l1_beta'],0.)
            row['per_condition'].append(dict(stand_ratio=float(r),split='validation' if i in val_cols else 'train',
                MAE=float(error.abs().mean()),SmoothL1=float(absolute)))
        if step==0:
            baseline_difference=(prediction[n:,baseline].cpu()-torch.from_numpy(cached['watch_prediction'][:,baseline])).abs()
            all_difference=(prediction[:n].cpu()-torch.from_numpy(cached['prediction'])).abs()
            row['baseline_watch_max_abs_difference']=float(baseline_difference.max())
            row['baseline_all_query_max_abs_difference']=float(all_difference.max())
            save_prediction('initial_predictions.npz',prediction,step)
            assert row['baseline_watch_max_abs_difference']==0.
        if validation['total']<best:
            best,best_step=validation['total'],step
            torch.save(checkpoint(step),output/'best.pt')
            save_prediction('best_predictions.npz',prediction,step)
        evaluations.append(row)
        write_json(output/'evaluations.json',evaluations)
        torch.save(checkpoint(step),output/'last.pt')
        save_prediction('last_predictions.npz',prediction,step)
        write_json(output/'progress.json',dict(stage='training' if step<config['steps'] else 'training_complete',
            step=step,steps=config['steps'],percentage=100*step/config['steps'],
            train=row['train'],validation=validation,best_step=best_step,best_score=best,
            trainable_parameters=trainable_count,elapsed_s=time.perf_counter()-started))
        print(f'EVAL step={step} train={row["train"]} validation={validation} best_step={best_step}',flush=True)
        return row

    initial=evaluate(0)
    print(f'START trainable={trainable_count} steps={config["steps"]} full_batch={n}x{len(train_cols)}',flush=True)
    for step in range(1,config['steps']+1):
        model.condition_head.train();optimizer.zero_grad(set_to_none=True)
        with autocast(device,config['amp']):
            prediction=model.decode_cost(features[:n],conditions[:,train_cols],query_batch[:n])
            parts=loss_parts(prediction,target[:,train_cols],config,train_pairs)
        parts['total'].backward()
        grad_norm=torch.sqrt(sum(p.grad.float().square().sum() for p in model.condition_head.parameters()))
        used_lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        if step%config['log_every_steps']==0:
            row=dict(step=step,**scalar_parts(parts),learning_rate=used_lr,FiLM_gradient_norm=float(grad_norm),
                elapsed_s=time.perf_counter()-started)
            history.append(row);write_json(output/'history.json',history)
            print(f'TRAIN step={step} total={row["total"]:.7f} absolute={row["absolute"]:.7f} stand={row["stand"]:.7f}',flush=True)
        if step%config['evaluate_every_steps']==0 or step==config['steps']:evaluate(step)
    torch.cuda.synchronize(device)
    # Verify actual trained tensors, not just requires_grad flags.
    verification={}
    for label,path in [('best',output/'best.pt'),('last',output/'last.pt')]:
        state=torch.load(path,map_location='cpu',weights_only=False)
        changed=[key for key,value in state['model'].items() if not torch.equal(value,source['model'][key])]
        frozen_changed=[key for key in changed if not key.startswith('condition_head.')]
        assert not frozen_changed
        verification[label]=dict(step=state['diagnostic_step'],changed_tensors=changed,
            frozen_tensors_changed=frozen_changed,all_non_FiLM_tensors_identical=True)
        model.load_state_dict(state['model']);model.eval()
        with torch.no_grad(),autocast(device,config['amp']):
            reloaded=model.decode_cost(features,conditions,query_batch)
        saved=np.load(output/f'{label}_predictions.npz')
        expected=np.concatenate([saved['prediction'],saved['watch_prediction']])
        reload_difference=float(abs(reloaded.cpu().numpy()-expected).max())
        assert reload_difference==0.
        verification[label]['reloaded_full_checkpoint_prediction_max_abs_difference']=reload_difference
    assert sha(source_path)==source_sha
    verification.update(original_best_sha256_unchanged=True,trainable_parameters=trainable_count,
        baseline_watch_prediction_exact=True,watch_points_used_for_training=False,heldout_conditions_used_for_training=False,
        optimizer_updates=config['steps'],astar_run=False,gpmp_run=False,
        full_model_prediction_metadata_preserved=True)
    write_json(output/'verification.json',verification)
    write_json(output/'training_complete.json',dict(best_step=best_step,steps=config['steps'],
        initial=initial,last=evaluations[-1],best=next(row for row in evaluations if row['step']==best_step),
        wall_s=time.perf_counter()-started,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30))
    print(f'TRAINING_COMPLETE best_step={best_step} wall_s={time.perf_counter()-started:.3f}',flush=True)
    subprocess.run([config['geometry_python'],'-m','Traversability.experiments.stand_film_report','--run',str(output)],cwd=ROOT,check=True)


if __name__=='__main__':main()
