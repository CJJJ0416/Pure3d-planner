"""Full-model deterministic R4 fitting with unchanged inference architecture."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from Traversability.data.dataset import collate_queries,to_device
from Traversability.experiments.decoupling_r4 import R4Queries
from Traversability.models.model import ConditionedCost
from Traversability.training.r4_controlled import losses,evaluate_arrays
from Traversability.training.train import autocast,save_checkpoint
from utils.pointcloud import read_json,workspace_path
from Traversability.experiments.r4_controlled_data import write_json

MODULES=['encoder','projection','local_query','global_query','fusion','condition_head','cost_head']


def scalar(values):return {k:float(v.detach()) for k,v in values.items()}


def gradient_report(named,nominal,auxiliary):
    result={}
    for group in MODULES+['all']:
        chosen=[i for i,(name,p) in enumerate(named) if group=='all' or name.split('.')[0]==group]
        dot=aa=bb=0.
        for i in chosen:
            a,b=nominal[i],auxiliary[i]
            if a is not None:aa+=float(a.detach().float().square().sum())
            if b is not None:bb+=float(b.detach().float().square().sum())
            if a is not None and b is not None:dot+=float((a.detach().float()*b.detach().float()).sum())
        result[group]=dict(nominal_norm=aa**.5,auxiliary_norm=bb**.5,
                           cosine=dot/(aa*bb)**.5 if aa and bb else None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--variant',choices=['A','B','C'],required=True)
    p.add_argument('--stop-step',type=int,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();settings=read_json(args.config);run=workspace_path(settings['output']);out=run/args.variant
    out.mkdir(parents=True,exist_ok=True)
    protocol=read_json(run/'protocol.json');teacher=dict(np.load(run/'teacher.npz'))
    torch.set_num_threads(settings['cpu_threads']);torch.manual_seed(settings['seed']);np.random.seed(settings['seed'])
    device=torch.device(settings['device']);source=torch.load(workspace_path(settings['source_checkpoint']),map_location='cpu',weights_only=False)
    config=copy.deepcopy(source['config']);config['model'].update(decoder_precision='fp32',augment_yaw=False)
    config['output']=str(out);config['diagnostic_training']=settings
    query=np.r_[teacher['query_xyz'],teacher['watch_xyz']];n=len(teacher['query_xyz'])
    dataset=R4Queries(config,query,teacher['conditions'])
    batch=to_device(collate_queries([dataset[i] for i in protocol['active_cores']]),device)
    order=torch.argsort(batch['query_index'])
    model=ConditionedCost(config['model'],source['condition_scale'],pretrained=False).to(device)
    model.load_state_dict(source['model']);model.requires_grad_(True);model.eval()
    initial_buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    named=list(model.named_parameters());encoder=list(model.encoder.parameters());ids={id(p) for p in encoder}
    optimizer=torch.optim.AdamW([dict(params=encoder,lr=settings['encoder_learning_rate']),
        dict(params=[p for p in model.parameters() if id(p) not in ids],lr=settings['learning_rate'])],
        weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings['total_steps'])
    target=torch.as_tensor(teacher['target'],device=device);parts=torch.as_tensor(teacher['part'],device=device)
    edges=torch.as_tensor(teacher['edges'],device=device);all_conditions=torch.as_tensor(teacher['conditions'],device=device)
    train_count=protocol['train_condition_count'];train_columns=train_count if args.variant=='C' else 1
    history=[];evaluations=[];step=0;best_score=float('inf');best_step=0;prior_elapsed=0.
    if args.resume:
        restored=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
        step=restored['step'];best_score=restored['best_score'];best_step=restored['best_step']
        history=read_json(out/'history.json');evaluations=read_json(out/'evaluations.json');prior_elapsed=restored['elapsed_s']
        torch.set_rng_state(restored['torch_rng']);torch.cuda.set_rng_state_all(restored['cuda_rng'])
    began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)

    def checkpoint():
        value={k:v for k,v in source.items() if k in ['condition_names','condition_scale','blocks_config',
            'coordinate_frame','inference_query_source','cost_scale','block_size_m','context_m']}
        value.update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,diagnostic_step=step,
            source_epoch=source.get('epoch'),epoch=source.get('epoch'),best_score=best_score,best_step=best_step,
            variant=args.variant,experiment_type='R4 controlled full-model fitting',elapsed_s=prior_elapsed+time.perf_counter()-began,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        return value

    def forward(columns):
        with autocast(device,settings['amp']):
            geometry=model.encode_geometry(batch)
            conditions=all_conditions[:columns][None].expand(len(protocol['active_cores']),-1,-1)
            prediction=model.decode_cost(geometry,conditions,batch['query_batch'])
        return prediction[order]

    def prediction_file(name,prediction):
        np.savez(out/name,query_xyz=teacher['query_xyz'],watch_xyz=teacher['watch_xyz'],prediction=prediction[:n],
            watch_prediction=prediction[n:],target=teacher['target'],watch_target=teacher['watch_target'],
            conditions=teacher['conditions'],part=teacher['part'],step_index=teacher['step'],edges=teacher['edges'],diagnostic_step=step)

    @torch.no_grad()
    def evaluate():
        nonlocal best_score,best_step
        prediction=forward(len(all_conditions));values=prediction.float().cpu().numpy()
        metrics=evaluate_arrays(values,teacher,protocol)
        loss,terms=losses(prediction[:n,:train_columns],target[:,:train_columns],parts,edges,
                          protocol['condition_pairs'],settings,args.variant,protocol['condition_groups'])
        row=dict(step=step,losses=scalar(terms),metrics=metrics,elapsed_s=prior_elapsed+time.perf_counter()-began,
                 peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,learning_rates=scheduler.get_last_lr())
        if step==0:
            reference=np.load(run/'reference_baseline.npy')
            row['reference_baseline_max_abs_difference']=float(abs(values[:,0]-reference).max())
            prediction_file('initial_predictions.npz',values)
        if metrics['score']<best_score:
            best_score=metrics['score'];best_step=step;save_checkpoint(out/'best.pt',checkpoint());prediction_file('best_predictions.npz',values)
        evaluations.append(row)
        if step==settings['pilot_steps']:prediction_file('pilot_predictions.npz',values)
        write_json(out/'evaluations.json',evaluations);write_json(out/'history.json',history)
        save_checkpoint(out/'last.pt',checkpoint());prediction_file('last_predictions.npz',values)
        write_json(out/'progress.json',dict(variant=args.variant,stage='fitting',step=step,total_steps=settings['total_steps'],
            score=metrics['score'],best_score=best_score,best_step=best_step,losses=row['losses'],
            tread_MAE_raw50=metrics['tread_macro_MAE_raw50'],low_cost_bias_raw50=metrics['low_cost_corridor']['bias_raw50'],
            spatial=metrics['spatial'],elapsed_s=row['elapsed_s']))
        print('EVAL',args.variant,step,'S',metrics['score'],'tread_raw',metrics['tread_macro_MAE_raw50'],'best',best_step,flush=True)
        return row

    if not args.resume:evaluate()
    for step in range(step+1,args.stop_step+1):
        model.eval();optimizer.zero_grad(set_to_none=True)
        prediction=forward(train_columns)
        loss,terms=losses(prediction[:n],target[:,:train_columns],parts,edges,protocol['condition_pairs'],settings,args.variant,protocol['condition_groups'])
        gradient=None
        if args.variant=='C' and step%settings['evaluate_every']==0:
            nominal=torch.autograd.grad(terms['nominal'],[p for _,p in named],retain_graph=True,allow_unused=True)
            auxiliary=settings['auxiliary_weight']*terms['auxiliary']+settings['condition_weight']*terms['conditional']
            auxiliary=torch.autograd.grad(auxiliary,[p for _,p in named],retain_graph=True,allow_unused=True)
            gradient=gradient_report(named,nominal,auxiliary)
            del nominal,auxiliary
        loss.backward()
        numeric=scalar(terms)
        row=dict(step=step,losses=numeric)
        if step%settings['evaluate_every']==0:
            row['gradient_norms']={group:float(sum(p.grad.detach().float().square().sum() for name,p in named
                if name.split('.')[0]==group and p.grad is not None).sqrt()) for group in MODULES}
        if gradient is not None:row['nominal_auxiliary_gradient']=gradient
        optimizer.step();scheduler.step();history.append(row)
        del prediction,loss,terms
        if step%settings['evaluate_every']==0:evaluate()
        elif step%5==0:
            latest=evaluations[-1]['metrics']
            write_json(out/'progress.json',dict(variant=args.variant,stage='fitting',step=step,total_steps=settings['total_steps'],
                best_step=best_step,best_score=best_score,losses=numeric,elapsed_s=prior_elapsed+time.perf_counter()-began,
                latest_evaluation_step=evaluations[-1]['step'],tread_MAE_raw50=latest['tread_macro_MAE_raw50'],
                low_cost_bias_raw50=latest['low_cost_corridor']['bias_raw50'],spatial=latest['spatial'],
                parameter_response=latest['response']))
    changed_buffers=[k for k,v in model.named_buffers() if not torch.equal(initial_buffers[k],v.detach().cpu())]
    parameter_changes={group:dict(tensors=0,changed=0) for group in MODULES}
    for name,p in named:
        group=name.split('.')[0];parameter_changes[group]['tensors']+=1
        parameter_changes[group]['changed']+=int(not torch.equal(p.detach().cpu(),source['model'][name]))
    save_json=dict(variant=args.variant,step=step,best_step=best_step,best_score=best_score,normalization_buffers_changed=changed_buffers,
        all_model_parameters_trainable=all(p.requires_grad for p in model.parameters()),active_cores=protocol['active_cores'],
        supervised_queries=n,observation_only_points=4,true_batch_size=len(protocol['active_cores']),
        elapsed_s=prior_elapsed+time.perf_counter()-began,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        parameter_changes=parameter_changes)
    write_json(out/'training_complete.json',save_json)
    print('TRAINING_COMPLETE',args.variant,step,flush=True)


if __name__=='__main__':main()
