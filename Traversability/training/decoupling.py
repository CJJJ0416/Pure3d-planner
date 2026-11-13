"""Four-parameter losses and scene-macro, two-bank evaluation."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from Traversability.data.dataset import PointBlocks, to_device
from Traversability.training.condition_response import pair_deltas, response_totals
from Traversability.training.train import autocast, make_loader

NAMES=['clearance_min','slope_max','step_max','stand_ratio']


def decoupling_loss(prediction,target,facility,pairs,config):
    valid=torch.isfinite(target)
    values=torch.where(valid,target,0.)
    errors=F.smooth_l1_loss(prediction,values,beta=config['smooth_l1_beta'],reduction='none')
    if config['region_balanced']:
        regions=[valid&facility[:,None],valid&~facility[:,None]]
        absolute=torch.stack([(errors*m).sum()/m.sum() for m in regions if m.any()]).mean()
    else:
        absolute=(errors*valid).sum()/valid.sum()
    terms=[];parts={'absolute':absolute}
    for name,group in zip(NAMES,pairs):
        predicted,truth,common=pair_deltas(prediction,target,group)
        error=F.smooth_l1_loss(predicted,truth,beta=config['smooth_l1_beta'],reduction='none')
        active=common&(truth.abs()>config['change_threshold'])
        stable=common&~active
        present=[]
        for label,mask in [('changed',active),('stable',stable)]:
            value=(error*mask).sum()/mask.sum().clamp_min(1)
            parts[name+'_'+label]=value
            if mask.any():present.append(value)
        terms.append(torch.stack(present).mean())
    weights=prediction.new_tensor(config['parameter_weights'])
    conditional=(torch.stack(terms)*weights).sum()/weights.sum()
    total=absolute+config['condition_difference_weight']*conditional
    parts.update(conditional=conditional,total=total)
    return total,parts


def mean(values):
    v=[x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def scene_result(record):
    result=dict(cost_mae=record['absolute']/record['count'],points=record['count'],parameter_response={})
    for name,r in zip(NAMES,record['response']):
        e,g,p,correct,n,se,sp,sn,dot,square=r
        result['parameter_response'][name]=dict(
            changed_delta_mae=float(e/n) if n else None,
            stable_delta_mae=float(se/sn) if sn else None,
            teacher_abs_delta=float(g/n) if n else None,
            prediction_abs_delta=float(p/n) if n else None,
            direction_correct_pct=float(100*correct/n) if n else None,
            stable_prediction_abs_delta=float(sp/sn) if sn else None,
            response_gain=float(dot/square) if square else None,
            changed_pairs=int(n),stable_pairs=int(sn))
    return result


def macro(records):
    responses={}
    for name in NAMES:
        keys=['changed_delta_mae','stable_delta_mae','teacher_abs_delta','prediction_abs_delta',
              'direction_correct_pct','stable_prediction_abs_delta','response_gain']
        r={k:mean([v['parameter_response'][name][k] for v in records]) for k in keys}
        r['response_error']=mean([r['changed_delta_mae'],r['stable_delta_mae']])
        r['changed_pairs']=sum(v['parameter_response'][name]['changed_pairs'] for v in records)
        r['stable_pairs']=sum(v['parameter_response'][name]['stable_pairs'] for v in records)
        responses[name]=r
    mae=mean([v['cost_mae'] for v in records])
    response_error=mean([r['response_error'] for r in responses.values()])
    return dict(cost_mae=mae,cost_mae_raw=50*mae,parameter_response=responses,
                mean_response_error=response_error,score=mae+response_error,scenes=len(records))


@torch.no_grad()
def evaluate_decoupling(model,config,split,predictions=None):
    """Encode geometry once per batch, then evaluate both banks at identical XYZ."""
    dataset=PointBlocks(config['data'],split,config['model'],config['seed'],condition_bank=0)
    device=next(model.parameters()).device
    model.eval()
    records=defaultdict(lambda:dict(absolute=0.,count=0,response=np.zeros((4,10),np.float64)))
    costs={s['id']:np.load(s['arrays']['cost'],mmap_mode='r') for s in dataset.metadata['scenes']
           if any(row['scene']==s['id'] for row in dataset.rows)}
    snapshots={name:[] for name in ['coord','prediction','target','scene','bank','query_index']}
    for batch in make_loader(dataset,config,device):
        batch=to_device(batch,device)
        with autocast(device,config['amp']):geometry=model.encode_geometry(batch)
        qb=batch['query_batch']
        query_indices=batch['query_index'].cpu().numpy()
        qb_cpu=qb.cpu().numpy()
        for bank in range(2):
            conditions=[];target=np.empty((len(qb),40),np.float32)
            for b,name in enumerate(batch['scene']):
                scene=dataset.scene_rows[name];columns=scene['profile_banks'][bank]
                conditions.append([[scene['profiles'][i]['condition'][j] for j in dataset.condition_columns] for i in columns])
                mask=qb_cpu==b
                target[mask]=costs[name][query_indices[mask]][:,columns]
            conditions=torch.tensor(conditions,dtype=torch.float32,device=device)
            truth=torch.from_numpy(target).to(device)
            with autocast(device,config['amp']):prediction=model.decode_cost(geometry,conditions,qb)
            for b,name in enumerate(batch['scene']):
                take=qb==b;p,t=prediction[take],truth[take]
                valid=torch.isfinite(t)
                r=records[name]
                r['absolute']+=float(torch.where(valid,abs(p-t),0.).sum().cpu())
                r['count']+=int(valid.sum().cpu())
                r['response']+=response_totals(p,t,dataset.condition_pairs,config['change_threshold']).cpu().numpy()
            if predictions is not None:
                xyz=batch['query_coord']+batch['origin'][qb]
                snapshots['coord'].append(xyz.cpu().numpy())
                snapshots['prediction'].append(prediction.cpu().numpy())
                snapshots['target'].append(target)
                snapshots['scene'].append(np.asarray(batch['scene'])[qb_cpu])
                snapshots['bank'].append(np.full(len(qb),bank,np.int8))
                snapshots['query_index'].append(query_indices)
    scenes={name:dict(source=dataset.scene_rows[name]['source'],kind=dataset.scene_rows[name]['kind'],
                       **scene_result(record)) for name,record in records.items()}
    groups={name:macro([v for v in scenes.values() if v['source']==name])
            for name in ['new','replay'] if any(v['source']==name for v in scenes.values())}
    if 'new' in groups and 'replay' in groups:
        score=.7*groups['new']['score']+.3*groups['replay']['score']
    else:score=next(iter(groups.values()))['score']
    if predictions is not None:
        Path(predictions).parent.mkdir(parents=True,exist_ok=True)
        np.savez(predictions,**{k:np.concatenate(v) for k,v in snapshots.items()},profiles=np.asarray(dataset.profile_ids))
    return dict(selection_score=score,groups=groups,scenes=scenes,banks_evaluated=2,
                aggregation='Per scene first, then equal scene means. Changed/stable each half. New/replay=0.7/0.3.')
