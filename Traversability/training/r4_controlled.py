"""Point, same-support spatial, and conditional losses for the R4 diagnosis."""
import numpy as np
import torch
from torch.nn import functional as F

from Traversability.training.condition_response import balanced_difference_loss
from Traversability.experiments.r4_controlled_data import point_metrics,TREAD_PARTS,LANDING_PARTS


def balanced_means(values,active):
    n=active.sum(0);stable=~active;m=stable.sum(0)
    changed=(values*active).sum(0)/n.clamp_min(1)
    unchanged=(values*stable).sum(0)/m.clamp_min(1)
    return (changed+unchanged)/((n>0).float()+(m>0).float())


def point_space(pred,target,parts,edges,beta,threshold):
    errors=F.smooth_l1_loss(pred,target,beta=beta,reduction='none')
    means=torch.stack([errors[parts==i].mean(0) for i in range(11)])
    point=.6*means[TREAD_PARTS].mean(0)+.4*means[LANDING_PARTS].mean(0)
    u,v=edges.T
    truth=target[v]-target[u];prediction=pred[v]-pred[u]
    spatial=F.smooth_l1_loss(prediction,truth,beta=beta,reduction='none')
    space=balanced_means(spatial,truth.abs()>threshold)
    return point,space


def losses(pred,target,parts,edges,pairs,config,variant,groups):
    point,space=point_space(pred,target,parts,edges,config['smooth_l1_beta'],config['change_threshold'])
    nominal=point[0]+config['space_weight']*space[0]
    original=F.smooth_l1_loss(pred[:,0],target[:,0],beta=config['smooth_l1_beta'])
    auxiliary=pred.new_zeros(());conditional=pred.new_zeros(())
    if variant=='A':total=original
    elif variant=='B':total=nominal
    else:
        columns=[list(dict.fromkeys(i for i in g if i!=0)) for g in groups]
        auxiliary=torch.stack([(point[g]+config['space_weight']*space[g]).mean() for g in columns]).mean()
        conditional=balanced_difference_loss(pred,target,pairs,config['smooth_l1_beta'],config['change_threshold'],[1.]*4)
        total=nominal+config['auxiliary_weight']*auxiliary+config['condition_weight']*conditional
    return total,dict(total=total,original_absolute=original,balanced_absolute=point[0],spatial=space[0],
                      nominal=nominal,auxiliary=auxiliary,conditional=conditional)


def spatial_metrics(gt,pred,edges,threshold):
    u,v=edges.T;truth=gt[v]-gt[u];values=pred[v]-pred[u];active=abs(truth)>threshold
    result={}
    for name,mask in [('changed',active),('stable',~active),('large_change',abs(truth)>.2)]:
        result[name]=dict(edges=int(mask.sum()),delta_MAE=float(abs(values[mask]-truth[mask]).mean()) if mask.any() else None,
            teacher_abs_delta=float(abs(truth[mask]).mean()) if mask.any() else None,
            prediction_abs_delta=float(abs(values[mask]).mean()) if mask.any() else None)
    result['balanced_delta_MAE']=float(np.mean([result[k]['delta_MAE'] for k in ['changed','stable'] if result[k]['edges']]))
    return result


def evaluate_arrays(prediction,teacher,protocol):
    n=len(teacher['query_xyz']);target=teacher['target'];parts=teacher['part'];edges=teacher['edges']
    gt,pred=target[:,0],prediction[:n,0]
    rows=[dict(part=i,**point_metrics(gt[parts==i],pred[parts==i])) for i in range(11)]
    tread=np.mean([rows[i]['MAE'] for i in TREAD_PARTS]);landing=np.mean([rows[i]['MAE'] for i in LANDING_PARTS])
    spatial=spatial_metrics(gt,pred,edges,protocol['configuration']['change_threshold'])
    result=dict(nominal=point_metrics(gt,pred),parts=rows,tread_macro_MAE=float(tread),tread_macro_MAE_raw50=float(50*tread),
        landing_macro_MAE=float(landing),spatial=spatial,
        score=float(.6*tread+.4*landing+.25*spatial['balanced_delta_MAE']),conditions=[],response={})
    body=(parts>=1)&(parts<=9);low=body&(gt<.2)
    result['low_cost_corridor']=point_metrics(gt[low],pred[low])
    for i,c in enumerate(teacher['conditions']):
        result['conditions'].append(dict(column=i,condition=c.tolist(),split='train' if i<protocol['train_condition_count'] else 'heldout',
            **point_metrics(target[:,i],prediction[:n,i])))
    for name,groups in [('training',protocol['condition_groups']),('heldout',protocol['heldout_groups'])]:
        responses=[]
        for p,columns in enumerate(groups):
            other=[i for i in columns if i!=0]
            truth=target[:,other]-target[:,:1];values=prediction[:n,other]-prediction[:n,:1]
            active=abs(truth)>.02;stable=~active
            changed=float(abs(values[active]-truth[active]).mean()) if active.any() else None
            unchanged=float(abs(values[stable]-truth[stable]).mean()) if stable.any() else None
            responses.append(dict(parameter=p,changed_delta_MAE=changed,stable_delta_MAE=unchanged,
                changed_pairs=int(active.sum()),stable_pairs=int(stable.sum()),
                response_error=float(np.mean([v for v in [changed,unchanged] if v is not None])),
                response_gain=float((values[active]*truth[active]).sum()/(truth[active]**2).sum()) if active.any() else None))
        result['response'][name]=responses
    result['watch_prediction']=prediction[n:].tolist()
    result['watch_target']=teacher['watch_target'].tolist()
    return result
