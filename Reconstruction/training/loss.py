"""Mesh surface supervision and differentiable Flow endpoint geometry loss."""
import torch
from torch.nn import functional as F

OBSERVED, MISSING, NEAR_EMPTY, GLOBAL_EMPTY = range(4)


def group_means(values,groups):
    sums=torch.zeros(4,device=values.device,dtype=values.dtype).scatter_add_(0,groups,values)
    counts=torch.bincount(groups,minlength=4)
    return sums/counts.clamp_min(1),counts


def weighted_active(means,counts,weights):
    weight=means.new_tensor(weights)*(counts>0)
    return (means*weight).sum()/weight.sum().clamp_min(1)


def geometry_loss(field,queries,stage='ae',config=None):
    cfg=dict(beta=.1,missing_weight=2.,near_empty_weight=2.)
    cfg.update(config or {})
    losses=[];classification=[];position=[]
    for q in queries:
        logits,delta=field.predict_cells(q['xyz'],q['size'])
        groups=q['group']
        bce=F.binary_cross_entropy_with_logits(logits.float(),q['label'],reduction='none')
        means,counts=group_means(bce,groups)
        missing_weight=cfg['missing_weight'] if stage=='flow' else 1.
        positive=weighted_active(means,counts,[1.,missing_weight,0.,0.])
        negative=weighted_active(means,counts,[0.,0.,cfg['near_empty_weight'],1.])
        cls=.5*(positive+negative)
        point_loss=F.smooth_l1_loss(delta.float()/q['size'],q['offset']/q['size'],
                                    beta=cfg['beta'],reduction='none').mean(1)
        means,counts=group_means(point_loss,groups)
        loc=weighted_active(means,counts,[1.,missing_weight,0.,0.])
        losses.append(cls+loc);classification.append(cls.detach());position.append(loc.detach())
    return torch.stack(losses).mean(),dict(surface=float(torch.stack(classification).mean()),
                                         offset=float(torch.stack(position).mean()))


def geometry_weight(epoch_progress,maximum=.1,warmup_epochs=5):
    return maximum*min(1.,max(0.,epoch_progress/warmup_epochs))


def flow_matching_loss(flow,ae,sample,target_latent,queries,epoch_progress,
                       config=None,noise=None,time=None):
    """One velocity evaluation, one endpoint decode. AE params stay frozen.

    The caller freezes ae parameters with requires_grad_(False), without using
    no_grad around its decoder. Spatial gradients must reach the velocity net.
    """
    cfg=dict(geometry_weight=.1,geometry_warmup_epochs=5)
    cfg.update(config or {})
    target=flow.normalized(target_latent.float())
    noise=torch.randn_like(target) if noise is None else noise
    time=torch.rand(target.shape[0],device=target.device) if time is None else time
    t=time.reshape(-1,1,1,1,1)
    mixed=(1-t)*noise+t*target
    velocity=flow.velocity(mixed,time,flow.encode_condition(sample))
    fm=F.mse_loss(velocity.float(),target-noise)
    endpoint=mixed+(1-t)*velocity.float()
    physical_endpoint=endpoint*flow.latent_std+flow.latent_mean
    field=ae.decoder(physical_endpoint,sample['origin'])
    geo,parts=geometry_loss(field,queries,'flow',cfg)
    weight=geometry_weight(epoch_progress,cfg['geometry_weight'],cfg['geometry_warmup_epochs'])
    return fm+weight*geo,dict(flow_mse=float(fm.detach()),endpoint_geometry=float(geo.detach()),
                              geometry_weight=weight,**parts)
