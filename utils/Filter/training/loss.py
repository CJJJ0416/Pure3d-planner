"""Five-group continuous loss, without hard filtering the training points."""
import torch
from torch.nn import functional as F


def regression_loss(prediction,target,group,beta=.1,semantic=None):
    error=F.smooth_l1_loss(prediction.float(),target.float(),beta=beta,reduction='none')
    group=group.long()
    if semantic is not None:
        bucket=group*11+semantic.long()
        sums=torch.zeros(55,device=prediction.device).index_add_(0,bucket,error).reshape(5,11)
        counts=torch.bincount(bucket,minlength=55).reshape(5,11)
        present=counts>0
        means=((sums/counts.clamp_min(1))*present).sum(1)/present.sum(1).clamp_min(1)
        return means[present.any(1)].mean()
    sums=torch.zeros(5,device=prediction.device).index_add_(0,group,error)
    counts=torch.bincount(group,minlength=5)
    present=counts>0
    # Missing groups contribute no loss. Existing labels remain untouched.
    return (sums/counts.clamp_min(1))[present].mean()
