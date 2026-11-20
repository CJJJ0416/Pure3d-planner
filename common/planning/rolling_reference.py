"""Arc-length sampling and an unchanged near prefix for rolling references."""
import numpy as np


class FrozenPrefixInvalid(ValueError):
    """The latest map no longer supports retaining the executing near reference."""


def arc_lengths(path):
    p=np.asarray(path,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('Expected at least two finite reference points')
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]


def sample_reference(path,spacing=.1):
    """Keep every original corner, adding samples so no arc gap exceeds spacing."""
    p=np.asarray(path,dtype=float);s=arc_lengths(p)
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Invalid sampling spacing')
    keep=np.r_[True,np.diff(s)>1e-9];p,s=p[keep],s[keep]
    if len(p)<2:raise ValueError('Degenerate reference')
    samples=np.unique(np.r_[s,np.arange(0,s[-1],spacing)])
    return np.column_stack([np.interp(samples,s,p[:,k]) for k in range(3)])


def rolling_prefix(route,body,keep_fraction=1./3,spacing=.1):
    p=np.asarray(route,dtype=float);s=arc_lengths(p);body=np.asarray(body,dtype=float)
    if not 0<keep_fraction<1:raise ValueError('Invalid frozen fraction')
    delta=np.diff(p,axis=0);length=np.diff(s)
    alpha=np.clip(np.einsum('ij,ij->i',body-p[:-1],delta)/np.maximum(length**2,1e-18),0.,1.)
    points=p[:-1]+alpha[:,None]*delta
    i=int(np.argmin(np.linalg.norm(points-body,axis=1)))
    distance=float(np.linalg.norm(points[i]-body))
    progress=float(s[i]+alpha[i]*length[i]);remaining=float(s[-1]-progress)
    if remaining<.3 or distance>.5:
        raise ValueError('Insufficient remaining reference or robot too far from it')
    end=progress+keep_fraction*remaining
    samples=np.r_[progress,s[(s>progress+1e-9)&(s<end-1e-9)],end]
    prefix=np.column_stack([np.interp(samples,s,p[:,k]) for k in range(3)])
    return sample_reference(prefix,spacing),dict(projected_arc_m=progress,remaining_arc_m=remaining,
        kept_arc_m=end-progress,keep_fraction=keep_fraction,projection_distance_m=distance)


def join_reference(prefix,tail,spacing=.1):
    prefix=np.asarray(prefix,dtype=float);tail=np.asarray(tail,dtype=float)
    arc_lengths(prefix)
    if tail.ndim!=2 or tail.shape[1]!=3 or not len(tail) or not np.isfinite(tail).all():
        raise ValueError('Invalid replacement tail')
    gap=float(np.linalg.norm(prefix[-1]-tail[0]))
    if gap>.2:raise ValueError('Replacement tail snap exceeds 0.2m')
    # Caller must validate this connector in the existing graph, never bridge gaps blindly.
    combined=np.vstack((prefix,tail[1:] if gap<1e-9 else tail))
    return sample_reference(combined,spacing),gap
