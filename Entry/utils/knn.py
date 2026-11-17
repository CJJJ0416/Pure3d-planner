"""Exact CPU KNN statistics and explicit geometric outlier masks.

Scores/costs and mesh ground truth are not inputs to either filter rule.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.spatial import cKDTree
import hashlib
from Entry.utils.local_map import ContextSnapshot


@dataclass(frozen=True)
class KNNOptions:
    k: int = 16
    far_k: int = 64
    alpha: float = 2.
    small_radius: float = .15
    large_radius: float = .5
    min_other_neighbors: int = 3
    workers: int = 4

    def __post_init__(self):
        integer_fields=[self.k,self.far_k,self.min_other_neighbors,self.workers]
        if (any(not isinstance(v,(int,np.integer)) or isinstance(v,bool) for v in integer_fields) or
                not np.isfinite([self.alpha,self.small_radius,self.large_radius]).all() or
                self.k < 1 or self.far_k < self.k or self.alpha <= 0 or
                not 0 < self.small_radius < self.large_radius or self.min_other_neighbors < 1 or self.workers < 1):
            raise ValueError('Invalid KNN options')


def statistics(xyz, options=KNNOptions(), full=True):
    """Exclude one self point from both KNN distances and radius counts."""
    started=time.perf_counter()
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all():
        raise ValueError('Expected at least two finite XYZ points')
    tree=cKDTree(xyz,leafsize=16)
    index_ready=time.perf_counter()
    count=min(options.far_k if full else options.k,len(xyz)-1)
    distances=tree.query(xyz,k=count+1,workers=options.workers,eps=0.)[0][:,1:]
    knn_ready=time.perf_counter()
    mean=distances[:,:min(options.k,count)].mean(1)
    values=dict(mean_distance=mean,k_distance=distances[:,min(options.k,count)-1].copy())
    if full:
        values['far_distance']=distances[:,-1].copy()
        values['mean_to_far_ratio']=np.divide(mean,distances[:,-1],out=np.zeros_like(mean),where=distances[:,-1]>0)
    radius_started=time.perf_counter()
    values['small_radius_count']=tree.query_ball_point(xyz,options.small_radius,return_length=True,workers=options.workers)-1
    small_ready=time.perf_counter()
    if full:
        values['large_radius_count']=tree.query_ball_point(xyz,options.large_radius,return_length=True,workers=options.workers)-1
    done=time.perf_counter()
    mu=float(mean.mean());sigma=float(mean.std())
    threshold=mu+options.alpha*sigma
    outlier=mean>threshold
    masks=dict(sor=~outlier,conservative=~(outlier&(values['small_radius_count']<options.min_other_neighbors)))
    finish=time.perf_counter()
    timing=dict(index_s=index_ready-started,knn_s=knn_ready-index_ready,
                feature_s=radius_started-knn_ready,small_radius_s=small_ready-radius_started,
                large_radius_s=done-small_ready,decision_s=finish-done,total_s=finish-started)
    return values,masks,dict(options=asdict(options),full_statistics=full,mean=mu,std=sigma,
                            threshold_m=threshold,points=len(xyz),timing=timing,
                            rule_sor='mean KNN distance > mean + alpha * population std',
                            rule_conservative='SOR outlier AND other neighbors within small radius < minimum')


def options_from_config(value):
    value=value or {}
    if not isinstance(value.get('enabled',False),bool):raise ValueError('knn_filter.enabled must be boolean')
    if value.get('method','conservative')!='conservative':raise ValueError('Online KNN supports the retained conservative method')
    radius=value.get('radius_m',.15);k=value.get('k',16)
    return KNNOptions(k=k,far_k=max(64,k),alpha=value.get('alpha',2.),small_radius=radius,
                      large_radius=max(.5,2*radius),min_other_neighbors=value.get('min_other_neighbors',3),
                      workers=value.get('workers',4))


@dataclass(frozen=True)
class FilteredSnapshot(ContextSnapshot):
    raw_context_coord: np.ndarray
    context_keep: np.ndarray
    raw_coord: np.ndarray
    query_keep: np.ndarray


class LocalKNNFilter:
    def __init__(self,config=None):
        self.config=dict(config or {});self.enabled=self.config.get('enabled',False)
        self.options=options_from_config(self.config)
        self.signature=None;self.keep=None;self.context=None;self.mapping=None;self.definition=None
        self.evaluations=0;self.query_signature=None;self.query_version=0

    def apply(self,snapshot):
        if not self.enabled:return snapshot
        started=time.perf_counter()
        raw=np.ascontiguousarray(snapshot.context_coord,dtype=np.float32)
        signature=hashlib.blake2b(memoryview(raw).cast('B') if raw.size else b'',digest_size=20).hexdigest()
        cache_hit=signature==self.signature
        statistics_s=0.
        if not cache_hit:
            tick=time.perf_counter()
            if len(raw)<2:
                keep=np.ones(len(raw),bool)
                definition=dict(threshold_m=None,mean=None,std=None,timing={},skip_reason='fewer_than_two_context_points')
            else:
                _,masks,definition=statistics(raw,self.options,full=False)
                keep=masks['conservative']
            statistics_s=time.perf_counter()-tick
            self.keep=keep.copy();self.context=raw[keep].copy()
            self.mapping=np.full(len(raw),-1,np.int64)
            self.mapping[keep]=np.arange(int(keep.sum()),dtype=np.int64)
            self.definition=definition;self.signature=signature;self.evaluations+=1
            self.keep.flags.writeable=False;self.context.flags.writeable=False;self.mapping.flags.writeable=False
        query_keep=self.keep[snapshot.query_indices].copy()
        rows=self.mapping[snapshot.query_indices[query_keep]].copy()
        query=self.context[rows].copy()
        query_signature=hashlib.blake2b(memoryview(query).cast('B') if query.size else b'',digest_size=20).hexdigest()
        if query_signature!=self.query_signature:self.query_version+=1;self.query_signature=query_signature
        finished=time.perf_counter()
        info=dict(enabled=True,method='conservative',implementation='minimal_statistics',
                  k=self.options.k,alpha=self.options.alpha,radius_m=self.options.small_radius,
                  min_other_neighbors=self.options.min_other_neighbors,workers=self.options.workers,
                  input_context_points=len(raw),kept_context_points=len(self.context),removed_context_points=int((~self.keep).sum()),
                  input_query_points=len(snapshot.coord),kept_query_points=len(query),removed_query_points=int((~query_keep).sum()),
                  threshold_m=self.definition['threshold_m'],cache_hit=cache_hit,evaluations=self.evaluations,
                  filter_s=finished-started,statistics_s=statistics_s,
                  statistics_timing={} if cache_hit else self.definition['timing'],
                  raw_context_signature=signature,filtered_query_revision=self.query_version,
                  filtered_query_signature=query_signature,raw_context_revision=snapshot.context_revision,
                  raw_query_revision=snapshot.query_revision)
        metadata=dict(snapshot.metadata)
        metadata.update(knn_filter=info,raw_context_points=len(raw),raw_query_points=len(snapshot.coord),
                        context_points=len(self.context),query_points=len(query),
                        context_array_bytes=int(self.context.nbytes),query_array_bytes=int(query.nbytes),
                        filtered_query_revision=self.query_version)
        return FilteredSnapshot(snapshot.revision,snapshot.stamp,query,snapshot.blocks,
                                self.context,rows,snapshot.context_revision,snapshot.query_revision,metadata,
                                raw,self.keep,snapshot.coord,query_keep)
