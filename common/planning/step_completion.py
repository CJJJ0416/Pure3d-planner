"""Bounded geometric tread hypotheses before PTv3, independent of predicted cost.

Only observed points seed checked DC-style step connections. Generated points
are appended and labelled, never inserted into the observed LIO context map.
"""
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from common.mapping.local_map import ContextSnapshot


@dataclass(frozen=True)
class CompletedSnapshot(ContextSnapshot):
    observed: ContextSnapshot
    generated_context: np.ndarray
    query_generated_mask: np.ndarray


def observed_snapshot(snapshot):
    return snapshot.observed if isinstance(snapshot, CompletedSnapshot) else snapshot


class LocalStepCompletion:
    def __init__(self, config=None, condition=(.5,.4,.2,0.), voxel_m=.05):
        self.config=dict(config or {})
        self.enabled=self.config.get('enabled',False)
        if not isinstance(self.enabled,bool):raise ValueError('step_completion.enabled must be boolean')
        self.spacing=float(voxel_m)
        self.gateway=float(self.config.get('gateway_spacing_m',.1))
        self.maximum=self.config.get('max_generated_points',25000)
        self.ratio=float(self.config.get('max_generated_ratio',.5))
        self.condition=np.asarray(condition,dtype=float)
        if (self.condition.shape!=(4,) or not np.isfinite(self.condition).all() or
                not 0<self.condition[1]<np.pi/2 or self.condition[2]<=0 or
                not np.isfinite([self.spacing,self.gateway,self.ratio]).all() or
                min(self.spacing,self.gateway,self.ratio)<=0 or
                not isinstance(self.maximum,int) or isinstance(self.maximum,bool) or self.maximum<1):
            raise ValueError('Invalid step completion configuration')
        self.signature=None;self.generated=None;self.definition=None;self.evaluations=0

    def _complete(self, xyz):
        from planner.light_planner import PlannerConfig
        from planner.light_planner.surface import SparseSurfacePlanner,SurfaceOptions
        from scipy import sparse
        from scipy.spatial import cKDTree
        from scipy.sparse.csgraph import connected_components
        from planner.light_planner.graph.adjacency import sparse_step_support,step_surface_pairs,connect_step_surfaces
        started=time.perf_counter()
        cfg=PlannerConfig(cost_threshold=1.,dc_input_cost_max=1.,
                          slope_max=float(self.condition[1]),step_max=float(self.condition[2]))
        options=SurfaceOptions(step_connections=False,max_patch_span=.04,
                               step_gateway_spacing_m=self.gateway)
        planner=SparseSurfacePlanner(cfg,options=options)
        try:
            # Dummy costs only allow the existing geometric checks to inspect
            # every observed patch. No model prediction is read or fabricated.
            planner.load_arrays(xyz,np.zeros(len(xyz),np.float32))
            nodes=planner.node_xyz
            base=planner.step_graph.copy()
            rows=np.repeat(np.arange(len(nodes)),np.diff(base.indptr));cols=base.indices
            base.data[np.abs(nodes[rows,2]-nodes[cols,2])>cfg.minimum_step_height]=0
            base.eliminate_zeros()
            # Inspect geometry before enumerating pairs. Unfiltered contexts have
            # many vertical-wall patches that would otherwise dominate pair work.
            xy_tree=cKDTree(nodes[:,:2])
            normals,support,overhead=sparse_step_support(nodes,planner.full_xyz,planner.frame,
                options.xy_resolution,options.z_resolution,xy_tree=xy_tree)
            clear=support & (overhead>=cfg.interval_free)
            eligible=np.flatnonzero(clear & (normals[:,2]>=np.cos(cfg.slope_max)))
            candidates=(eligible[step_surface_pairs(nodes[eligible],cfg.step_max,cfg.step_horizontal_radius,
                cfg.gravity_up,cfg.minimum_step_height,xy_tree=cKDTree(nodes[eligible,:2]))]
                if len(eligible) else np.empty((0,2),dtype=np.int64))
            _,labels=connected_components(base,directed=False)
            candidates=candidates[labels[candidates[:,0]]!=labels[candidates[:,1]]]
            linked,connections=connect_step_surfaces(nodes,normals,np.arange(len(nodes)),base,
                cfg.step_max,cfg.slope_max,cfg.step_horizontal_radius,cfg.gravity_up,cfg.minimum_step_height,
                support_mask=clear,candidate_pairs=candidates,gateway_spacing=self.gateway)
            graph=sparse.triu(linked,k=1).tocoo()
            selected=np.abs(nodes[graph.row,2]-nodes[graph.col,2])>cfg.minimum_step_height
            pairs=np.column_stack((graph.row[selected],graph.col[selected]))
            geometry_ready=time.perf_counter()
            budget=min(self.maximum,int(len(xyz)*self.ratio))
            pieces=[]
            # Distribute the budget over the checked transitions before sampling.
            per_edge=3*int(np.ceil(cfg.step_horizontal_radius/self.spacing))+int(np.ceil(cfg.step_max/self.spacing))+3
            limit=max(1,budget//max(per_edge,1))
            limited=len(pairs)>limit
            if limited:pairs=pairs[np.linspace(0,len(pairs)-1,limit,dtype=int)]
            for a,b in pairs:
                lo,hi=nodes[a],nodes[b]
                if lo[2]>hi[2]:lo,hi=hi,lo
                middle=(lo[:2]+hi[:2])/2
                # Extend each observed tread at its own height. The vertical
                # gap remains a step and contributes no synthetic support rows.
                for left,right in ((lo,np.r_[middle,lo[2]]),(np.r_[middle,hi[2]],hi)):
                    n=max(1,int(np.ceil(np.linalg.norm(right-left)/self.spacing)))
                    pieces.append(left+(right-left)*np.arange(n+1)[:,None]/n)
            generated=np.asarray(np.vstack(pieces) if pieces else np.empty((0,3)),np.float32)
            if len(generated):
                # Original rows win every occupied world voxel. Generated rows
                # never replace actual returns, including high-cost wall points.
                keys=np.floor(np.vstack((xyz,generated)).astype(np.float64)/self.spacing).astype(np.int64)
                _,first=np.unique(keys,axis=0,return_index=True)
                rows=np.sort(first[first>=len(xyz)]-len(xyz))
                generated=generated[rows]
                if len(generated)>budget:
                    generated=generated[np.linspace(0,len(generated)-1,budget,dtype=int)]
                    limited=True
            report=dict(algorithm='observed_tread_extensions_v1',
                        cost_used_for_completion=False,network_cost_overridden=False,
                        observed_context_points=len(xyz),generated_context_points=len(generated),
                        checked_step_connections=connections,
                        sampled_connections=len(pairs),point_budget=budget,capacity_limited=limited,
                        spacing_m=self.spacing,gateway_spacing_m=self.gateway,
                        step_max_m=cfg.step_max,slope_max_rad=cfg.slope_max,
                        geometry_s=geometry_ready-started,sampling_s=time.perf_counter()-geometry_ready)
            return generated,report
        finally:planner.close()

    def apply(self, snapshot):
        if not self.enabled:return snapshot
        if isinstance(snapshot,CompletedSnapshot):raise ValueError('Completion requires observed input, not a previous completion')
        started=time.perf_counter()
        raw=np.ascontiguousarray(snapshot.context_coord,dtype=np.float32)
        signature=hashlib.blake2b(raw.tobytes(),digest_size=20).hexdigest()
        hit=signature==self.signature
        if not hit:
            if len(raw)<3:
                generated=np.empty((0,3),np.float32)
                definition=dict(skip_reason='fewer_than_three_observed_points',
                                generated_context_points=0,cost_used_for_completion=False)
            else:generated,definition=self._complete(raw)
            self.generated=generated;self.definition=definition;self.signature=signature
            self.evaluations+=1
        generated=self.generated
        # Clamp to retained observed context. Stable-core context never consumes
        # hypotheses from an older prediction as evidence for further completion.
        for lo,hi in [(snapshot.metadata.get('context_lower'),snapshot.metadata.get('context_upper'))]:
            if lo is not None and hi is not None:
                generated=generated[((generated>=lo)&(generated<hi)).all(1)]
        context=np.ascontiguousarray(np.vstack((raw,generated)),dtype=np.float32)
        lower,upper=snapshot.metadata.get('roi_lower'),snapshot.metadata.get('roi_upper')
        inside=np.ones(len(generated),bool) if lower is None else ((generated>=lower)&(generated<upper)).all(1)
        added=np.flatnonzero(inside)
        query=np.ascontiguousarray(np.vstack((snapshot.coord,generated[added])),dtype=np.float32)
        indices=np.r_[snapshot.query_indices,len(raw)+added].astype(np.int64)
        mask=np.r_[np.zeros(len(snapshot.coord),bool),np.ones(len(added),bool)]
        info=dict(self.definition,enabled=True,cache_hit=hit,evaluations=self.evaluations,
                  observed_context_signature=signature,generated_context_points=len(generated),
                  observed_query_points=len(snapshot.coord),generated_query_points=len(added),
                  completion_s=time.perf_counter()-started,
                  provenance='generated support hypothesis, not an observed return')
        metadata=dict(snapshot.metadata,step_completion=info,context_points=len(context),
                      query_points=len(query),context_array_bytes=context.nbytes,query_array_bytes=query.nbytes)
        return CompletedSnapshot(snapshot.revision,snapshot.stamp,query,snapshot.blocks,
                                 context,indices,snapshot.context_revision,snapshot.query_revision,
                                 metadata,snapshot,generated,mask)
