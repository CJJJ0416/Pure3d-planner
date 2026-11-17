"""Sparse, gravity-aligned observed support graph; no EDT or DC dependency."""
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

from .graph.astar import NativeAStar
from .native import compile_cached
from .optimization.gpmp import GravityGPMP


@dataclass(frozen=True)
class SurfaceOptions:
    xy_resolution: float = .1
    z_resolution: float = .05
    # Representation only: long vertical runs are split, never rejected.
    max_patch_span: float = .1
    validation_spacing: float = .025
    slice_dh: float = .5


def gravity_frame(up):
    up = np.asarray(up, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
        raise ValueError('Invalid gravity_up')
    up = up / np.linalg.norm(up)
    axis = np.eye(3)[np.argmin(np.abs(up))]
    horizontal = axis - up*np.dot(axis, up)
    horizontal /= np.linalg.norm(horizontal)
    return np.column_stack((horizontal, np.cross(up, horizontal), up))


class SparseSurfacePlanner:
    backend = 'surface'

    def __init__(self, config, build_folder=None, options=None):
        self.config = config
        self.options = options or SurfaceOptions()
        self.frame = gravity_frame(config.gravity_up)
        self.build_folder = Path(build_folder or Path(__file__).resolve().parents[1]/'.cache/native/requests')
        o = self.options
        values = np.r_[o.xy_resolution,o.z_resolution,o.max_patch_span,o.validation_spacing,o.slice_dh]
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Invalid sparse map options')
        limit = config.cost_threshold
        if (not np.isscalar(limit) or not np.isfinite(limit) or not 0 < limit <= 1 or
                config.dc_input_cost_max != limit):
            raise ValueError('Surface cost gates must match and lie in (0, 1]')
        library, self.compile_s, self.compile_cache_hit = compile_cached(
            Path(__file__).with_name('geometry')/'surface_native.cpp', self.build_folder, 'libsurface')
        self.lib = ctypes.CDLL(str(library))
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        u32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')
        self.lib.surface_build.argtypes = [f32, f32, ctypes.c_uint64, f64]
        self.lib.surface_build.restype = ctypes.c_void_p
        self.lib.surface_error.restype = ctypes.c_char_p
        self.lib.surface_free.argtypes = [ctypes.c_void_p]
        self.lib.surface_sizes.argtypes = [ctypes.c_void_p, u64, f64]
        self.lib.surface_copy_full.argtypes = [ctypes.c_void_p, f64]
        self.lib.surface_copy_patches.argtypes = [ctypes.c_void_p, f64]
        self.lib.surface_copy_graph.argtypes = [ctypes.c_void_p, f64, f64, u64, u32, f64]
        self.lib.surface_validate.argtypes = [ctypes.c_void_p, f64, ctypes.c_uint64, ctypes.c_double, u64]
        self.lib.surface_validate.restype = ctypes.c_int
        self.lib.surface_match_layer.argtypes = [ctypes.c_void_p,ctypes.c_double,ctypes.c_double,ctypes.c_double]
        self.lib.surface_match_layer.restype = ctypes.c_int64
        self.handle = None
        self._optimizer = None

    def close(self):
        if getattr(self,'handle',None) is not None:
            self.lib.surface_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def load_arrays(self, xyz, cost):
        started = time.perf_counter()
        xyz = np.asarray(xyz)
        cost = np.asarray(cost)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or cost.shape != (len(xyz),) or not len(xyz):
            raise ValueError('Expected nonempty aligned XYZ and cost arrays')
        self.close()
        self._optimizer = None
        local = np.ascontiguousarray(xyz if np.array_equal(self.frame, np.eye(3)) else xyz@self.frame, dtype=np.float32)
        values = np.ascontiguousarray(cost, dtype=np.float32)
        o, c = self.options, self.config
        parameters = np.array([o.xy_resolution,o.z_resolution,c.dc_input_cost_max,o.max_patch_span,o.slice_dh],dtype=np.float64)
        self.handle = self.lib.surface_build(local, values, len(local), parameters)
        if not self.handle:
            raise ValueError(self.lib.surface_error().decode())
        counts, timing = np.zeros(15, np.uint64), np.zeros(5)
        self.lib.surface_sizes(self.handle, counts, timing)
        full_count, node_count, edge_count = map(int, counts[[3,12,13]])
        full = np.empty((full_count,7), np.float64)
        self.lib.surface_copy_full(self.handle, full)
        self.patches=np.empty((int(counts[14]),9),np.float64)
        self.lib.surface_copy_patches(self.handle,self.patches)
        nodes = np.empty((node_count,3), np.float64)
        self.node_cost = np.empty(node_count, np.float64)
        offsets = np.empty(node_count+1,np.uint64)
        neighbors = np.empty(edge_count,np.uint32)
        self.detail = np.empty((node_count,6),np.float64)
        self.lib.surface_copy_graph(self.handle,nodes,self.node_cost,offsets,neighbors,self.detail)
        self.node_xyz = np.ascontiguousarray(nodes@self.frame.T)
        self.xyz = self.node_xyz
        self.cost = self.node_cost
        self.full_xyz = np.ascontiguousarray(full[:,:3]@self.frame.T,dtype=np.float32)
        self.full_cost = np.ascontiguousarray(full[:,3],dtype=np.float32)
        self.full_counts = full[:,4].astype(np.uint64)
        self.full_heights = full[:,5:7].copy()
        self.step_graph = sparse.csr_matrix((np.ones(edge_count,np.uint8),neighbors,offsets),shape=(node_count,node_count))
        self.components,self.component_labels=connected_components(self.step_graph,directed=False)
        self.anchor_tree = cKDTree(self.node_xyz) if node_count else None
        tick = time.perf_counter()
        self.search = NativeAStar(self.build_folder)
        self.setup = dict(aggregate_s=float(timing[0]),support_prepare_s=float(timing[1]),
            graph_prepare_s=float(timing[2]),native_build_s=float(timing[3]),
            native_compile_s=self.compile_s+self.search.compile_s,
            native_compile_cache_hit=self.compile_cache_hit and self.search.compile_cache_hit,
            search_prepare_s=time.perf_counter()-tick,total_prepare_s=time.perf_counter()-started)
        names = ['input_points','invalid_points','retained_low_cost_points','full_voxels',
            'height_patches','rejected_height_match','reserved_6','reserved_7','reserved_8',
            'rejected_layer_pairing','rejected_corner','reserved_11','surface_nodes','directed_edges','candidate_patches']
        self.report = dict(zip(names,map(int,counts)))
        self.report={k:v for k,v in self.report.items() if not k.startswith('reserved_')}
        self.report.update(backend=self.backend,options=asdict(o),gravity_up=list(c.gravity_up),
            connected_components=int(self.components),
            cost_limit=float(c.cost_threshold),cost_gate=f'0 <= raw cost < {c.cost_threshold:g}',gap_policy='disconnect',dense_field_cells=0,
            traversability_policy='cost_only; no geometric slope, step, normal, clearance or body collision filters',
            layer_adjacency='reciprocal best (cost, height difference) in adjacent XY columns, abs(dh) <= slice_dh; no within-column vertical edges',
            cost_aggregation=f'raw cost < {c.cost_threshold:g} mean for graph; full cost voxel mean for GPMP',
            occupancy_meaning='observed cost cells only; missing XY columns are disconnected')
        if not node_count:
            raise ValueError('No valid observed support surfaces remain: '+str(self.report))

    @property
    def optimizer(self):
        if self._optimizer is None:
            tick=time.perf_counter()
            self._optimizer=GravityGPMP(self.full_xyz,self.full_cost,self.config)
            self.setup['gpmp_query_prepare_s']=time.perf_counter()-tick
        return self._optimizer

    def match_best_layer(self,x,y,target_height):
        """Gravity-frame XY: min (cost, height difference) inside slice_dh, or None."""
        if self.handle is None or not np.isfinite([x,y,target_height]).all():return None
        layer=int(self.lib.surface_match_layer(self.handle,float(x),float(y),float(target_height)))
        return None if layer<0 else layer

    def match_endpoint(self,position,max_distance):
        """Prefer its own column; bridge only the endpoint's bounded scan blind spot.

        This adds no graph edge and never relaxes the height window.
        """
        local=np.asarray(position)@self.frame
        selected=self.match_best_layer(*local)
        if selected is not None and np.linalg.norm(self.node_xyz[selected]-position)<=max_distance:
            return selected,False
        candidates=self.anchor_tree.query_ball_point(position,max_distance)
        best=None;best_key=(float('inf'),)*3;visited=set()
        for candidate in candidates:
            cell=tuple(self.detail[candidate,:2].astype(np.int64))
            if cell in visited:continue
            visited.add(cell)
            xy=self.node_xyz[candidate]@self.frame
            match=self.match_best_layer(xy[0],xy[1],local[2])
            if match is None:continue
            distance=float(np.linalg.norm(self.node_xyz[match]-position))
            if distance>max_distance:continue
            key=(distance,float(self.node_cost[match]),float(abs((self.node_xyz[match]@self.frame)[2]-local[2])))
            if key<best_key:best,best_key=match,key
        return best, best is not None

    def astar(self,start,goal,step_edges=None,snap_limits=(1.,.5)):
        query=np.asarray([start,goal],dtype=float)
        if query.shape!=(2,3) or not np.isfinite(query).all():raise ValueError('Invalid endpoint coordinates')
        if len(snap_limits)!=2 or not np.isfinite(snap_limits).all() or min(snap_limits)<=0:
            raise ValueError('Invalid endpoint snap limits')
        tick=time.perf_counter()
        matches=[self.match_endpoint(p,limit) for p,limit in zip(query,snap_limits)]
        selected=[m[0] for m in matches]
        matched=all(i is not None for i in selected)
        nodes=np.asarray(selected,dtype=np.int64) if matched else None
        anchors=[None if i is None else self.node_xyz[i].tolist() for i in selected]
        distances=[None if i is None else float(np.linalg.norm(self.node_xyz[i]-q)) for i,q in zip(selected,query)]
        snap=dict(query_xyz=query.tolist(),pcd_anchor_xyz=anchors,node_ids=selected,
            source_voxel_indices=selected,surface_vertex_ids=selected,
            query_to_anchor_m=distances,anchor_to_dc_m=[0. if i is not None else None for i in selected],query_to_dc_m=distances,
            anchor_kind='height-gated local layer; bounded endpoint blind-spot fallback',
            matched=matched,slice_dh=self.options.slice_dh,endpoint_snap_limits_m=list(snap_limits),
            endpoint_blind_spot_fallback=[m[1] for m in matches])
        snap_s=time.perf_counter()-tick
        if matched:
            metrics,path=self.search.search(self.step_graph,self.node_xyz,self.node_cost,nodes,self.config.cost_weight)
        else:
            metrics=dict(found=False,search_s=0.,call_wall_s=0.,expanded_nodes=0,examined_edges=0,path_points=0)
            path=np.empty(0,np.uint32)
        metrics.update(endpoint_snap_s=snap_s,backend=self.backend)
        metrics['endpoint_components']=[None if i is None else int(self.component_labels[i]) for i in selected]
        rejected=[]
        for position in query:
            local=position@self.frame;distance2=np.sum((self.patches[:,:3]-local)**2,axis=1)
            k=min(3,len(distance2));indices=np.argpartition(distance2,k-1)[:k]
            indices=indices[np.argsort(distance2[indices])]
            rejected.append([dict(xyz=(self.patches[i,:3]@self.frame.T).tolist(),distance_m=float(np.sqrt(distance2[i])),
                cost=float(self.patches[i,3]),state='cost_eligible') for i in indices])
        snap['nearest_observed_patches']=rejected
        result=dict(success=metrics['found'],metrics=metrics,endpoint_snap=snap,node_path=path)
        if not matched:
            result['error']='No height-matched cost-eligible layer in '+', '.join(
                name for name,i in zip(('start XY cell','goal XY cell'),selected) if i is None)
        if metrics['found']:
            xyz=self.node_xyz[path];cost=self.node_cost[path];length=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
            integral=float(np.sum(length*(cost[:-1]+cost[1:])/2))
            metrics.update(integral_cost_m=integral,mean_cost=integral/max(metrics['length_m'],1e-12),
                max_cost=float(cost.max()),step_edges=self.config.step_edges,
                max_edge_height_m=float(np.max(np.abs(np.diff(xyz,axis=0)@self.frame[:,2]))) if len(xyz)>1 else 0.,
                slice_dh=self.options.slice_dh)
            result.update(support_xyz=xyz.copy(),xyz=xyz.copy(),cost=cost)
        return result

    def validate_trajectory(self,support,check_cost=True):
        support=np.asarray(support,dtype=float)
        if support.ndim!=2 or support.shape[1]!=3 or not len(support) or not np.isfinite(support).all():
            return dict(valid=False,reason='invalid_trajectory')
        tick=time.perf_counter();parts=[]
        for a,b in zip(support[:-1],support[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/self.options.validation_spacing)))
            parts.append(a+(b-a)*np.arange(count)[:,None]/count)
        dense=np.vstack([*parts,support[-1:]])
        local=np.ascontiguousarray(dense@self.frame,dtype=np.float64)
        detail=np.zeros(2,np.uint64)
        valid=bool(self.lib.surface_validate(self.handle,local,len(local),0.,detail))
        reasons={0:'ok',1:'missing_cost_cell',2:'height_mismatch',3:'disconnected_layer',5:'nonfinite'}
        result=dict(valid=valid,reason=reasons[int(detail[1])],checked_points=len(local),wall_s=time.perf_counter()-tick)
        if not valid:result.update(first_bad_sample=int(detail[0]),position=dense[int(detail[0])].tolist())
        elif check_cost:
            values,_=self.optimizer.query_cost(dense)
            result['max_cost']=float(values.max())
            bad=np.flatnonzero(~np.isfinite(values)|(values>=self.config.cost_threshold))
            if len(bad):
                i=int(bad[0]);result.update(valid=False,reason='high_cost',first_bad_sample=i,position=dense[i].tolist())
        result['wall_s']=time.perf_counter()-tick
        return result
