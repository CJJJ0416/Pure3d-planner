"""PCD → DC表面 → 跨阶连边 → C++单向A* → 反重力GPMP。"""
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from common.pointcloud_io import fields_xyz,read_pcd_fields
from .geometry.voxel import voxel_mean,observed_voxel_field,support_clearance
from .geometry.dual_contouring import dual_contouring,outward_vertex_normals
from .graph.adjacency import surface_graph,connect_step_surfaces
from .graph.astar import NativeAStar
from .optimization.gpmp import GravityGPMP


class DCSurfacePlanner:
    """一张地图构建一次，正式方案和无跨阶边消融共用全部节点与代价。"""
    def __init__(self,config,build_folder=None):
        self.config=config
        self.build_folder=Path(build_folder) if build_folder is not None else Path(__file__).resolve().parents[1]/'datasets/cache/planner_dc_native'

    def load_pcd(self,path):
        began=time.perf_counter();self.source_pcd=str(Path(path).resolve());cfg=self.config;self.setup={}
        started=time.perf_counter();raw=read_pcd_fields(path);coord=fields_xyz(raw)
        self.input_points=len(raw);self.setup['pcd_read_s']=time.perf_counter()-started
        started=time.perf_counter();voxel=voxel_mean(coord,raw['trav_cost'],cfg.voxel_resolution)
        keep=np.ones(len(voxel['xyz']),dtype=bool)
        if cfg.roi_min is not None:
            keep=((voxel['xyz']>=cfg.roi_min)&(voxel['xyz']<=cfg.roi_max)).all(1)
        self.xyz=voxel['xyz'][keep];self.cost=voxel['cost'][keep]
        self.original_point_count=int(voxel['counts'][keep].sum())
        self.setup['voxel_prepare_s']=time.perf_counter()-started
        field,origin,self.field_info=observed_voxel_field(self.xyz,cfg.voxel_resolution)
        self.setup['field_prepare_s']=self.field_info['field_prepare_s']
        started=time.perf_counter();self.surface,self.triangles,self.dc_info=dual_contouring(field,origin,cfg.voxel_resolution)
        self.setup['dc_extract_s']=time.perf_counter()-started
        started=time.perf_counter();self.normals,self.normal_info=outward_vertex_normals(self.surface,self.triangles,field,origin,cfg.voxel_resolution)
        self.setup['normals_s']=time.perf_counter()-started
        started=time.perf_counter();_,nearest=cKDTree(self.xyz).query(self.surface,workers=-1)
        # 与cost映射共用来源编号，端点只吸附到该PCD体素关联的表面候选。
        self.surface_source_voxel=nearest
        self.surface_cost=self.cost[nearest].astype(float)
        self.search_surface=self.surface+cfg.search_normal_offset*self.normals
        self.setup['surface_cost_transfer_s']=time.perf_counter()-started
        started=time.perf_counter()
        self.source_clear,self.source_overhead_distance=support_clearance(
            self.xyz,field,origin,cfg.voxel_resolution,cfg.gravity_up,cfg.interval_free)
        self.surface_step_support=self.source_clear[self.surface_source_voxel]
        self.setup['support_clearance_s']=time.perf_counter()-started
        # 原三角网格及cost不变，来源净空仅限制新增跨阶边。
        started=time.perf_counter()
        nodes,c,_,_,ids,base=surface_graph(self.search_surface,self.triangles,self.surface_cost,cfg.cost_threshold)
        self.node_xyz=nodes;self.node_cost=c;self.node_indices=ids
        self.base_graph=base.astype(bool).astype(np.int8)
        linked,self.link_info=connect_step_surfaces(self.surface,self.normals,ids,self.base_graph,
            cfg.step_max,cfg.slope_max,cfg.step_horizontal_radius,cfg.gravity_up,cfg.minimum_step_height,
            support_mask=self.surface_step_support)
        self.link_info['source_interval_free_m']=cfg.interval_free
        self.step_graph=linked.astype(bool).astype(np.int8)
        self.setup['cost_and_graph_s']=time.perf_counter()-started
        started=time.perf_counter();self._build_endpoint_index()
        self.setup['endpoint_index_s']=time.perf_counter()-started
        self.optimizer=GravityGPMP(self.xyz,self.cost,cfg)
        self.search=NativeAStar(self.build_folder);self.setup['native_compile_s']=self.search.compile_s
        self.setup['total_prepare_s']=time.perf_counter()-began

    def _build_endpoint_index(self):
        """每个来源体素固定一个最近DC候选，距离相同时取原顶点编号最小者。"""
        cfg=self.config;ids=self.node_indices
        self.up=np.asarray(cfg.gravity_up,dtype=float);self.up/=np.linalg.norm(self.up)
        self.endpoint_nodes=np.flatnonzero(self.normals[ids]@self.up>=np.cos(cfg.slope_max))
        vertices=ids[self.endpoint_nodes];owners=self.surface_source_voxel[vertices]
        delta=self.surface[vertices]-self.xyz[owners]
        distance2=np.einsum('ij,ij->i',delta,delta)
        order=np.lexsort((vertices,distance2,owners))
        self.anchor_voxel_indices,first=np.unique(owners[order],return_index=True)
        self.anchor_node_indices=self.endpoint_nodes[order[first]]
        self.anchor_tree=cKDTree(self.xyz[self.anchor_voxel_indices])

    def anchor(self,points):
        """查询XYZ表示PCD支撑位置，返回同形状的输入体素锚点XYZ。"""
        _,index=self.anchor_tree.query(np.asarray(points,dtype=float))
        return self.xyz[self.anchor_voxel_indices[index]]

    def _snap(self,points):
        query=np.asarray(points,dtype=float)
        _,index=self.anchor_tree.query(query)
        voxels=self.anchor_voxel_indices[index];nodes=self.anchor_node_indices[index]
        support=self.surface[self.node_indices[nodes]]
        return np.linalg.norm(support-query,axis=-1),nodes,support,voxels,self.xyz[voxels]

    def snap(self,points):
        """PCD锚点→对应DC表面，保持(distance,node,support_xyz)返回格式。"""
        distance,nodes,support,_,_=self._snap(points)
        return distance,nodes,support

    def astar(self,start,goal,step_edges=None):
        enabled=self.config.step_edges if step_edges is None else step_edges
        began=time.perf_counter()
        query=np.asarray([start,goal],dtype=float)
        snap,endpoints,support_endpoints,voxels,anchors=self._snap(query)
        snap_s=time.perf_counter()-began
        endpoint_snap=dict(query_xyz=query.tolist(),pcd_anchor_xyz=anchors.tolist(),
            source_voxel_indices=voxels.tolist(),node_ids=endpoints.tolist(),
            surface_vertex_ids=self.node_indices[endpoints].tolist(),dc_support_xyz=support_endpoints.tolist(),
            query_to_anchor_m=np.linalg.norm(anchors-query,axis=1).tolist(),
            anchor_to_dc_m=np.linalg.norm(support_endpoints-anchors,axis=1).tolist(),
            query_to_dc_m=snap.tolist())
        graph=self.step_graph if enabled else self.base_graph
        metrics,path=self.search.search(graph,self.node_xyz,self.node_cost,endpoints,self.config.cost_weight)
        metrics.update(start_snap_m=float(snap[0]),goal_snap_m=float(snap[1]),step_edges=enabled,
            endpoint_snap_s=snap_s)
        result=dict(success=metrics['found'],metrics=metrics,endpoint_nodes=endpoints,
            surface_endpoints=support_endpoints,node_path=path,endpoint_snap=endpoint_snap)
        if metrics['found']:
            ids=self.node_indices[path];xyz=self.node_xyz[path];c=self.node_cost[path]
            length=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
            integral=float(np.sum(length*(c[:-1]+c[1:])/2))
            metrics.update(integral_cost_m=integral,mean_cost=integral/metrics['length_m'],max_cost=float(c.max()))
            edges=np.column_stack((path[:-1],path[1:]))
            metrics['new_edges_used']=int((np.asarray(self.base_graph[edges[:,0],edges[:,1]]).ravel()==0).sum())
            result.update(xyz=xyz,cost=c,point_indices=ids,support_xyz=self.surface[ids])
        return result

    def plan(self,start,goal,step_edges=None):
        began=time.perf_counter();astar=self.astar(start,goal,step_edges)
        result=dict(success=astar['success'],astar=astar)
        if astar['success']:
            support_path=astar['support_xyz'].copy()
            # 查询约定为输入PCD支撑位置，机体端点从PCD锚点沿反重力偏移。
            # 内部DC节点和A*结果保持原样，避免把壳面偏移叠到端点高度上。
            support_path[[0,-1]]=astar['endpoint_snap']['pcd_anchor_xyz']
            result['gpmp']=self.optimizer.optimize(support_path)
            result['gpmp']['metrics']['endpoint_support_reference']='input PCD voxel anchors'
        result['planning_wall_s']=time.perf_counter()-began
        return result

    def graph_data(self):
        """供RViz消融显示使用，不保存密集几何场或重复CSR缓存。"""
        base=sparse.triu(self.base_graph,k=1).tocoo()
        delta=(sparse.triu(self.step_graph,k=1)-sparse.triu(self.base_graph,k=1)).tocsr();delta.eliminate_zeros();delta=delta.tocoo()
        n0,l0=connected_components(self.base_graph,directed=False);n1,l1=connected_components(self.step_graph,directed=False)
        return dict(xyz=self.node_xyz,cost=self.node_cost,original_vertex_ids=self.node_indices,
            base_edges=np.column_stack((base.row,base.col)),added_edges=np.column_stack((delta.row,delta.col)),
            labels_before=l0,labels_after=l1,component_counts=np.array([n0,n1]))
