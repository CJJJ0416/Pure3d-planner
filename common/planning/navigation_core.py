"""Selectable observed-surface/DC A*, GPMP and current-base SCAN paths."""
from dataclasses import asdict, replace
from pathlib import Path
import time
import numpy as np
from common.io import atomic_json, atomic_npy
from common.pointcloud_io import write_pcd


def gpmp_support_points(points):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid GPMP support path')
    if len(points)>1:points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]]
    if len(points)<2:raise ValueError('GPMP requires distinct endpoints')
    if len(points)==2:points=np.vstack((points[0],points.mean(0),points[1]))
    return points


class NavigationPlanner:
    def __init__(self,config,directory,body_height=.4,max_snap=.5,scan_path_source='gpmp',backend='dc',surface_options=None):
        if scan_path_source not in ('astar','gpmp'):raise ValueError('Expected astar or gpmp path source')
        if backend not in ('dc','surface'):raise ValueError('Expected dc or surface backend')
        self.config=replace(config,endpoint_offset_m=body_height)
        self.directory=Path(directory);self.body_height=body_height;self.max_snap=max_snap;self.scan_path_source=scan_path_source
        self.backend=backend;self.surface_options=surface_options
        self.planner=None;self.map_key=None;self.prediction_id=None;self.progress=dict(stage='waiting_cost')

    def compute(self,xyz,cost,map_key,start_body,goal_body,query_id,prediction_id=None):
        started=time.perf_counter();prepared=False
        map_folder=self.directory/'maps'/(str(map_key) if prediction_id is None else f'{map_key}_p{prediction_id}')
        if self.planner is None or self.map_key!=map_key or self.prediction_id!=prediction_id:
            self.progress=dict(stage='building_'+self.backend+'_graph',map_key=map_key,points=len(xyz))
            folder=map_folder;folder.mkdir(parents=True,exist_ok=True)
            if self.backend=='surface':
                from planner.light_planner.surface import SparseSurfacePlanner
                planner=SparseSurfacePlanner(self.config,self.directory/'native',self.surface_options)
            else:
                from planner.light_planner import DCSurfacePlanner
                planner=DCSurfacePlanner(self.config,self.directory/'native')
            try:
                if self.backend=='surface':planner.load_arrays(xyz,cost)
                else:
                    write_pcd(folder/'input_world.pcd',xyz,cost)
                    planner.load_pcd(folder/'input_world.pcd')
            except Exception as error:
                self.progress=dict(stage='failed',map_key=map_key,error=str(error))
                atomic_json(folder/'map_failure.json',dict(self.progress,report=getattr(planner,'report',{})))
                raise
            if self.planner is not None and hasattr(self.planner,'close'):self.planner.close()
            self.planner=planner;self.map_key=map_key;self.prediction_id=prediction_id;prepared=True
            self.map_record=dict(backend=self.backend,frame_id='world',input_points=len(xyz),
                prediction_id=prediction_id,
                voxel_points=len(planner.xyz),search_nodes=len(planner.node_xyz),setup=planner.setup,config=asdict(self.config))
            if self.backend=='surface':
                self.map_record['surface']=planner.report
                for name,array in [('surface_xyz',planner.node_xyz),('surface_cost',planner.node_cost),
                    ('surface_detail',planner.detail),('full_voxel_xyz',planner.full_xyz),('full_voxel_cost',planner.full_cost)]:
                    atomic_npy(folder/(name+'.npy'),array)
            else:self.map_record.update(dc_input_filter=planner.prefilter_info,geometry_field=planner.field_info)
            atomic_json(folder/'map.json',self.map_record)
        planner=self.planner;self.progress=dict(stage='astar',map_key=map_key)
        up=np.asarray(self.config.gravity_up,dtype=float);up/=np.linalg.norm(up)
        support_query=np.array([start_body,goal_body])-self.body_height*up
        limits=np.array([max(1.,self.max_snap),self.max_snap])
        result=planner.astar(*support_query,snap_limits=limits) if self.backend=='surface' else planner.astar(*support_query)
        folder=self.directory/'queries'/f'{query_id:06d}';folder.mkdir(parents=True,exist_ok=True)
        snap=result['endpoint_snap'];error=None
        distances=snap['query_to_anchor_m']
        if all(x is not None for x in distances) and np.any(np.asarray(distances)>limits):
            error=f'Endpoint exceeds PCD anchor limits {limits.tolist()} m (start, goal)'
        if not result['success']:error=result.get('error',f'{self.backend} A* found no route in observed traversable graph')
        support=body=None
        artifacts={};gpmp_metrics=None;validation={}
        if error is None:
            support=result['support_xyz'].copy();support[[0,-1]]=snap['pcd_anchor_xyz']
            if self.backend=='surface':
                validation['astar']=planner.validate_trajectory(support,check_cost=False)
                if not validation['astar']['valid']:error='A* support validation failed: '+validation['astar']['reason']
        if error is None:
            astar_body=support+self.body_height*up
            atomic_npy(folder/'astar_world.npy',astar_body);atomic_npy(folder/'support_world.npy',support)
            artifacts['astar_world']=str(folder/'astar_world.npy')
            if self.scan_path_source=='astar':
                body=astar_body
            else:
                self.progress=dict(stage='gpmp',map_key=map_key,query_id=query_id)
                try:
                    tick=time.perf_counter();optimizer=planner.optimizer;cost_field_s=time.perf_counter()-tick
                    gpmp=optimizer.optimize(gpmp_support_points(support));gpmp_metrics=dict(gpmp['metrics'],cost_field_prepare_s=cost_field_s)
                    gpmp_body=np.asarray(gpmp['xyz'])
                    if gpmp_body.ndim!=2 or gpmp_body.shape[1]!=3 or len(gpmp_body)<2 or not np.isfinite(gpmp_body).all():
                        raise ValueError('GPMP returned an invalid trajectory')
                    np.testing.assert_allclose(gpmp_body,np.asarray(gpmp['support_xyz'])+self.body_height*up,atol=1e-7)
                    np.testing.assert_allclose(gpmp_body[[0,-1]],astar_body[[0,-1]],atol=1e-6)
                    if self.backend=='surface':
                        validation['gpmp']=planner.validate_trajectory(gpmp['support_xyz'])
                        if not validation['gpmp']['valid']:
                            atomic_npy(folder/'rejected_gpmp_world.npy',gpmp_body)
                            raise ValueError('support validation: '+validation['gpmp']['reason'])
                    atomic_npy(folder/'gpmp_world.npy',gpmp_body)
                    atomic_npy(folder/'gpmp_support_world.npy',gpmp['support_xyz']);atomic_npy(folder/'gpmp_time.npy',gpmp['time'])
                    np.savetxt(folder/'gpmp_trajectory.csv',np.column_stack((gpmp['time'],gpmp_body,gpmp['velocity'],gpmp['yaw'],gpmp['cost'])),
                        delimiter=',',header='t,x,y,z,vx,vy,vz,yaw,trav_cost',comments='')
                    artifacts['gpmp_world']=str(folder/'gpmp_world.npy')
                    if self.scan_path_source=='gpmp' and not gpmp_metrics['converged']:error='GPMP did not converge; holding position'
                    else:body=gpmp_body if self.scan_path_source=='gpmp' else astar_body
                except Exception as failure:
                    error=f'GPMP failed: {failure}'
        record=dict(success=error is None,error=error,query_id=query_id,map_key=map_key,frame_id='world',
            start_body=np.asarray(start_body).tolist(),goal_body=np.asarray(goal_body).tolist(),
            endpoint_snap=snap,astar=result['metrics'],body_height_m=self.body_height,
            gpmp=gpmp_metrics,gpmp_skipped=self.scan_path_source=='astar',path_source=self.scan_path_source,artifacts=artifacts,
            backend=self.backend,validation=validation,map_cache_hit=not prepared,prediction_id=prediction_id,
            endpoint_anchor_limits_m=limits.tolist(),
            graph_rebuilt=prepared,wall_s=time.perf_counter()-started,
            method=self.backend+' observed support graph + C++ A*'+(' + gravity GPMP' if self.scan_path_source=='gpmp' else '')+'; SCAN local tracking')
        if error is None:
            atomic_npy(folder/'body_world.npy',body)
            record['actual_goal_body']=body[-1].tolist()
        atomic_json(folder/'query.json',record);self.progress=dict(stage='ready' if error is None else 'failed',query_id=query_id)
        atomic_json(map_folder/'map.json',self.map_record)
        return record,body


# Historical callers retain their DC default; online navigation selects explicitly.
DCNavigationPlanner=NavigationPlanner


class ArrivalHold:
    """One completion per goal; require measured proximity continuously."""
    def __init__(self,tolerance=.2,dwell=3.):self.tolerance=tolerance;self.dwell=dwell;self.reset()
    def reset(self):self.inside_since=None;self.complete=False
    def observe(self,position,goal,stamp):
        if self.complete:return False
        if np.linalg.norm(np.asarray(position)-goal)>self.tolerance:
            self.inside_since=None;return False
        if self.inside_since is None:self.inside_since=stamp
        if stamp-self.inside_since+1e-9<self.dwell:return False
        self.complete=True;return True
