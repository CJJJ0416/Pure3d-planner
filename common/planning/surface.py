"""Cost-cloud surface queries shared by online planning and comparisons."""
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from common.io import atomic_json, atomic_npy


def curve_samples(control, knots, degree, step=.03):
    control=np.asarray(control,dtype=float);knots=np.asarray(knots,dtype=float)
    if control.ndim!=2 or control.shape[1]!=3 or not np.isfinite(control).all():
        raise ValueError('Invalid SCAN control points')
    if len(knots)!=len(control)+degree+1 or not np.isfinite(knots).all() or np.any(np.diff(knots)<0):
        raise ValueError('Invalid SCAN knots')
    lo,hi=knots[degree],knots[len(control)]
    if hi<=lo:raise ValueError('SCAN trajectory has no duration')
    t=np.linspace(lo,hi,max(3,int(np.ceil((hi-lo)/step))+1))
    return BSpline(knots,control,degree,extrapolate=False)(t),float(hi-lo)


class SurfaceQueries:
    def __init__(self,directory,cost_limit=.5,surface_options=None,step_max=None):
        from planner.light_planner import PlannerConfig
        from planner.light_planner.surface import SparseSurfacePlanner
        config=PlannerConfig(cost_threshold=cost_limit,dc_input_cost_max=cost_limit,step_edges=False)
        if step_max is not None:config.step_max=step_max
        self.planner=SparseSurfacePlanner(config,directory/'native',options=surface_options)
        self.key=None

    def compute(self,cloud,request,ground_height,query_id,directory,*,online=False):
        started=time.perf_counter();p=self.planner
        prepared=self.key!=cloud['key'];prepare_s=0.
        if prepared:
            # Loading can replace graph state before rejecting an empty graph.
            # Invalidate first so a later return to the old cloud rebuilds it.
            self.key=None
            tick=time.perf_counter();p.load_arrays(cloud['xyz'],cloud['cost']);prepare_s=time.perf_counter()-tick
            self.key=cloud['key']
        endpoints=request['body_endpoints'].copy()
        height=float(endpoints[0,2]-ground_height)
        support=endpoints-[0,0,height]
        repeats=[];result=None
        for _ in range(1 if online else 3):
            tick=time.perf_counter();result=p.astar(*support,**({} if online else dict(snap_limits=(.2,.2))))
            repeats.append(dict(wall_s=time.perf_counter()-tick,**result['metrics']))
        body=None
        if result['success']:
            body=result['support_xyz']+[0,0,height]
        folder=directory/'queries'/f'{query_id:06d}';folder.mkdir(parents=True)
        if prepared and not online:
            atomic_npy(folder/'input_xyz.npy',cloud['xyz']);atomic_npy(folder/'input_cost.npy',cloud['cost'])
        if body is not None:atomic_npy(folder/'astar_body.npy',body)
        def score(path):
            # All raw cost voxels, not just eligible A* nodes. Reject distant
            # or mismatched-height queries, avoiding extrapolation across gaps.
            q=path-[0,0,height];tree=cKDTree(p.full_xyz[:,:2]);groups=tree.query_ball_point(q[:,:2],.2)
            values=[]
            for point,ids in zip(q,groups):
                ids=np.asarray(ids,dtype=int);ids=ids[np.abs(p.full_xyz[ids,2]-point[2])<=.12]
                if not len(ids):continue
                d=np.linalg.norm(p.full_xyz[ids]-point,axis=1);w=np.exp(-.5*(d/.1)**2)
                values.append(float(w@p.full_cost[ids]/w.sum()))
            return dict(samples=len(path),covered=len(values),coverage=float(len(values)/len(path)),
                        mean=None if not values else float(np.mean(values)),max=None if not values else float(max(values)))
        distances=result['endpoint_snap']['query_to_anchor_m']
        report=dict(query_id=query_id,success=bool(result['success']),error=result.get('error'),
                    cost_limit=p.config.cost_threshold,cost_gate=p.report['cost_gate'],
                    manual_goal_id=request['id'],cloud_stamp=cloud['stamp'],cloud_key=cloud['key'],
                    cloud_points=len(cloud['xyz']),frame='world',ground_height=ground_height,body_height=height,
                    requested_body_endpoints=endpoints.tolist(),endpoint_snap=result['endpoint_snap'],
                    resolved_goal_body=None if body is None else body[-1].tolist(),
                    graph_nodes=len(p.node_xyz),graph_edges=int(p.step_graph.nnz),
                    graph_prepare_s=prepare_s,graph_cache_hit=not prepared,
                    compile_s=p.compile_s,compile_cache_hit=p.compile_cache_hit,
                    search_repeats=repeats,astar_length=None if body is None else float(np.linalg.norm(np.diff(body,axis=0),axis=1).sum()),
                    start_source=request.get('start_source','/Odometry_gazebo base at goal reception'),goal_source='/scan/goal',
                    base_transform_stamp_ns=request.get('base_transform_stamp_ns'),
                    planner_source='planner.light_planner.surface.SparseSurfacePlanner + planner.graph.astar.NativeAStar',
                    direct_cost_cloud_input=True,depends_on_scan_path=False,
                    astar_cost=None if body is None or online else score(body),
                    endpoint_snapping_is_not_a_verified_connection=True,
                    elapsed_before_report_s=time.perf_counter()-started,
                    gpmp_used=False,actual_astar_tracking=False,scan_uses_predicted_cost=False)
        atomic_json(folder/'comparison.json',report)
        return report,body


class LocalCandidateQueries(SurfaceQueries):
    """The offline-tested one-tree selector, using the live cost snapshot directly."""
    def __init__(self,directory,cost_limit=.8,options=None,step_max=.2):
        from planner.light_planner.surface import SurfaceOptions
        super().__init__(directory,cost_limit,SurfaceOptions(step_connections=False),step_max)
        from planner.light_planner.graph.candidates import CandidateOptions
        self.options=options or CandidateOptions()
        self.candidate_search=None

    def compute(self,cloud,request,ground_height,query_id,directory,*,online=True):
        from planner.light_planner.graph.candidates import LocalCandidateSearch
        from dataclasses import replace
        from common.planning.rolling_reference import join_reference, sample_reference, FrozenPrefixInvalid
        started=time.perf_counter();p=self.planner
        prefix=request.get('frozen_prefix_body')
        prepared=self.key!=cloud['key'];prepare_s=weighted_s=0.
        if prepared:
            self.key=None;self.candidate_search=None
            tick=time.perf_counter()
            try:p.load_arrays(cloud['xyz'],cloud['cost'])
            except ValueError as error:
                if prefix is not None:raise FrozenPrefixInvalid(str(error)) from error
                raise
            prepare_s=time.perf_counter()-tick
            self.candidate_search=LocalCandidateSearch(p.step_graph,p.node_xyz,p.node_cost,
                astar_cost_weight=p.config.cost_weight,cost_limit=p.config.cost_threshold,
                gravity_up=p.config.gravity_up)
            weighted_s=self.candidate_search.prepare_s;self.key=cloud['key']
        endpoints=np.asarray(request['body_endpoints']).copy()
        height=float(request.get('body_reference_height_m',endpoints[0,2]-ground_height))
        support=endpoints-[0,0,height]
        goal_node,goal_fallback=p.match_endpoint(support[1],.5)
        resolved_goal=support[1] if goal_node is None else p.node_xyz[goal_node]
        if prefix is not None:
            prefix_validation=p.validate_trajectory(np.asarray(prefix)-[0,0,height],check_cost=False)
            if not prefix_validation['valid']:raise FrozenPrefixInvalid(str(prefix_validation))
        search_start=support[0] if prefix is None else np.asarray(prefix)[-1]-[0,0,height]
        options=self.options
        if prefix is not None:
            kept=float(np.linalg.norm(np.diff(prefix,axis=0),axis=1).sum())
            horizon=max(.3,self.options.horizon_m-kept)
            options=replace(options,horizon_m=horizon,minimum_length_m=min(options.minimum_length_m,horizon))
        tick=time.perf_counter();start,fallback=p.match_endpoint(search_start,1. if prefix is None else .2);snap_s=time.perf_counter()-tick
        result=None;body=None;candidates=[]
        occupancy=request.get('body_occupancy')
        blocked=None
        if start is not None:
            if occupancy is not None:
                blocked=occupancy.blocked_edges(self.candidate_search.graph,p.node_xyz,height)
            result=self.candidate_search.query(start,resolved_goal,options,blocked_edges=blocked)
            for item in result['candidates']:
                candidate={k:v for k,v in item.items() if k!='node_path'}
                tail=p.node_xyz[item['node_path']]+[0,0,height]
                if prefix is not None:
                    try:
                        path,gap=join_reference(prefix,tail)
                        validation=p.validate_trajectory(path-[0,0,height],check_cost=False)
                        if not validation['valid']:continue
                        candidate.update(connector_gap_m=gap,reference_validation=validation,
                                         tail_length_m=candidate['length_m'],length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))
                    except ValueError:continue
                else:path=sample_reference(tail) if len(tail)>1 else tail
                if request.get('raw_support_candidates'):
                    # Preserve original graph edges for the terrain speed profile.
                    raw=tail if prefix is None else np.vstack((prefix,tail))
                    candidate['support_xyz']=(raw-[0,0,height]).tolist()
                candidate['candidate_id']=len(candidates);candidate['body_xyz']=path.tolist()
                candidates.append(candidate)
            if candidates:body=np.asarray(candidates[0]['body_xyz'])
        metrics=dict(found=body is not None,call_wall_s=0.,graph_search_calls=0) if result is None else dict(
            found=result['success'],call_wall_s=result['metrics']['total_query_s'],**result['metrics'])
        metrics.update(found=body is not None,candidates_after_splice=len(candidates))
        report=dict(query_id=query_id,manual_goal_id=request['id'],backend='light',
            success=body is not None,error=None if body is not None else 'No reachable local candidate or start support',
            reason='start_snap_failed' if result is None else ('replacement_tail_invalid' if not candidates else result['reason']),
            reference_update='initial' if prefix is None else 'tail',frozen_prefix_body=prefix,
            rolling=request.get('rolling'),cost_score_scope='whole_initial_path' if prefix is None else 'replacement_tail_with_common_fixed_prefix',
            cloud_key=cloud['key'],cloud_stamp=cloud['stamp'],cloud_points=len(cloud['xyz']),frame='world',
            ground_height=ground_height,body_height=height,
            requested_body_endpoints=endpoints.tolist(),
            resolved_goal_body=None if goal_node is None else (resolved_goal+[0,0,height]).tolist(),
            goal_snap=dict(node=goal_node,fallback=goal_fallback,
                           distance_m=None if goal_node is None else float(np.linalg.norm(resolved_goal-support[1]))),
            endpoint_snap=dict(start_node=start,start_fallback=fallback,start_snap_s=snap_s,
                start_snap_distance_m=None if start is None else float(np.linalg.norm(p.node_xyz[start]-search_start)),
                goal_requires_local_snap=False),
            cost_limit=p.config.cost_threshold,cost_gate=p.report['cost_gate'],
            graph_nodes=len(p.node_xyz),graph_edges=int(p.step_graph.nnz),graph_prepare_s=prepare_s,
            occupancy_masked_edges=None if blocked is None else int(blocked.sum()),
            graph_cache_hit=not prepared,weighted_graph_prepare_s=weighted_s,
            search_repeats=[metrics],candidate_metrics=metrics,candidates=candidates,
            candidate_options=vars(options),selected_id=0 if body is not None else None,
            astar_length=None if body is None else candidates[0]['length_m'],
            start_source=request.get('start_source'),base_transform_stamp_ns=request.get('base_transform_stamp_ns'),
            goal_source='/scan/goal',planner_source='LocalCandidateSearch: one Dijkstra + path NMS + analytic score',
            direct_cost_cloud_input=True,depends_on_scan_path=False,gpmp_used=False,
            actual_astar_tracking=False,scan_uses_predicted_cost=False,
            endpoint_snapping_is_not_a_verified_connection=True,elapsed_before_report_s=time.perf_counter()-started)
        folder=directory/'queries'/f'{query_id:06d}';folder.mkdir(parents=True)
        if body is not None:atomic_npy(folder/'astar_body.npy',body)
        atomic_json(folder/'comparison.json',report)
        return report,body
