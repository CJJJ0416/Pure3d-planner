"""Cost-cloud surface queries shared by online planning and comparisons."""
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from Entry.utils.io import atomic_json, atomic_npy


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
    def __init__(self,directory,cost_limit=.5):
        from planner import PlannerConfig
        from planner.surface import SparseSurfacePlanner
        self.planner=SparseSurfacePlanner(PlannerConfig(cost_threshold=cost_limit,dc_input_cost_max=cost_limit),directory/'native')
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
                    graph_nodes=len(p.node_xyz),graph_edges=int(p.step_graph.nnz),
                    graph_prepare_s=prepare_s,graph_cache_hit=not prepared,
                    compile_s=p.compile_s,compile_cache_hit=p.compile_cache_hit,
                    search_repeats=repeats,astar_length=None if body is None else float(np.linalg.norm(np.diff(body,axis=0),axis=1).sum()),
                    start_source=request.get('start_source','/Odometry_gazebo base at goal reception'),goal_source='/scan/goal',
                    base_transform_stamp_ns=request.get('base_transform_stamp_ns'),
                    planner_source='planner.surface.SparseSurfacePlanner + planner.graph.astar.NativeAStar',
                    direct_cost_cloud_input=True,depends_on_scan_path=False,
                    astar_cost=None if body is None or online else score(body),
                    endpoint_snapping_is_not_a_verified_connection=True,
                    elapsed_before_report_s=time.perf_counter()-started,
                    gpmp_used=False,actual_astar_tracking=False,scan_uses_predicted_cost=False)
        atomic_json(folder/'comparison.json',report)
        return report,body
