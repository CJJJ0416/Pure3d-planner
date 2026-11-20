"""Current world<-base + /scan/goal + local raw cost -> planner -> /pure_path.

SCAN modes 3/4 remain responsible for collision-aware local B-splines and cmd_vel.
Full A* triggers per goal. Light candidates also renew near local segment ends.
SCAN owns continuous local trajectory replanning.
This node owns a steady-clock navigation-hold heartbeat while it is running.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from common.planning.surface import SurfaceQueries, LocalCandidateQueries, curve_samples
from common.io import atomic_json
from common.planning.rolling_reference import rolling_prefix, FrozenPrefixInvalid


def path_distances(points,polyline):
    points=np.asarray(points,dtype=float);route=np.asarray(polyline,dtype=float)
    if route.ndim!=2 or route.shape[1]!=3 or len(route)<2:raise ValueError('Expected a route with at least two XYZ points')
    delta=route[1:]-route[:-1];length2=np.einsum('ij,ij->i',delta,delta)
    length2=np.maximum(length2,1e-18)
    relative=points[:,None,:]-route[None,:-1,:]
    alpha=np.clip(np.einsum('nij,ij->ni',relative,delta)/length2,0.,1.)
    nearest=route[None,:-1,:]+alpha[:,:,None]*delta[None,:,:]
    return np.linalg.norm(points[:,None,:]-nearest,axis=2).min(axis=1)


def route_ack_matches(received_stamp,received_ends,expected_stamp,body_ends,height_offset):
    if expected_stamp is None or received_stamp!=expected_stamp:return False
    received=np.asarray(received_ends);expected=np.asarray(body_ends)-[0,0,height_offset]
    return received.shape==(2,3) and expected.shape==(2,3) and np.isfinite(received).all() and np.allclose(received,expected,rtol=0,atol=1e-5)


def light_refresh_due(route,body,goal,elapsed):
    """Handoff near a segment's end, never periodic path resets while following."""
    if route is None or len(route)<2 or elapsed<1. or np.linalg.norm(body-goal)<=.2:return False
    route=np.asarray(route);delta=np.diff(route,axis=0);length=np.linalg.norm(delta,axis=1)
    alpha=np.clip(np.einsum('ij,ij->i',body-route[:-1],delta)/np.maximum(length**2,1e-18),0,1)
    nearest=route[:-1]+alpha[:,None]*delta
    segment=int(np.argmin(np.linalg.norm(nearest-body,axis=1)))
    remaining=float(length[segment]*(1-alpha[segment])+length[segment+1:].sum())
    progress=float(length[:segment].sum()+length[segment]*alpha[segment])
    return progress>=min(.4,float(length.sum())/2) and remaining<=.8 and np.linalg.norm(body-route[-1])<=.9


def arrival_errors(body,requested,resolved=None,xy_tolerance=.3,z_tolerance=.2,settling=False):
    """Keep requested XY, but use the planning support layer for body height."""
    body=np.asarray(body);requested=np.asarray(requested)
    target=requested if resolved is None else np.asarray(resolved)
    xy=float(np.linalg.norm(body[:2]-requested[:2]));z=float(abs(body[2]-target[2]))
    entry=xy<=xy_tolerance and z<=z_tolerance
    latched=settling and xy<=xy_tolerance+.1 and z<=z_tolerance+.05
    return dict(inside=entry or latched,within_entry=entry,settling_latched=latched,xy_m=xy,z_m=z,
                raw_z_m=float(abs(body[2]-requested[2])),resolved=resolved is not None)


def main():
    import rclpy
    from rclpy.clock import Clock,ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from rclpy.time import Time
    from rcl_interfaces.srv import GetParameters
    from nav_msgs.msg import Path as PathMsg,Odometry
    from geometry_msgs.msg import PoseStamped,Point
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool,String,Header
    from std_srvs.srv import Trigger
    from visualization_msgs.msg import Marker
    from visualization_msgs.msg import MarkerArray
    from scan_planner_msgs.msg import Bspline
    from tf2_ros import Buffer,TransformListener,TransformException
    from Simulator.unitree_gazebo.scripts.frame_config import rotation_matrix

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bridge-body-height',type=float,default=.4)
    parser.add_argument('--cost-limit',type=float,default=.5)
    parser.add_argument('--no-goal-fallback',action='store_true')
    parser.add_argument('--backend',choices=['astar','light'],default='astar')
    parser.add_argument('--light-update-hz',type=float,default=2.)
    parser.add_argument('--light-keep-fraction',type=float,default=1./3)
    parser.add_argument('--goal-tolerance-xy',type=float,default=.3)
    parser.add_argument('--goal-tolerance-z',type=float,default=.2)
    args=parser.parse_args()
    if not np.isfinite(args.bridge_body_height) or args.bridge_body_height<=0:
        parser.error('--bridge-body-height must be positive and finite')
    if not np.isfinite(args.cost_limit) or not 0<args.cost_limit<=1:
        parser.error('--cost-limit must lie in (0, 1]')
    if not np.isfinite(args.light_update_hz) or args.light_update_hz<=0 or not 0<args.light_keep_fraction<1:
        parser.error('Invalid rolling reference rate or frozen fraction')
    if not np.isfinite([args.goal_tolerance_xy,args.goal_tolerance_z]).all() or min(args.goal_tolerance_xy,args.goal_tolerance_z)<=0:
        parser.error('Goal tolerances must be positive and finite')
    lock=Path(f'/tmp/pure3d_local_astar_follow_{os.getuid()}_{os.environ.get("ROS_DOMAIN_ID","0")}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init(args=[])
    stamp_ns=lambda s:s.sec*1000000000+s.nanosec
    durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sensor=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)

    class Follower(Node):
        def __init__(self):
            super().__init__('entry_local_astar_follow',parameter_overrides=[Parameter('use_sim_time',value=True)])
            self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
            engine=LocalCandidateQueries if args.backend=='light' else SurfaceQueries
            self.engine=engine(args.output/'planner',cost_limit=args.cost_limit);self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
            self.goal=None;self.generation=0;self.pending=False;self.attempt=0
            self.cloud=None;self.cloud_wall=0.;self.cost_valid=False;self.status_wall=0.
            self.body=None;self.tf_stamp=None;self.tf_changed_wall=0.;self.speed=0.
            self.phase='waiting_goal';self.hold=True;self.error=None;self.mode=None;self.mode_wall=0.
            self.mode_future=None;self.mode_sent=0.;self.last_mode_check=0.
            self.route=None;self.route_stamp=None;self.route_generation=None;self.ack_wall=None;self.published_wall=None
            self.arrival_since=None;self.events=[];self.records=[];self.trajectories=[];self.odometry=[]
            self.last_hold=0.;self.last_status=0.;self.last_marker=0.;self.mode_ready=False
            self.started_wall=0.;self.source=None;self.command_stamp=None;self.accepted=False
            self.query_times=[];self.fallback_enabled=False;self.handoff_started=None
            self.future_goal_id=None;self.path_publications=0;self.fallback_publications=0
            self.path_pub=self.create_publisher(PathMsg,'/pure_path',durable)
            self.fallback_pub=self.create_publisher(PoseStamped,'/scan/fallback_goal',10)
            self.astar_pub=self.create_publisher(PathMsg,'/entry/astar_path',durable)
            self.hold_pub=self.create_publisher(Bool,'/entry/navigation/hold',durable)
            self.status_pub=self.create_publisher(String,'/entry/local_astar_follow/status',durable)
            self.marker_pub=self.create_publisher(Marker,'/scan/optimal_list',durable)
            self.candidate_pub=self.create_publisher(MarkerArray,'/entry/local/candidate_paths',durable)
            self.last_report=None;self.continuations=0
            self.last_query_wall=0.;self.future_tail=False;self.rolling_ready=False;self.awaiting_reference_ack=False
            self.blocked_key=None;self.next_retry_wall=0.;self.recovery_queries=0
            self.resolved_goal=None;self.resolved_goal_wall=None;self.last_goal_result=None
            self.cancel_stamp=None;self.cancel_acknowledged=False;self.goal_received_stamp=-1
            self.cancel_pub=self.create_publisher(Header,'/scan/planning/cancel_goal',durable)
            self.create_subscription(Header,'/scan/planning/cancel_goal',self.on_external_cancel,durable)
            self.create_subscription(Header,'/scan/planning/goal_cancelled',self.on_cancel_ack,durable)
            self.create_service(Trigger,'/entry/navigation/cancel_goal',self.cancel_service)
            self.client=self.create_client(GetParameters,'/scan_planner_node/get_parameters')
            self.create_subscription(PoseStamped,'/scan/goal',self.on_goal,10)
            self.create_subscription(PointCloud2,'/entry/local/cost_cloud',self.on_cloud,durable)
            self.create_subscription(String,'/entry/local/status',self.on_status,durable)
            self.create_subscription(PathMsg,'/scan/planning/reference_accepted',self.on_ack,10)
            self.create_subscription(PathMsg,'/scan/planning/reference_rejected',self.on_rejected,10)
            self.create_subscription(PoseStamped,'/scan/planning/fallback_accepted',self.on_fallback_ack,10)
            self.create_subscription(Bspline,'/scan/planning/bspline',self.on_trajectory,10)
            self.create_subscription(Odometry,'/Odometry_gazebo',self.on_odometry,sensor)
            self.timer=self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
            for namespace in ['entry_astar_comparison','entry_scan_reference']:
                for number in [0,1]:
                    m=Marker();m.header.frame_id='world';m.ns=namespace;m.id=number;m.action=Marker.DELETE;self.marker_pub.publish(m)
            self.hold_pub.publish(Bool(data=True));self.event('started',route_topic='/pure_path',start_frame='base',goal_topic='/scan/goal',backend=args.backend,trigger='rolling_far_tail' if args.backend=='light' else 'goal_position_change')

        def event(self,name,**values):
            row=dict(event=name,wall=time.perf_counter(),sim=self.get_clock().now().nanoseconds*1e-9,**values)
            self.events.append(row);self.events=self.events[-5000:]
            print(json.dumps(row),flush=True)
            with (args.output/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

        def on_odometry(self,m):
            v=m.twist.twist.linear;self.speed=float(np.hypot(v.x,v.y))
            p=m.pose.pose.position
            self.odometry.append(dict(stamp=stamp_ns(m.header.stamp)*1e-9,xyz=[p.x,p.y,p.z],goal_id=self.generation))
            if len(self.odometry)>10000:self.odometry=self.odometry[-10000:]

        def on_status(self,m):
            value=json.loads(m.data);self.cost_valid=bool(value.get('valid')) and not value.get('error');self.status_wall=time.perf_counter()

        def on_cloud(self,m):
            if not m.width*m.height:self.cloud=None;return
            fields=[(f.name,f.offset,f.datatype,f.count) for f in m.fields]
            expected=[(n,4*i,7,1) for i,n in enumerate(['x','y','z','trav_cost'])]
            if (m.header.frame_id!='world' or m.height!=1 or m.point_step!=16 or fields!=expected or m.is_bigendian
                    or m.row_step != m.width*16 or len(m.data) != m.row_step):
                self.cloud=None;self.error='Unexpected local cost cloud contract';return
            a=np.frombuffer(m.data,dtype='<f4').reshape(-1,4).copy()
            if not np.isfinite(a).all():self.cloud=None;self.error='Nonfinite local cost';return
            self.cloud=dict(xyz=a[:,:3],cost=a[:,3],stamp=stamp_ns(m.header.stamp)*1e-9,
                            key=hashlib.blake2b(a.tobytes(),digest_size=10).hexdigest())
            self.cloud_wall=time.perf_counter()

        def on_goal(self,m):
            p=m.pose.position;goal=np.array([p.x,p.y,p.z])
            if not np.isfinite(goal).all() or not m.header.frame_id:return
            try:
                if m.header.frame_id!='world':
                    tf=self.buffer.lookup_transform('world',m.header.frame_id,Time.from_msg(m.header.stamp))
                    t,q=tf.transform.translation,tf.transform.rotation
                    goal=rotation_matrix([q.x,q.y,q.z,q.w])@goal+[t.x,t.y,t.z]
            except TransformException as error:self.error=str(error);return
            if self.goal is not None and np.linalg.norm(goal-self.goal)<.01:return
            self.goal=goal;self.generation+=1;self.pending=True;self.hold=True;self.phase='waiting_cost';self.error=None
            self.ack_wall=None;self.arrival_since=None;self.source=None;self.accepted=False
            self.rolling_ready=False;self.awaiting_reference_ack=False
            self.blocked_key=None;self.next_retry_wall=0.
            self.resolved_goal=None;self.resolved_goal_wall=None
            self.goal_received_stamp=stamp_ns(self.get_clock().now().to_msg())
            self.started_wall=time.perf_counter();self.handoff_started=None
            self.hold_pub.publish(Bool(data=True))
            self.event('goal_received',goal_id=self.generation,goal=goal.tolist())

        def goal_errors(self,now):
            settling=self.phase=='settling_goal'
            resolved=self.resolved_goal if self.resolved_goal_wall is not None and (settling or now-self.resolved_goal_wall<2.5) else None
            return arrival_errors(self.body,self.goal,resolved,args.goal_tolerance_xy,args.goal_tolerance_z,settling)

        def cancel_goal(self,reason,header=None):
            target=None if self.goal is None else self.goal.tolist()
            self.hold=True;self.hold_pub.publish(Bool(data=True))
            self.pending=False;self.accepted=False;self.source=None;self.rolling_ready=False
            self.awaiting_reference_ack=False;self.handoff_started=None;self.command_stamp=None
            self.clear_route();self.last_report=None
            self.goal=None;self.phase='arrived' if reason=='arrived' else 'cancelled'
            self.last_goal_result=dict(goal_id=self.generation,requested_goal=target,reason=reason,
                resolved_goal=None if self.resolved_goal is None else self.resolved_goal.tolist())
            message=header or Header(frame_id='world',stamp=self.get_clock().now().to_msg())
            self.cancel_stamp=stamp_ns(message.stamp);self.cancel_acknowledged=False
            if header is None:self.cancel_pub.publish(message)
            self.event('goal_cancel_requested',**self.last_goal_result,cancel_stamp_ns=self.cancel_stamp)
            atomic_json(args.output/'odometry.json',self.odometry)

        def on_external_cancel(self,message):
            stamp=stamp_ns(message.stamp)
            if stamp==self.cancel_stamp or stamp<self.goal_received_stamp:return
            self.cancel_goal('external_cancel',message)

        def on_cancel_ack(self,message):
            if stamp_ns(message.stamp)==self.cancel_stamp and not self.cancel_acknowledged:
                self.cancel_acknowledged=True;self.event('scan_goal_cancelled',goal_id=self.generation)

        def cancel_service(self,_request,response):
            self.cancel_goal('manual_cancel')
            response.success=True;response.message='Goal cleared and stop requested'
            return response

        def on_ack(self,m):
            if self.goal is None or self.phase in ['arrived','settling_goal','cancelled','handoff_failed','execution_timeout'] or self.source!='astar' or self.route is None or len(m.poses)<2 or m.header.frame_id!='world':return
            ends=np.array([[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in [m.poses[0],m.poses[-1]]])
            if route_ack_matches(stamp_ns(m.header.stamp),ends,self.route_stamp,self.route[[0,-1]],args.bridge_body_height):
                self.accepted=True;self.hold=False;self.phase='following_astar';self.handoff_started=None
                self.awaiting_reference_ack=False
                self.hold_pub.publish(Bool(data=False))
                self.event('scan_path_ack',goal_id=self.generation,poses=len(m.poses))

        def on_rejected(self,m):
            if (not self.awaiting_reference_ack or self.route is None or
                    self.route_generation!=self.generation or len(m.poses)<2):return
            ends=np.array([[p.pose.position.x,p.pose.position.y,p.pose.position.z]
                           for p in (m.poses[0],m.poses[-1])])
            if not route_ack_matches(stamp_ns(m.header.stamp),ends,self.route_stamp,
                                     self.route[[0,-1]],args.bridge_body_height):return
            now=time.perf_counter()
            report=self.last_report or {}
            index=report.get('scan_candidate_index',0)+1
            candidates=report.get('candidates',[])
            self.event('scan_reference_rejected',goal_id=self.generation,
                       candidate_index=index-1)
            if args.backend=='light' and index<len(candidates):
                report=dict(report,scan_candidate_index=index)
                self.publish_route(report,np.asarray(candidates[index]['body_xyz']),now)
            else:self.fallback('SCAN rejected all available candidate references',now)

        def on_fallback_ack(self,m):
            if self.phase in ['arrived','handoff_failed','execution_timeout'] or self.source!='goal' or stamp_ns(m.header.stamp)!=self.command_stamp:return
            p=m.pose.position
            if m.header.frame_id=='world' and np.allclose([p.x,p.y,p.z],self.goal,rtol=0,atol=1e-5):
                self.accepted=True;self.hold=False;self.phase='following_goal';self.handoff_started=None
                self.hold_pub.publish(Bool(data=False))
                self.event('scan_goal_ack',goal_id=self.generation)

        def clear_route(self):
            self.route=None;self.route_stamp=None
            msg=PathMsg();msg.header.frame_id='world';msg.header.stamp=self.get_clock().now().to_msg()
            self.astar_pub.publish(msg);self.path_pub.publish(msg)
            m=Marker();m.header=msg.header;m.ns='entry_local_astar_follow';m.id=0;m.action=Marker.DELETE;self.marker_pub.publish(m)
            self.publish_candidates(None,msg.header)

        def publish_candidates(self,report,header):
            markers=MarkerArray()
            for number in range(16):
                marker=Marker();marker.header=header;marker.ns='entry_light_candidates';marker.id=number
                marker.pose.orientation.w=1.
                items=[] if report is None else report.get('candidates',[])
                if number>=len(items):marker.action=Marker.DELETE
                else:
                    marker.type=Marker.LINE_STRIP;marker.action=Marker.ADD;marker.scale.x=.04 if number==0 else .018
                    marker.color.r=.95 if number==0 else .2;marker.color.g=.1 if number==0 else .5
                    marker.color.b=.5 if number==0 else 1.;marker.color.a=1. if number==0 else .35
                    marker.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2]+.02)) for p in items[number]['body_xyz']]
                markers.markers.append(marker)
            marker=Marker();marker.header=header;marker.ns='entry_light_frozen';marker.id=0;marker.pose.orientation.w=1.
            prefix=None if report is None else report.get('frozen_prefix_body')
            if prefix is None:marker.action=Marker.DELETE
            else:
                marker.type=Marker.LINE_STRIP;marker.scale.x=.055
                marker.color.r=1.;marker.color.g=.65;marker.color.a=1.
                marker.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2]+.04)) for p in prefix]
            markers.markers.append(marker)
            self.candidate_pub.publish(markers)

        def fallback(self,reason,now,*,replan_from_base=False):
            self.error=reason
            if self.mode==4 or args.no_goal_fallback:
                self.clear_route();self.source=None;self.accepted=False;self.hold=True
                self.phase='cost_blocked';self.pending=False;self.handoff_started=None
                self.rolling_ready=False;self.awaiting_reference_ack=False
                self.blocked_key=None if self.cloud is None else self.cloud['key']
                self.next_retry_wall=now+.5
                self.hold_pub.publish(Bool(data=True))
                self.event('cost_blocked',goal_id=self.generation,reason=reason,cost_limit=args.cost_limit)
                if replan_from_base and args.backend=='light':
                    self.phase='waiting_cost';self.pending=True;self.recovery_queries+=1
                    self.event('prefix_recovery_scheduled',goal_id=self.generation,
                               reason=reason,policy='hold_then_search_from_current_base')
                return
            if self.source=='goal':return
            self.clear_route();self.source='goal';self.accepted=False;self.hold=True;self.phase='waiting_scan_goal'
            self.hold_pub.publish(Bool(data=True))
            msg=PoseStamped();msg.header.frame_id='world';msg.header.stamp=self.get_clock().now().to_msg();msg.pose.orientation.w=1.
            msg.pose.position=Point(x=float(self.goal[0]),y=float(self.goal[1]),z=float(self.goal[2]))
            self.command_stamp=stamp_ns(msg.header.stamp);self.handoff_started=now;self.fallback_pub.publish(msg)
            self.fallback_publications+=1
            self.event('direct_goal_published',goal_id=self.generation,goal=self.goal.tolist(),reason=reason)

        def on_trajectory(self,m):
            try:
                path,duration=curve_samples([[p.x,p.y,p.z] for p in m.pos_pts],m.knots,m.order)
            except (ValueError, TypeError) as error:
                self.event('invalid_scan_trajectory',reason=str(error))
                return
            d=path_distances(path,self.route) if self.route is not None else None
            if self.source=='astar' and self.route_generation==self.generation:self.rolling_ready=True
            self.trajectories.append(dict(goal_id=self.generation,traj_id=m.traj_id,duration_s=duration,
                source=self.source,mean_deviation_m=None if d is None else float(d.mean()),max_deviation_m=None if d is None else float(d.max()),start=path[0].tolist(),end=path[-1].tolist()))
            self.event('scan_local_trajectory',**self.trajectories[-1]);self.trajectories=self.trajectories[-2000:]

        def check_mode(self,now):
            if self.mode_future is not None:
                if self.mode_future.done():
                    try:
                        response=self.mode_future.result();self.mode=response.values[0].integer_value if response.values else None;self.mode_wall=now
                        self.fallback_enabled=len(response.values)>1 and response.values[1].bool_value
                    except Exception:self.mode=None
                    self.mode_future=None
                elif now-self.mode_sent>2.:
                    self.mode_future.cancel();self.mode_future=None;self.mode=None
            if self.mode_future is None and now-self.last_mode_check>=.5 and self.client.service_is_ready():
                request=GetParameters.Request();request.names=['fsm.navi_mode','interface.allow_goal_fallback']
                self.mode_future=self.client.call_async(request);self.mode_sent=now;self.last_mode_check=now
            fallback_ready=(self.mode==4 or args.no_goal_fallback or
                            (self.fallback_enabled and self.fallback_pub.get_subscription_count()>0))
            self.mode_ready=(self.mode in (3,4) and fallback_ready and now-self.mode_wall<3. and
                             self.path_pub.get_subscription_count()>0)

        def publish_route(self,report,path,now):
            incremental=(report.get('reference_update')=='tail' and self.source=='astar' and
                         self.route_generation==self.generation and self.accepted)
            msg=PathMsg();msg.header.frame_id='world';msg.header.stamp=self.get_clock().now().to_msg()
            for p in path:
                pose=PoseStamped();pose.header=msg.header;pose.pose.orientation.w=1.;pose.pose.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]));msg.poses.append(pose)
            self.route=path;self.route_stamp=stamp_ns(msg.header.stamp);self.route_generation=self.generation
            switching=self.source!='astar'
            self.source='astar';self.command_stamp=self.route_stamp;self.published_wall=now;self.error=None
            self.awaiting_reference_ack=True
            if incremental:self.handoff_started=now
            elif switching or args.backend=='light':
                self.accepted=False;self.phase='waiting_scan_path';self.hold=True;self.handoff_started=now
                self.hold_pub.publish(Bool(data=True))
            self.astar_pub.publish(msg);self.path_pub.publish(msg);self.path_publications+=1
            self.last_report=report;self.publish_candidates(report,msg.header)
            self.event('pure_path_published',goal_id=self.generation,poses=len(path),start=path[0].tolist(),goal=path[-1].tolist(),
                       requested_goal=self.goal.tolist(),cloud_key=report['cloud_key'],graph_prepare_s=report['graph_prepare_s'],
                       reference_update=report.get('reference_update','full'),rolling=report.get('rolling'),
                       query_s=[r['call_wall_s'] for r in report['search_repeats']])
            m=Marker();m.header=msg.header;m.ns='entry_local_astar_follow';m.id=0;m.type=Marker.LINE_STRIP;m.pose.orientation.w=1.
            m.scale.x=.045;m.color.r=.1;m.color.g=.85;m.color.b=1.;m.color.a=1.
            m.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2]+.02)) for p in path];self.marker_pub.publish(m)

        def tick(self):
            now=time.perf_counter();self.check_mode(now)
            base_valid=False
            try:
                transform=self.buffer.lookup_transform('world','base',Time())
                tf_stamp=stamp_ns(transform.header.stamp)
                if tf_stamp!=self.tf_stamp:self.tf_stamp=tf_stamp;self.tf_changed_wall=now
                p=transform.transform.translation;self.body=np.array([p.x,p.y,p.z])
                age=self.get_clock().now().nanoseconds-tf_stamp
                base_valid=np.isfinite(self.body).all() and -.1e9<=age<=.3e9 and now-self.tf_changed_wall<1.
            except TransformException:pass
            fresh=bool(base_valid and self.cloud is not None and self.cost_valid and now-self.cloud_wall<2.5 and now-self.status_wall<2.5)
            owners=[p.node_name for p in self.get_publishers_info_by_topic('/pure_path') if p.node_name!=self.get_name()]
            other_holds=[p.node_name for p in self.get_publishers_info_by_topic('/entry/navigation/hold') if p.node_name!=self.get_name()]
            ready=base_valid and self.mode_ready and not owners and not other_holds
            if (args.backend=='light' and self.phase=='cost_blocked' and self.goal is not None
                    and ready and fresh and self.cloud['key']!=self.blocked_key
                    and now>=self.next_retry_wall):
                self.phase='waiting_cost';self.pending=True;self.recovery_queries+=1
                self.event('cost_map_recovery_scheduled',goal_id=self.generation,cloud_key=self.cloud['key'])
            active=self.goal is not None and self.phase not in ['arrived','execution_timeout','handoff_failed','cost_blocked']
            if ready and self.goal is not None and self.goal_errors(now)['inside']:active=True
            if not ready:
                self.hold=True
                if owners or other_holds:self.phase='conflicting_publisher';self.error=str(owners+other_holds)
                elif active:self.phase='waiting_scan_mode' if not self.mode_ready else 'waiting_base_tf'
            if self.future is not None and self.future.done() and (ready or self.future_goal_id!=self.generation or not active):
                future,self.future=self.future,None
                try:
                    report,path=future.result();self.records.append(report);self.records=self.records[-2000:]
                    self.query_times.append(report['elapsed_before_report_s']);self.query_times=self.query_times[-5000:]
                    if self.goal is None:self.event('cancelled_plan_discarded',goal_id=report['manual_goal_id'])
                    elif report['manual_goal_id']!=self.generation:self.event('superseded_plan',goal_id=report['manual_goal_id'])
                    elif ready and active:
                        if self.phase!='settling_goal':
                            resolved=report.get('resolved_goal_body')
                            self.resolved_goal=None if resolved is None else np.asarray(resolved)
                            self.resolved_goal_wall=now
                        if self.goal_errors(now)['inside']:
                            self.hold=True;self.accepted=False;self.phase='settling_goal'
                        elif not report['success'] or path is None or len(path)<2:
                            if self.future_tail:
                                self.event('tail_update_rejected',goal_id=self.generation,reason=report.get('error'))
                            else:self.fallback(report.get('error') or f'No connected A* route with cost < {args.cost_limit:g}',now)
                        elif not fresh or np.linalg.norm(np.asarray(report['requested_body_endpoints'][0])-self.body)>.3:
                            self.event('stale_plan_discarded',goal_id=self.generation)
                            if not self.future_tail:self.fallback('Planning snapshot became stale before handoff',now)
                        elif args.backend=='light' or self.source!='astar' or self.accepted:self.publish_route(report,path,now)
                except Exception as error:
                    self.event('planning_error',error=str(error))
                    if ready and active and self.future_goal_id==self.generation and (not self.future_tail or isinstance(error,FrozenPrefixInvalid)):
                        self.fallback(str(error),now,replan_from_base=isinstance(error,FrozenPrefixInvalid))
            if ready and active:
                errors=self.goal_errors(now)
                if errors['inside']:
                    self.hold=True;self.phase='settling_goal';active=False;self.handoff_started=None
                    self.pending=False;self.awaiting_reference_ack=False;self.accepted=False;self.rolling_ready=False
                    self.hold_pub.publish(Bool(data=True))
                    if self.speed<.1:
                        if self.arrival_since is None:self.arrival_since=self.tf_stamp
                        if self.tf_stamp-self.arrival_since>=.5e9:
                            self.event('arrived',goal_id=self.generation,body=self.body.tolist(),
                                       requested_goal_error_m=float(np.linalg.norm(self.body-self.goal)),
                                       arrival_errors=errors)
                            self.cancel_goal('arrived')
                    else:self.arrival_since=None
                elif args.backend=='light' and self.phase=='settling_goal':
                    self.arrival_since=None
                    if self.accepted:self.phase='following_astar'
                    elif self.published_wall is None or now-self.published_wall>=1.:
                        self.pending=True;self.phase='waiting_cost'
                if self.handoff_started is not None and now-self.handoff_started>3.:
                    self.hold=True;self.phase='handoff_failed';self.error='SCAN command acknowledgement timeout'
                    self.event('handoff_failed',error=self.error);active=False
                if active and self.accepted:
                    self.hold=False
                    self.phase='following_astar' if self.source=='astar' else 'following_goal'
                    self.arrival_since=None
                    if (args.backend=='light' and active and self.rolling_ready and self.future is None and
                            not self.pending and not self.awaiting_reference_ack and fresh and
                            now-self.last_query_wall>=1./args.light_update_hz):
                        self.pending=True;self.continuations+=1
                        self.event('rolling_reference_refresh',goal_id=self.generation,update=self.continuations)
                    if args.backend=='light' and (not fresh or
                            (self.route is not None and np.linalg.norm(self.body-self.route[-1])<.2)):
                        self.hold=True;self.phase='waiting_reference_update'
                if active and now-self.started_wall>120.:
                    self.hold=True;self.phase='execution_timeout';active=False;self.event('execution_timeout')
                if active and self.pending and self.future is None and now>=self.next_retry_wall:
                    self.pending=False
                    if not fresh:
                        self.fallback('Cost unavailable or stale for this goal',now)
                    else:
                        self.start_goal_query(now)
            if now-self.last_hold>=.2:
                self.hold_pub.publish(Bool(data=self.hold));self.last_hold=now
            if now-self.last_status>=1.:
                self.save_status(now,fresh)

        def start_goal_query(self,now):
            self.attempt+=1;self.future_goal_id=self.generation
            self.last_query_wall=now;self.future_tail=False
            ground=float(self.body[2]-args.bridge_body_height)
            request=dict(id=self.generation,body_endpoints=np.array([self.body.copy(),self.goal.copy()]),
                         body_reference_height_m=args.bridge_body_height,
                         start_source='TF world<-base at planning snapshot',base_transform_stamp_ns=self.tf_stamp)
            if args.backend=='light' and self.accepted and self.route_generation==self.generation:
                try:
                    prefix,detail=rolling_prefix(self.route,self.body,args.light_keep_fraction)
                    request.update(frozen_prefix_body=prefix.tolist(),rolling=detail)
                    self.future_tail=True
                except ValueError:
                    if not light_refresh_due(self.route,self.body,self.goal,now-self.published_wall):return
            self.event('planning',goal_id=self.generation,start_frame='base',start=self.body.tolist(),
                       goal=self.goal.tolist(),cloud_key=self.cloud['key'],
                       body_reference_height_m=args.bridge_body_height)
            self.future=self.pool.submit(self.engine.compute,self.cloud,request,ground,self.attempt,args.output/'planner',online=True)

        def save_status(self,now,fresh):
            times=np.asarray(self.query_times[-1000:])*1000.
            value=dict(phase=self.phase,hold=self.hold,error=self.error,goal_id=self.generation,source=self.source,
                body=None if self.body is None else self.body.tolist(),goal=None if self.goal is None else self.goal.tolist(),
                scan_mode=self.mode,scan_mode_ready=self.mode_ready,cost_fresh=fresh,
                route_topic='/pure_path',start_frame='base',goal_topic='/scan/goal',fallback_topic='/scan/fallback_goal',
                cost_integration='global_astar_reference',refresh_mode='goal_position_change',refresh_hz=0.,
                body_reference_height_m=args.bridge_body_height,cost_limit=args.cost_limit,goal_fallback_allowed=self.mode==3 and self.fallback_enabled and not args.no_goal_fallback,
                additional_step_connections=self.engine.planner.options.step_connections,
                resolved_goal=None if self.resolved_goal is None else self.resolved_goal.tolist(),
                goal_tolerance_xy_m=args.goal_tolerance_xy,goal_tolerance_z_m=args.goal_tolerance_z,
                arrival_errors=None if self.goal is None or self.body is None else self.goal_errors(now),
                cancel_acknowledged=self.cancel_acknowledged,last_goal_result=self.last_goal_result,
                goal_position_threshold_m=.01,pending_goal=self.pending,
                astar_publications=self.path_publications,fallback_publications=self.fallback_publications,
                query_p50_ms=None if not len(times) else float(np.median(times)),
                query_p95_ms=None if not len(times) else float(np.percentile(times,95)),
                attempts=self.attempt,local_trajectories=len(self.trajectories),base_transform_stamp_ns=self.tf_stamp)
            if args.backend=='light':value.update(backend='light',cost_integration='local_candidate_reference',
                refresh_mode='rolling_far_tail',refresh_hz=args.light_update_hz,keep_fraction=args.light_keep_fraction,
                segment_continuations=self.continuations,awaiting_reference_ack=self.awaiting_reference_ack,
                recovery_queries=self.recovery_queries,blocked_cloud_key=self.blocked_key,
                last_candidates=None if self.last_report is None else self.last_report['candidate_metrics'],
                last_selected=None if self.last_report is None else {k:v for k,v in self.last_report['candidates'][0].items() if k!='body_xyz'})
            atomic_json(args.output/'status.json',value);self.status_pub.publish(String(data=json.dumps(value)));self.last_status=now
            atomic_json(args.output/'execution.json',dict(phase=self.phase,events=self.events,trajectories=self.trajectories))


        def close(self):
            self.hold=True
            if rclpy.ok():self.hold_pub.publish(Bool(data=True))
            self.pool.shutdown(wait=True);self.engine.planner.close()
            atomic_json(args.output/'odometry.json',self.odometry);self.event('closed_and_held')
            atomic_json(args.output/'execution.json',dict(phase=self.phase,events=self.events,trajectories=self.trajectories))

    node=Follower()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        lock.close()


if __name__=='__main__':main()
