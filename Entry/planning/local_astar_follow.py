"""Current world<-base + /scan/goal + local raw cost -> planner -> /pure_path.

SCAN modes 3/4 remain responsible for collision-aware local B-splines and cmd_vel.
A* is triggered once per changed goal. SCAN owns continuous local replanning.
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
from Entry.utils.surface import SurfaceQueries, curve_samples
from Entry.utils.io import atomic_json


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
    from std_msgs.msg import Bool,String
    from visualization_msgs.msg import Marker
    from scan_planner_msgs.msg import Bspline
    from tf2_ros import Buffer,TransformListener,TransformException
    from common.unitree_gazebo.scripts.frame_config import rotation_matrix

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bridge-body-height',type=float,default=.4)
    parser.add_argument('--cost-limit',type=float,default=.5)
    parser.add_argument('--no-goal-fallback',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.cost_limit) or not 0<args.cost_limit<=1:
        parser.error('--cost-limit must lie in (0, 1]')
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
            self.engine=SurfaceQueries(args.output/'planner',cost_limit=args.cost_limit);self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
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
            self.client=self.create_client(GetParameters,'/scan_planner_node/get_parameters')
            self.create_subscription(PoseStamped,'/scan/goal',self.on_goal,10)
            self.create_subscription(PointCloud2,'/entry/local/cost_cloud',self.on_cloud,durable)
            self.create_subscription(String,'/entry/local/status',self.on_status,durable)
            self.create_subscription(PathMsg,'/scan/planning/reference_accepted',self.on_ack,10)
            self.create_subscription(PoseStamped,'/scan/planning/fallback_accepted',self.on_fallback_ack,10)
            self.create_subscription(Bspline,'/scan/planning/bspline',self.on_trajectory,10)
            self.create_subscription(Odometry,'/Odometry_gazebo',self.on_odometry,sensor)
            self.timer=self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
            for namespace in ['entry_astar_comparison','entry_scan_reference']:
                for number in [0,1]:
                    m=Marker();m.header.frame_id='world';m.ns=namespace;m.id=number;m.action=Marker.DELETE;self.marker_pub.publish(m)
            self.hold_pub.publish(Bool(data=True));self.event('started',route_topic='/pure_path',start_frame='base',goal_topic='/scan/goal',trigger='goal_position_change')

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
            self.started_wall=time.perf_counter();self.handoff_started=None
            self.hold_pub.publish(Bool(data=True))
            self.event('goal_received',goal_id=self.generation,goal=goal.tolist())

        def on_ack(self,m):
            if self.phase in ['arrived','handoff_failed','execution_timeout'] or self.source!='astar' or self.route is None or len(m.poses)<2 or m.header.frame_id!='world':return
            ends=np.array([[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in [m.poses[0],m.poses[-1]]])
            if route_ack_matches(stamp_ns(m.header.stamp),ends,self.route_stamp,self.route[[0,-1]],args.bridge_body_height):
                self.accepted=True;self.hold=False;self.phase='following_astar';self.handoff_started=None
                self.hold_pub.publish(Bool(data=False))
                self.event('scan_path_ack',goal_id=self.generation,poses=len(m.poses))

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

        def fallback(self,reason,now):
            self.error=reason
            if self.mode==4 or args.no_goal_fallback:
                self.clear_route();self.source=None;self.accepted=False;self.hold=True
                self.phase='cost_blocked';self.pending=False;self.handoff_started=None
                self.hold_pub.publish(Bool(data=True))
                self.event('cost_blocked',goal_id=self.generation,reason=reason,cost_limit=args.cost_limit)
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
            msg=PathMsg();msg.header.frame_id='world';msg.header.stamp=self.get_clock().now().to_msg()
            for p in path:
                pose=PoseStamped();pose.header=msg.header;pose.pose.orientation.w=1.;pose.pose.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]));msg.poses.append(pose)
            self.route=path;self.route_stamp=stamp_ns(msg.header.stamp);self.route_generation=self.generation
            switching=self.source!='astar'
            self.source='astar';self.command_stamp=self.route_stamp;self.published_wall=now;self.error=None
            if switching:
                self.accepted=False;self.phase='waiting_scan_path';self.hold=True;self.handoff_started=now
                self.hold_pub.publish(Bool(data=True))
            self.astar_pub.publish(msg);self.path_pub.publish(msg);self.path_publications+=1
            self.event('pure_path_published',goal_id=self.generation,poses=len(path),start=path[0].tolist(),goal=path[-1].tolist(),
                       requested_goal=self.goal.tolist(),cloud_key=report['cloud_key'],graph_prepare_s=report['graph_prepare_s'],
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
            active=self.goal is not None and self.phase not in ['arrived','execution_timeout','handoff_failed','cost_blocked']
            if not ready:
                self.hold=True
                if owners or other_holds:self.phase='conflicting_publisher';self.error=str(owners+other_holds)
                elif active:self.phase='waiting_scan_mode' if not self.mode_ready else 'waiting_base_tf'
            if self.future is not None and self.future.done() and (ready or self.future_goal_id!=self.generation or not active):
                future,self.future=self.future,None
                try:
                    report,path=future.result();self.records.append(report);self.records=self.records[-2000:]
                    self.query_times.append(report['elapsed_before_report_s']);self.query_times=self.query_times[-5000:]
                    if report['manual_goal_id']!=self.generation:self.event('superseded_plan',goal_id=report['manual_goal_id'])
                    elif ready and active:
                        if not report['success'] or path is None or len(path)<2:
                            self.fallback(report.get('error') or f'No connected A* route with cost < {args.cost_limit:g}',now)
                        elif not fresh or np.linalg.norm(np.asarray(report['requested_body_endpoints'][0])-self.body)>.3:
                            self.event('stale_plan_discarded',goal_id=self.generation)
                            self.fallback('Planning snapshot became stale before handoff',now)
                        elif self.source!='astar' or self.accepted:self.publish_route(report,path,now)
                except Exception as error:
                    self.event('planning_error',error=str(error))
                    if ready and active and self.future_goal_id==self.generation:self.fallback(str(error),now)
            if ready and active:
                if self.handoff_started is not None and now-self.handoff_started>3.:
                    self.hold=True;self.phase='handoff_failed';self.error='SCAN command acknowledgement timeout'
                    self.event('handoff_failed',error=self.error);active=False
                if active and self.accepted:
                    self.hold=False
                    self.phase='following_astar' if self.source=='astar' else 'following_goal'
                    error=float(np.linalg.norm(self.body-self.goal));inside=error<=.2 and self.speed<.1
                    if inside:
                        if self.arrival_since is None:self.arrival_since=self.tf_stamp
                        if self.tf_stamp-self.arrival_since>=.5e9:
                            self.hold=True;self.phase='arrived';active=False
                            self.event('arrived',goal_id=self.generation,body=self.body.tolist(),requested_goal_error_m=error)
                            atomic_json(args.output/'odometry.json',self.odometry)
                    else:self.arrival_since=None
                if now-self.started_wall>120.:
                    self.hold=True;self.phase='execution_timeout';active=False;self.event('execution_timeout')
                if active and self.pending and self.future is None:
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
            xyz=self.cloud['xyz'];near=(np.linalg.norm(xyz[:,:2]-self.body[:2],axis=1)<1.5)&(xyz[:,2]<self.body[2]-.15)&(xyz[:,2]>self.body[2]-.65)
            if not near.any():self.fallback('No observed support-height candidates near base',now)
            else:
                ground=float(np.median(xyz[near,2]))
                request=dict(id=self.generation,body_endpoints=np.array([self.body.copy(),self.goal.copy()]),
                             start_source='TF world<-base at planning snapshot',base_transform_stamp_ns=self.tf_stamp)
                self.event('planning',goal_id=self.generation,start_frame='base',start=self.body.tolist(),goal=self.goal.tolist(),cloud_key=self.cloud['key'])
                self.future=self.pool.submit(self.engine.compute,self.cloud,request,ground,self.attempt,args.output/'planner',online=True)

        def save_status(self,now,fresh):
            times=np.asarray(self.query_times[-1000:])*1000.
            value=dict(phase=self.phase,hold=self.hold,error=self.error,goal_id=self.generation,source=self.source,
                body=None if self.body is None else self.body.tolist(),goal=None if self.goal is None else self.goal.tolist(),
                scan_mode=self.mode,scan_mode_ready=self.mode_ready,cost_fresh=fresh,
                route_topic='/pure_path',start_frame='base',goal_topic='/scan/goal',fallback_topic='/scan/fallback_goal',
                cost_integration='global_astar_reference',refresh_mode='goal_position_change',refresh_hz=0.,
                cost_limit=args.cost_limit,goal_fallback_allowed=self.mode==3 and self.fallback_enabled and not args.no_goal_fallback,
                goal_position_threshold_m=.01,pending_goal=self.pending,
                astar_publications=self.path_publications,fallback_publications=self.fallback_publications,
                query_p50_ms=None if not len(times) else float(np.median(times)),
                query_p95_ms=None if not len(times) else float(np.percentile(times,95)),
                attempts=self.attempt,local_trajectories=len(self.trajectories),base_transform_stamp_ns=self.tf_stamp)
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
