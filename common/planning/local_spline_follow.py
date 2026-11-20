"""Local support graph, ranked routes and scheduled XYZ B-spline execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from common.io import atomic_json
from common.planning.surface import LocalCandidateQueries
from planner.light_planner.graph.candidates import CandidateOptions
from planner.light_planner.optimization.support_spline import (
    InflatedOccupancy, SplineState, TerrainLimits, arc_path, fit_support_spline, project_path,
    scan_bspline_message, slice_path,
)


@dataclass(frozen=True)
class Handoff:
    predecessor: int
    elapsed: float
    state: SplineState


def rolling_cut(spline, elapsed):
    """Fix the switch at one third of the remaining sampled 3-D arc."""
    count = max(2, int(np.ceil((spline.duration - elapsed) / .025)) + 1)
    times = np.linspace(elapsed, spline.duration, count)
    positions = spline.curve(times)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    at = max(float(np.interp(arc[-1] / 3, arc, times)), elapsed + .45,
             spline.profile.get('start_exemption_until_s', 0.) + .20)
    at = min(at, spline.duration)
    return at, float(np.interp(at, times, arc)), float(arc[-1])


def check_start_connection(planner, support, times, origin, radius):
    """Exempt a single same-height prefix ending at the first observed support."""
    origin_local = origin @ planner.frame
    local = support @ planner.frame
    for index, sample_local in enumerate(local):
        match = planner.match_best_layer(*sample_local)
        if match is not None and abs((planner.node_xyz[match] @ planner.frame)[2] - sample_local[2]) <= .07:
            connection = planner.validate_trajectory(support[index:], check_cost=False)
            if not connection['valid']:
                raise ValueError('Spline support check failed after blind zone: ' + connection['reason'])
            return float(times[index])
        if (np.linalg.norm(sample_local[:2] - origin_local[:2]) > radius or
                abs(sample_local[2] - origin_local[2]) > .07):
            raise ValueError('Start connection leaves the blind zone before reaching support')
    raise ValueError('Start connection reaches no observed support')


def plan_candidates(engine, cloud, request, ground, attempt, directory, handoff):
    """Search once, then fit the first few ranked and geometrically valid alternatives."""
    occupied = request.get('occupied_voxels')
    occupancy = InflatedOccupancy(*occupied) if occupied is not None else None
    report, _ = engine.compute(cloud, dict(request, body_occupancy=occupancy), ground, attempt, directory)
    height = report['body_height']
    limits = replace(TerrainLimits(), body_height_m=height, **request.get('trajectory_limits', {}))
    offset = [0., 0., height]
    fits = []
    rejected = []
    origin = request['body_endpoints'][0] - offset
    blind_radius = request.get('start_blind_radius', 0.)
    ranked = report['candidates']
    by_sector = {}
    for candidate in ranked:
        by_sector.setdefault(candidate.get('direction_sector'), candidate)
    best_ids = {candidate['candidate_id'] for candidate in by_sector.values()}
    alternatives = [candidate for candidate in ranked if candidate['candidate_id'] not in best_ids]
    for candidate in [*by_sector.values(), *alternatives]:
        if len(candidate['support_xyz']) < 2:
            continue
        try:
            points, arc = arc_path(candidate['support_xyz'])
            start_support = origin.copy()
            delta = (points[0] - start_support) @ engine.planner.frame
            attach = (not handoff.predecessor and report['endpoint_snap']['start_fallback'] and
                      0. < np.linalg.norm(delta[:2]) <= blind_radius and abs(delta[2]) <= .15)
            if attach:
                start_support[2] = points[0,2]
                reference = np.vstack((start_support, points))
            else:
                progress = project_path(points, arc, start_support, horizontal=True)
                reference = slice_path(points, progress, arc[-1])
            fit = fit_support_spline(reference, limits, start=handoff.state, occupancy=occupancy)
            times = np.linspace(0., fit.duration, int(np.ceil(fit.duration/.02))+1)
            support = fit.curve(times)
            support[:, 2] = np.interp(times, fit.profile['profile_t_s'], fit.profile['reference_support_xyz'][:, 2])
            if attach:
                fit.profile['start_exemption_until_s'] = check_start_connection(
                    engine.planner, support, times, start_support, blind_radius)
            else:
                check = engine.planner.validate_trajectory(support, check_cost=False)
                if not check['valid']:
                    raise ValueError('Spline support check failed: '+check['reason'])
            fits.append((fit, fit.profile['reference_body_xyz']))
        except ValueError as error:
            rejected.append(str(error))
        if len(fits)>=3:
            break
    return report, fits, list(dict.fromkeys(rejected))


def main():
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Odometry, Path as PathMsg
    from scan_planner_msgs.msg import ScheduledSpline, SplineProgress
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points
    from std_msgs.msg import Bool, String
    from tf2_ros import Buffer, TransformListener, TransformException
    from visualization_msgs.msg import Marker, MarkerArray
    from Simulator.unitree_gazebo.scripts.frame_config import rotation_matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cost-limit', type=float, default=.8)
    parser.add_argument('--start-blind-radius', type=float, default=.70,
                        help='Maximum same-layer start connection from base, in meters; zero disables it')
    parser.add_argument('--occupancy-resolution', type=float, default=.05)
    parser.add_argument('--body-offset', type=float, default=.18)
    parser.add_argument('--max-xy', type=float, default=TerrainLimits.max_xy_mps)
    parser.add_argument('--max-acc', type=float, default=TerrainLimits.max_acc_mps2)
    parser.add_argument('--body-reference-height', type=float, default=TerrainLimits.body_height_m)
    parser.add_argument('--min-body-clearance', type=float, default=TerrainLimits.min_body_clearance_m)
    parser.add_argument('--use-sim-time', choices=['true', 'false'], default='true')
    args = parser.parse_args()
    if not np.isfinite(args.cost_limit) or not 0<args.cost_limit<=1:
        parser.error('--cost-limit must be in (0,1]')
    if not np.isfinite(args.start_blind_radius) or not 0 <= args.start_blind_radius <= 1:
        parser.error('--start-blind-radius must be in [0,1]')
    if not (np.isfinite([args.occupancy_resolution, args.body_offset]).all() and
            args.occupancy_resolution > 0 and args.body_offset >= 0):
        parser.error('Invalid occupancy resolution or body offset')
    if not (np.isfinite([args.max_xy,args.max_acc,args.body_reference_height,args.min_body_clearance]).all() and
            0 < args.max_xy <= 1. and args.max_acc > 0 and
            0 <= args.min_body_clearance < args.body_reference_height):
        parser.error('Invalid speed, acceleration or body clearance limit')
    lock = open(f'/tmp/pure3d_local_astar_follow_{os.getuid()}_{os.environ.get("ROS_DOMAIN_ID","0")}.lock','a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init(args=[])
    durable = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

    class LocalPlanner(Node):
        def __init__(self):
            super().__init__('entry_local_spline', parameter_overrides=[
                Parameter('use_sim_time',value=args.use_sim_time=='true')])
            self.tf = Buffer()
            self.listener = TransformListener(self.tf,self)
            self.engine = LocalCandidateQueries(args.output/'planner',cost_limit=args.cost_limit,
                                                options=CandidateOptions(direction_sectors=8))
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.body = self.goal = self.cloud = self.progress = None
            self.occupied, self.occupied_wall = None, 0.
            self.body_wall = self.cloud_wall = self.progress_wall = self.cost_wall = 0.
            self.cost_valid = False
            self.speed = 0.
            self.generation = self.attempt = 0
            self.arrival_since = None
            self.arrived = False
            self.phase = 'waiting_goal'
            self.hold = True
            self.goal_wall = self.last_query = self.last_status = self.last_hold = 0.
            self.future = self.query_context = self.pending = None
            self.backups = []
            self.plans = {}
            self.cut = None
            self.last_id = int(time.time_ns())
            self.publications = 0
            self.last_query_ms = None
            self.error = None
            self.goal_sub = self.create_subscription(PoseStamped,'/scan/goal',self.on_goal,10)
            self.body_sub = self.create_subscription(Odometry,'/scan/body_pose',self.on_body,qos_profile_sensor_data)
            self.cloud_sub = self.create_subscription(PointCloud2,'/entry/local/cost_cloud',self.on_cloud,durable)
            self.occupancy_sub = self.create_subscription(
                PointCloud2, '/scan/grid_map/occupancy_inflate', self.on_occupancy, qos_profile_sensor_data)
            self.cost_sub = self.create_subscription(String,'/entry/local/status',self.on_cost,durable)
            self.progress_sub = self.create_subscription(SplineProgress,'/scan/planning/spline_progress',self.on_progress,10)
            self.spline_pub = self.create_publisher(ScheduledSpline,'/scan/planning/scheduled_spline',10)
            self.hold_pub = self.create_publisher(Bool,'/entry/navigation/hold',durable)
            self.path_pub = self.create_publisher(PathMsg,'/pure_path',durable)
            self.ref_pub = self.create_publisher(PathMsg,'/entry/astar_path',durable)
            self.candidates_pub = self.create_publisher(MarkerArray,'/entry/local/candidate_paths',durable)
            self.marker_pub = self.create_publisher(Marker,'/scan/optimal_list',durable)
            self.status_pub = self.create_publisher(String,'/entry/local_astar_follow/status',durable)
            self.timer = self.create_timer(.05,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.set_hold(True)

        def event(self, name, **fields):
            record = dict(event=name,goal_id=self.generation,sim=self.get_clock().now().nanoseconds*1e-9,**fields)
            with (args.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')

        def set_hold(self, value):
            if value and not self.hold:
                marker = Marker(ns='entry_support_spline', id=0, action=Marker.DELETE)
                marker.header.frame_id = 'world'
                self.marker_pub.publish(marker)
            self.hold=value
            self.hold_pub.publish(Bool(data=value))
            self.last_hold=time.monotonic()

        def on_goal(self, msg):
            point = msg.pose.position
            goal = np.array([point.x,point.y,point.z])
            if not msg.header.frame_id or not np.isfinite(goal).all():
                return
            if msg.header.frame_id!='world':
                try:
                    transform=self.tf.lookup_transform('world',msg.header.frame_id,Time.from_msg(msg.header.stamp))
                except TransformException as error:
                    self.error=str(error)
                    return
                t,q=transform.transform.translation,transform.transform.rotation
                goal=rotation_matrix([q.x,q.y,q.z,q.w])@goal+[t.x,t.y,t.z]
            if self.goal is not None and np.linalg.norm(goal-self.goal)<.01:
                return
            self.goal=goal
            self.generation+=1
            self.arrived=False
            self.arrival_since=None
            self.pending=None
            self.cut=None
            self.backups=[]
            self.goal_wall=time.monotonic()
            self.set_hold(True)
            self.phase='waiting_cost'
            self.error=None
            self.event('goal_received',goal=goal.tolist())

        def on_body(self, msg):
            p=msg.pose.pose.position
            v=msg.twist.twist.linear
            body=np.array([p.x,p.y,p.z])
            self.body=body if msg.header.frame_id=='world' and np.isfinite(body).all() else None
            self.speed=float(np.hypot(v.x,v.y))
            self.body_wall=time.monotonic()

        def on_cloud(self, msg):
            wanted=[(name,4*i,7,1) for i,name in enumerate(('x','y','z','trav_cost'))]
            fields=[(f.name,f.offset,f.datatype,f.count) for f in msg.fields]
            if (not msg.width or msg.height!=1 or msg.header.frame_id!='world' or msg.is_bigendian or
                    msg.point_step!=16 or fields!=wanted or len(msg.data)!=msg.width*16 or msg.row_step!=msg.width*16):
                self.cloud=None
                return
            rows=np.frombuffer(msg.data,dtype='<f4').reshape(-1,4).copy()
            if not np.isfinite(rows).all():
                self.cloud=None
                return
            self.cloud=dict(xyz=rows[:,:3],cost=rows[:,3],
                key=hashlib.blake2b(rows,digest_size=10).hexdigest(),
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
            self.cloud_wall=time.monotonic()

        def on_cost(self,msg):
            try:
                state=json.loads(msg.data)
                self.cost_valid=bool(state.get('valid')) and not state.get('error')
            except (ValueError, AttributeError):
                self.cost_valid=False
            self.cost_wall=time.monotonic()

        def on_occupancy(self, msg):
            now = time.monotonic()
            if msg.header.frame_id != 'world' or now - self.occupied_wall < .25:
                return
            rows = read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            xyz = np.column_stack((rows['x'], rows['y'], rows['z']))
            self.occupied = xyz[np.isfinite(xyz).all(axis=1)]
            self.occupied_wall = now

        def on_progress(self,msg):
            self.progress=msg
            self.progress_wall=time.monotonic()

        def start_query(self,now):
            active=self.progress.active_id
            if active:
                if active not in self.plans or self.progress.frozen:
                    return
                old, _, _, height=self.plans[active]
                if self.cut is None or self.cut[0] != active or self.progress.elapsed >= self.cut[1]:
                    at, kept, remaining = rolling_cut(old, self.progress.elapsed)
                    self.cut = (active, at)
                    self.event('lock_near_spline', trajectory_id=active, switch_at_s=at,
                               frozen_arc_m=kept, remaining_arc_m=remaining)
                at = self.cut[1]
                state=old.state(at)
                handoff=Handoff(active,at,state)
                ground=float(np.interp(at, old.profile['profile_t_s'], old.profile['reference_support_xyz'][:, 2]))
                start_point=np.array([state.position[0],state.position[1],ground+height])
            else:
                if not self.hold or not np.isfinite(self.speed) or self.speed>.12:
                    return
                handoff=Handoff(0,0.,SplineState(self.body.copy(),np.zeros(3),np.zeros(3)))
                start_point=self.body.copy()
                ground=float(start_point[2]-args.body_reference_height)
            self.attempt+=1
            self.last_query=now
            request=dict(id=self.generation, body_endpoints=np.array([start_point,self.goal]),
                         raw_support_candidates=True, start_blind_radius=args.start_blind_radius,
                         body_reference_height_m=args.body_reference_height,
                         trajectory_limits={'max_xy_mps': args.max_xy, 'max_acc_mps2': args.max_acc,
                                            'min_body_clearance_m': args.min_body_clearance})
            if self.occupied is not None and now - self.occupied_wall <= .8:
                request['occupied_voxels'] = (self.occupied, args.occupancy_resolution, args.body_offset)
            self.query_context=(self.generation,handoff,self.cloud_wall)
            self.future=self.pool.submit(plan_candidates,self.engine,self.cloud,request,ground,
                                         self.attempt,args.output/'planner',handoff)

        def display(self,plan,route):
            header=PathMsg().header
            header.frame_id='world'
            header.stamp=self.get_clock().now().to_msg()
            path=PathMsg(header=header)
            for x,y,z in route:
                pose=PoseStamped(header=header)
                pose.pose.position=Point(x=float(x),y=float(y),z=float(z))
                pose.pose.orientation.w=1.
                path.poses.append(pose)
            self.path_pub.publish(path)
            self.ref_pub.publish(path)
            self.publications+=1
            m=Marker(header=header,ns='entry_support_spline',id=0,type=Marker.LINE_STRIP,action=Marker.ADD)
            m.pose.orientation.w=1.
            m.scale.x=.045
            m.color.r=1.
            m.color.g=.7
            m.color.a=1.
            m.points=[Point(x=float(x),y=float(y),z=float(z)) for x,y,z in plan.curve(np.linspace(0.,plan.duration,101))]
            self.marker_pub.publish(m)

        def display_candidates(self,report):
            header=PathMsg().header
            header.frame_id='world'
            header.stamp=self.get_clock().now().to_msg()
            markers=MarkerArray()
            for index in range(16):
                m=Marker(header=header,ns='entry_light_candidates',id=index,action=Marker.DELETE)
                if index<len(report['candidates']):
                    m.action=Marker.ADD
                    m.type=Marker.LINE_STRIP
                    m.pose.orientation.w=1.
                    m.scale.x=.018
                    m.color.b=1.
                    m.color.g=.5
                    m.color.a=.35
                    m.points=[Point(x=float(x),y=float(y),z=float(z))
                              for x,y,z in report['candidates'][index]['body_xyz']]
                markers.markers.append(m)
            self.candidates_pub.publish(markers)

        def send_next(self,now):
            if not self.backups:
                return
            handoff, report, cloud_wall, fit, route=self.backups[0]
            if (now-cloud_wall>1.5 or self.progress.active_id!=handoff.predecessor or
                    (handoff.predecessor and self.progress.elapsed>=handoff.elapsed) or
                    (not handoff.predecessor and np.linalg.norm(handoff.state.position-self.body)>.10)):
                self.backups.clear()
                return
            if handoff.predecessor and handoff.elapsed - self.progress.elapsed > .65:
                return
            self.backups.pop(0)
            self.last_id+=1
            self.plans[self.last_id]=(fit,route,report,report['body_height'])
            self.pending=(self.last_id,now)
            msg=ScheduledSpline(spline=scan_bspline_message(fit,self.get_clock().now().to_msg(),self.last_id),
                predecessor_id=handoff.predecessor,predecessor_time=handoff.elapsed,
                cap_times=fit.profile['command_cap_t_s'].tolist(),
                xy_speed_caps=fit.profile['command_cap_xy_mps'].tolist())
            self.spline_pub.publish(msg)
            self.phase='waiting_handoff'
            self.event('spline_requested',trajectory_id=self.last_id,previous_id=handoff.predecessor,
                       switch_at_s=handoff.elapsed, duration_s=fit.duration,fit_ms=fit.profile['fit_wall_ms'],
                       collision_refits=fit.profile['collision_refits'], time_scale=fit.profile['time_scale'],
                       peak_jerk_mps3=fit.profile['peak_jerk_mps3'],
                       peak_z_jerk_mps3=fit.profile['peak_z_jerk_mps3'],
                       max_reference_lift_m=fit.profile['max_reference_lift_m'],
                       occupancy_masked_edges=report['occupancy_masked_edges'],
                       start_exemption_until_s=fit.profile.get('start_exemption_until_s'))

        def tick(self):
            now=time.monotonic()
            exclusive=(self.count_publishers('/entry/navigation/hold')==1 and
                       self.count_publishers('/scan/planning/bspline')==1)
            ready=(exclusive and self.progress is not None and self.progress.ready and now-self.progress_wall<.6 and
                   self.body is not None and now-self.body_wall<.6 and self.spline_pub.get_subscription_count()>0)
            fresh=(self.cost_valid and self.cloud is not None and now-self.cloud_wall<2.5 and now-self.cost_wall<2.5)
            active_id=self.progress.active_id if ready else 0
            if not ready or not fresh:
                if not self.hold:
                    self.set_hold(True)
                self.pending=None
                self.backups.clear()
                if self.goal is not None and not self.arrived:
                    self.phase='waiting_inputs'
            elif self.goal is not None and not self.arrived and now-self.goal_wall<120.:
                self.advance(now,active_id)
            elif self.goal is not None:
                if not self.hold:
                    self.set_hold(True)
                self.phase='arrived' if self.arrived else 'execution_timeout'
            for key in list(self.plans):
                if key!=active_id and (self.pending is None or key!=self.pending[0]):
                    del self.plans[key]
            if now-self.last_hold>.15:
                self.set_hold(self.hold)
            if now-self.last_status>.5:
                status=dict(backend='light',trajectory_backend='support_spline',scan_mode=4,
                    scan_mode_ready=bool(ready),cost_fresh=bool(fresh),phase=self.phase,hold=self.hold,
                    max_xy_mps=args.max_xy,max_acc_mps2=args.max_acc,body_reference_height_m=args.body_reference_height,
                    goal_id=self.generation,goal=None if self.goal is None else self.goal.tolist(),
                    body=None if self.body is None else self.body.tolist(),active_trajectory_id=active_id,
                    error=self.error,query_ms=self.last_query_ms,attempts=self.attempt,
                    astar_publications=self.publications)
                atomic_json(args.output/'status.json',status)
                self.status_pub.publish(String(data=json.dumps(status)))
                self.last_status=now

        def advance(self,now,active_id):
            if np.linalg.norm(self.body-self.goal)<=.2:
                self.set_hold(True)
                self.pending=None
                self.backups.clear()
                self.phase='settling_goal'
                if self.speed<.1:
                    if self.arrival_since is None:
                        self.arrival_since=self.get_clock().now().nanoseconds
                    elif self.get_clock().now().nanoseconds-self.arrival_since>=500_000_000:
                        self.phase='arrived'
                        self.arrived=True
                        self.error=None
                        self.event('arrived')
                else:
                    self.arrival_since=None
                return
            self.arrival_since=None
            if self.pending:
                pending_id,sent=self.pending
                queued = self.progress.accepted and self.progress.queued_id==pending_id
                if active_id==pending_id:
                    fit,route,_,_=self.plans[pending_id]
                    self.display(fit,route)
                    self.error=None
                    self.pending=None
                    self.backups.clear()
                    self.phase='following_spline'
                    self.event('spline_activated',trajectory_id=pending_id)
                elif self.progress.result_id==pending_id:
                    if queued:
                        if self.hold:
                            self.set_hold(False)
                    elif not self.progress.accepted:
                        self.error=self.progress.reason
                        self.event('spline_rejected',trajectory_id=pending_id,reason=self.error)
                        self.pending=None
                        self.send_next(now)
                if self.pending and not queued and now-sent>1.:
                    self.error='handoff_timeout'
                    self.pending=None
                    self.backups.clear()
                    if not active_id:
                        self.set_hold(True)
                return
            if active_id and self.progress.elapsed>=self.progress.duration-1e-6:
                # A late or rejected replacement can leave the controller at the old endpoint.
                # Stop there and let the next query use the measured pose from rest.
                self.set_hold(True)
                self.phase='restarting_from_rest'
                return
            if not active_id and not self.hold:
                self.set_hold(True)
                self.phase='stopped'
                self.error=self.progress.reason
                self.event('execution_stopped',reason=self.error)
            if self.future is not None and self.future.done():
                future,self.future=self.future,None
                generation,handoff,cloud_wall=self.query_context
                if generation==self.generation:
                    try:
                        report,fits,rejected=future.result()
                        self.display_candidates(report)
                        self.last_query_ms=1000*report['elapsed_before_report_s']
                        if fits:
                            self.backups=[(handoff,report,cloud_wall,fit,route) for fit,route in fits]
                        else:
                            self.error='; '.join(rejected) or report.get('error')
                            self.event('no_valid_spline',query_id=report['query_id'],reason=self.error)
                            if not active_id:
                                self.phase='no_support_path'
                    except (ValueError,RuntimeError) as error:
                        self.error=str(error)
                        self.event('planning_failed',reason=self.error)
            self.send_next(now)
            if self.future is None and self.pending is None and now-self.last_query>=.5:
                self.start_query(now)

    node=LocalPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.set_hold(True)
        node.pool.shutdown(wait=True)
        node.engine.planner.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        lock.close()


if __name__=='__main__':
    main()
