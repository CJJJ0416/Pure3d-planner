"""Human goal -> cost-only surface/DC A* -> GPMP -> existing SCAN tracker."""
from common.paths import ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path as PathMessage
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from planner.light_planner import PlannerConfig
from planner.light_planner.surface import SurfaceOptions
from common.io import atomic_json
from common.mapping.core import load_config
from common.io import cloud_xyz, transform_value, seconds
from common.planning.navigation_core import NavigationPlanner, ArrivalHold
from common.visualization.visualization import marker, point
from Simulator.unitree_gazebo.scripts.frame_config import rotation_matrix


class Navigation(Node):
    def __init__(self,config,output):
        super().__init__('entry_manual_navigation',parameter_overrides=[Parameter('use_sim_time',value=config['use_sim_time'])])
        self.directory=Path(output);self.directory.mkdir(parents=True,exist_ok=False)
        self.cfg=config;self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        planner_config=json.loads((ROOT/'planner/light_planner/config/dc.json').read_text())
        planner_config.update(dict(zip(['clearance_min','slope_max','step_max','stand_ratio'],config['condition'])))
        self.engine=NavigationPlanner(PlannerConfig(**planner_config),self.directory,
            scan_path_source=config.get('scan_path_source','gpmp'),backend=config.get('planner_backend','dc'),
            surface_options=SurfaceOptions(slice_dh=config.get('surface_slice_dh',.5)))
        atomic_json(self.directory/'config.json',dict(planner=asdict(self.engine.config),entry=config,body_height=.4))
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='global_astar')
        self.future=None;self.pending=False;self.generation=0;self.attempts=0
        self.state='waiting_initial_cost';self.error=None;self.entry={};self.cost=None;self.pending_cloud=None
        self.body=None;self.goal=None;self.goal_position=None;self.marker_ready=False
        self.startup_goal=None
        self.hold=True;self.arrival=ArrivalHold(.2,3.);self.events=[]
        self.wait_after_prediction=None;self.path_stamp=None;self.ack_time=None;self.published_wall=None
        self.steady=Clock(clock_type=ClockType.STEADY_TIME)
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub=self.create_publisher(PathMessage,'/pct_path',durable)
        self.astar_pub=self.create_publisher(PathMessage,'/entry/astar_path',durable)
        self.gpmp_pub=self.create_publisher(PathMessage,'/entry/gpmp_path',durable)
        self.surface_pub=self.create_publisher(PointCloud2,'/entry/surface_cloud',durable)
        self.displayed_surface_key=None
        self.goal_pub=self.create_publisher(PoseStamped,'/scan/goal',10)
        self.hold_pub=self.create_publisher(Bool,'/entry/navigation/hold',durable)
        self.status_pub=self.create_publisher(String,'/entry/navigation/status',durable)
        self.markers_pub=self.create_publisher(MarkerArray,'/entry/navigation/markers',durable)
        self.subs=[self.create_subscription(PointCloud2,'/entry/cost_cloud',self.cloud,durable),
            self.create_subscription(String,'/entry/status',self.entry_status,durable),
            self.create_subscription(PoseStamped,'/entry/navigation/goal',self.request_goal,10),
            self.create_subscription(PathMessage,'/scan/initial_path',self.scan_ack,durable)]
        self.server=InteractiveMarkerServer(self,'entry_navigation_controls')
        self.navigation_timers=[self.create_timer(.1,self.poll,clock=self.steady),
            self.create_timer(.2,self.heartbeat,clock=self.steady),
            self.create_timer(.5,self.publish_status,clock=self.steady)]
        self.heartbeat();self.publish_status()

    def event(self,name,**values):
        self.events.append(dict(event=name,stamp=self.get_clock().now().nanoseconds*1e-9,**values))
        atomic_json(self.directory/'execution.json',dict(state=self.state,events=self.events))

    def entry_status(self,msg):
        self.entry=json.loads(msg.data)

    def cloud(self,msg):
        self.pending_cloud=msg

    def decode_cloud(self):
        msg=self.pending_cloud
        if msg is None:return
        try:
            coord=cloud_xyz(msg)
            f=next(f for f in msg.fields if f.name=='trav_cost')
            if f.datatype!=7 or f.count!=1:raise ValueError('Expected scalar float32 trav_cost')
            dtype=np.dtype(dict(names=['cost'],formats=[('>' if msg.is_bigendian else '<')+'f4'],offsets=[f.offset],itemsize=msg.point_step))
            raw=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.row_step,msg.point_step))
            values=raw['cost'].ravel().copy()
            if not len(coord) or not np.isfinite(coord).all() or not np.isfinite(values).all():raise ValueError('Invalid cost cloud')
            if msg.header.frame_id!='world':
                transform=transform_value(self.buffer.lookup_transform('world',msg.header.frame_id,Time.from_msg(msg.header.stamp)))
                coord=(coord@rotation_matrix(transform[1]).T+transform[0]).astype(np.float32)
            self.cost=dict(xyz=coord,values=values,stamp=seconds(msg.header.stamp))
            self.pending_cloud=None
        except TransformException:return
        except (ValueError,KeyError,StopIteration) as error:
            self.error=str(error);self.pending_cloud=None

    def ready_cost(self):
        report=self.entry.get('cost_report') or {}
        return bool(self.cost is not None and self.entry.get('cost_trigger')=='goal' and
            self.entry.get('inference_frame')=='world' and self.entry.get('input_scope')=='whole_pcd' and
            self.entry.get('last_request_revision') is not None and
            (self.entry.get('cost_revision') or 0)>=self.entry['last_request_revision'] and
            report.get('prediction_id')==self.entry.get('latest_request_id') and
            abs(report.get('stamp',-1)-self.cost['stamp'])<1e-6 and not self.entry.get('pending_cost_requests'))

    def make_marker(self,name,position,movable):
        item=InteractiveMarker();item.name=name;item.header.frame_id='world';item.scale=.8
        item.description='GOAL body XYZ: drag and release' if movable else 'START: current base'
        item.pose.position=point(position);item.pose.orientation.w=1.
        sphere=Marker(type=Marker.SPHERE);sphere.pose.orientation.w=1.;sphere.scale.x=sphere.scale.y=sphere.scale.z=.25
        sphere.color.r,sphere.color.g,sphere.color.b,sphere.color.a=(1.,.15,.7,1.) if movable else (0.,.8,1.,1.)
        control=InteractiveMarkerControl();control.always_visible=True;control.markers.append(sphere);item.controls.append(control)
        if movable:
            for axis,q in [('x',(0.,0.,0.,1.)),('y',(0.,0.,2**-.5,2**-.5)),('z',(0.,-2**-.5,0.,2**-.5))]:
                c=InteractiveMarkerControl(name='move_'+axis,interaction_mode=InteractiveMarkerControl.MOVE_AXIS,orientation_mode=InteractiveMarkerControl.FIXED)
                c.orientation.x,c.orientation.y,c.orientation.z,c.orientation.w=q;item.controls.append(c)
        self.server.insert(item,feedback_callback=self.feedback)

    def feedback(self,msg):
        if msg.marker_name!='end_pos' or msg.event_type!=InteractiveMarkerFeedback.MOUSE_UP:return
        target=PoseStamped();target.header=msg.header;target.pose=msg.pose;self.request_goal(target)

    def request_goal(self,msg):
        try:
            if not msg.header.frame_id:raise ValueError('Goal frame is empty')
            target=msg if msg.header.frame_id=='world' else self.buffer.transform(msg,'world')
            p=target.pose.position;position=np.array([p.x,p.y,p.z])
            if not np.isfinite(position).all():raise ValueError('Invalid goal')
            if self.goal is not None and np.linalg.norm(position-self.goal)<.01 and self.state!='failed':return
            self.goal=position;self.generation+=1;self.pending=True;self.hold=True;self.error=None
            empty=PathMessage();empty.header.frame_id='world';empty.header.stamp=self.get_clock().now().to_msg()
            self.astar_pub.publish(empty);self.gpmp_pub.publish(empty)
            if self.wait_after_prediction is None:self.state='queued_goal'
            self.event('goal_requested',generation=self.generation,goal_body_world=position.tolist())
        except (TransformException,ValueError) as error:self.error=str(error)

    def heartbeat(self):self.hold_pub.publish(Bool(data=self.hold))

    def scan_ack(self,msg):
        if self.path_stamp is not None and msg.header.stamp==self.path_stamp and msg.poses:
            if self.ack_time is None:self.ack_time=time.monotonic()

    def path_message(self,body,stamp):
        path=PathMessage();path.header.frame_id='world';path.header.stamp=stamp
        for i,xyz in enumerate(body):
            pose=PoseStamped();pose.header=path.header;pose.pose.position=point(xyz)
            direction=body[min(i+1,len(body)-1)]-body[max(0,i-1)];yaw=float(np.arctan2(direction[1],direction[0]))
            pose.pose.orientation.z=np.sin(yaw/2);pose.pose.orientation.w=np.cos(yaw/2);path.poses.append(pose)
        return path

    def publish_candidates(self,record):
        stamp=self.get_clock().now().to_msg()
        surface_key=(record['map_key'],record.get('prediction_id'))
        if self.engine.backend=='surface' and self.displayed_surface_key!=surface_key:
            from common.visualization.display_sampling import display_rows
            from common.io import cloud_message
            planner=self.engine.planner
            rows,_=display_rows(planner.node_xyz,planner.node_cost,
                voxel_m=self.cfg.get('display_voxel_m',.1),max_points=self.cfg.get('display_max_points',200000))
            self.surface_pub.publish(cloud_message(planner.node_xyz[rows],stamp,
                planner.node_cost[rows],frame='world'))
            self.displayed_surface_key=surface_key
        for key,publisher in [('astar_world',self.astar_pub),('gpmp_world',self.gpmp_pub)]:
            filename=record.get('artifacts',{}).get(key)
            if filename:publisher.publish(self.path_message(np.load(filename),stamp))
            elif key=='gpmp_world' and record['path_source']=='astar':
                publisher.publish(self.path_message(np.empty((0,3)),stamp))

    def publish_route(self,record,body):
        stamp=self.get_clock().now().to_msg();path=self.path_message(body,stamp)
        self.goal_position=body[-1].copy();self.path_stamp=stamp;self.ack_time=None
        self.path_pub.publish(path);self.goal_pub.publish(path.poses[-1]);self.published_wall=time.monotonic()
        self.publication_completed_goals=self.entry.get('completed_goals',0);self.arrival.reset();self.state='waiting_scan_path'
        self.path_prediction_id=self.running_prediction_id;self.event('path_published',**record)
        array=MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        title=marker('global_path_label',0,Marker.TEXT_VIEW_FACING,stamp);title.header.frame_id='world'
        title.pose.position=point(body[-1]+[0.,0.,.7]);title.scale.z=.3
        title.color.r,title.color.g,title.color.b,title.color.a=1.,.4,.8,1.
        title.text=f'SCAN <- {record["path_source"].upper()}'
        array.markers.append(title);self.markers_pub.publish(array)

    def poll(self):
        try:
            transform=self.buffer.lookup_transform('world','base',Time());self.body=transform_value(transform)[0]
            if self.get_clock().now().nanoseconds*1e-9-seconds(transform.header.stamp)>1.:
                self.hold=True;self.error='Stale world -> base TF';return
        except TransformException:return
        if self.startup_goal is not None:
            target=PoseStamped();target.header.frame_id='world';target.header.stamp=self.get_clock().now().to_msg()
            target.pose.position=point(self.startup_goal);target.pose.orientation.w=1.
            self.startup_goal=None;self.request_goal(target)
        if not self.marker_ready:
            self.make_marker('start_pos',self.body,False)
            self.make_marker('end_pos',self.body+[1.5,0.,0.] if self.goal is None else self.goal,True);self.marker_ready=True
        pose=Pose();pose.position=point(self.body);pose.orientation.w=1.;self.server.setPose('start_pos',pose);self.server.applyChanges()
        owners=self.get_publishers_info_by_topic('/pct_path')
        if any(p.node_name!=self.get_name() for p in owners):
            self.hold=True;self.state='conflicting_path_publisher';self.error='Stop the other /pct_path publisher';return
        self.decode_cloud()
        if self.future is not None and self.future.done():
            future,self.future=self.future,None
            try:
                record,body=future.result()
                if self.running_generation==self.generation:
                    self.publish_candidates(record)
                    if not record['success']:self.state='failed';self.error=record['error']
                    elif (not self.ready_cost() or record['map_key']!=self.entry.get('cost_revision') or
                        self.running_prediction_id!=(self.entry.get('cost_report') or {}).get('prediction_id')):self.pending=True
                    elif np.linalg.norm(np.array(record['start_body'])-self.body)>.3:self.pending=True
                    elif np.linalg.norm(body[-1]-self.body)<.3:self.state='failed';self.error='Goal is already at current base'
                    else:self.publish_route(record,body)
            except Exception as error:self.state='failed';self.error=str(error)
        if self.state=='waiting_scan_path':
            if self.ack_time is not None and time.monotonic()-self.ack_time>.2:
                self.hold=False;self.state='following'
            elif time.monotonic()-self.published_wall>10.:
                self.state='failed';self.error='SCAN initial_path acknowledgement timed out';self.hold=True
        if self.state=='following':
            inside=np.linalg.norm(self.body-self.goal_position)<=self.arrival.tolerance
            self.hold=bool(inside)
            stamp=seconds(transform.header.stamp)
            if self.arrival.observe(self.body,self.goal_position,stamp):
                self.hold=True;self.state='waiting_goal_cost';self.wait_after_prediction=self.path_prediction_id
                self.event('arrived_and_held',goal_body_world=self.goal_position.tolist(),body_world=self.body.tolist(),dwell_s=self.arrival.dwell)
            elif time.monotonic()-self.published_wall>180.:
                self.hold=True;self.state='failed';self.error='Goal timeout; arrival was not recorded'
        ready=self.ready_cost()
        if (self.wait_after_prediction is not None and ready and
            self.entry['latest_request_id']>self.wait_after_prediction and
            self.entry.get('completed_goals',0)>self.publication_completed_goals):
            self.wait_after_prediction=None;self.state='awaiting_goal';self.event('goal_cost_ready',revision=self.entry['cost_revision'])
        if self.state=='waiting_initial_cost' and ready:self.state='awaiting_goal';self.event('initial_cost_ready',revision=self.entry['cost_revision'])
        if self.pending and self.future is None and ready and self.wait_after_prediction is None:
            if self.path_pub.get_subscription_count()==0:self.state='waiting_scan_subscriber';return
            self.pending=False;self.hold=True;self.state='planning';self.attempts+=1;self.running_generation=self.generation
            self.running_prediction_id=self.entry['cost_report']['prediction_id']
            self.future=self.pool.submit(self.engine.compute,self.cost['xyz'],self.cost['values'],self.entry['cost_revision'],
                self.body.copy(),self.goal.copy(),self.attempts,prediction_id=self.running_prediction_id)

    def publish_status(self):
        value=dict(state=self.state,error=self.error,hold=self.hold,generation=self.generation,
            body_world=None if self.body is None else self.body.tolist(),goal_world=None if self.goal is None else self.goal.tolist(),
            cost_revision=self.entry.get('cost_revision'),last_request_revision=self.entry.get('last_request_revision'),
            completed_goals=self.entry.get('completed_goals',0),input_scope='whole_pcd',
            scan_path_source=self.engine.scan_path_source,
            planner_backend=self.engine.backend,planner=self.engine.progress,
            displayed_surface_version=self.displayed_surface_key,
            dc=self.engine.progress if self.engine.backend=='dc' else None,
            frame_id='world',output=str(self.directory),automatic_goal_selection=False)
        atomic_json(self.directory/'status.json',value)
        if rclpy.ok():self.status_pub.publish(String(data=json.dumps(value)))

    def close(self):
        self.hold=True
        if rclpy.ok():self.heartbeat()
        for timer in self.navigation_timers:timer.cancel()
        if rclpy.ok():self.server.shutdown()
        self.pool.shutdown(wait=True);self.state='stopped';self.publish_status()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'Entrance/config/navigation.yaml')
    parser.add_argument('--planner-backend',choices=['surface','dc'],help='Override planner_backend for this navigation process')
    parser.add_argument('--output',type=Path);parser.add_argument('--rviz',action='store_true')
    parser.add_argument('--goal',nargs=3,type=float,help='Resume the already selected world body goal after a navigation restart')
    args,ros_args=parser.parse_known_args()
    config=load_config(args.config)
    if args.planner_backend is not None:config['planner_backend']=args.planner_backend
    if config.get('cost_trigger')!='goal' or config.get('inference_frame')!='world':parser.error('Use goal-triggered, gravity-aligned Entry config')
    lock=Path(f'/tmp/pure3d_navigation_{os.getuid()}_domain_{os.environ.get("ROS_DOMAIN_ID","0")}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    output=(args.output or ROOT/'outputs/entry_navigation'/stamp).resolve()
    rclpy.init(args=ros_args);node=Navigation(config,output)
    if args.goal is not None:node.startup_goal=np.asarray(args.goal)
    viewer=subprocess.Popen(['rviz2','-d',str(ROOT/'Entrance/config/navigation.rviz'),'--ros-args','-r','__node:=entry_navigation_rviz','-p','use_sim_time:='+str(config['use_sim_time']).lower()]) if args.rviz else None
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.close();node.destroy_node()
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        if rclpy.ok():rclpy.shutdown()
        lock.close()


if __name__=='__main__':main()
