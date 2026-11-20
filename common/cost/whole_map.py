"""Live scans -> cumulative map -> whole-PCD cost at initial/goal events."""
from common.io import seconds, ros_stamp, transform_value, pose_value, cloud_xyz
from common.paths import ROOT, CONFIG_DIR
import argparse
import fcntl
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import time
from zoneinfo import ZoneInfo
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
import yaml
from common.pointcloud_io import write_pcd
from Simulator.unitree_gazebo.scripts.frame_config import compose, rotation_matrix
from common.cost.client import WorkerClient
from common.mapping.core import BlockMap, load_config
from common.io import atomic_json, atomic_npy
from common.visualization.visualization import block_markers
from common.io import cloud_message
from common.cost.predict import final_prediction
from common.cost.predict import predict_map
from common.visualization.display import DisplayPublisher
from common.planning.goal_trigger import GoalCostTrigger


class EntryNode(Node):
    def __init__(self, config, mode, output):
        super().__init__('pure3d_entry', parameter_overrides=[Parameter('use_sim_time', value=config['use_sim_time'])])
        self.cfg, self.mode, self.directory = config, mode, Path(output)
        self.directory.mkdir(parents=True, exist_ok=False)
        config_path = self.directory / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        self.map = BlockMap(config['core_size_m'])
        self.goals=GoalCostTrigger(config['arrival_tolerance_m'],config['arrival_dwell_s'],config.get('initial_observation_s',3.))
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pending_scans = deque()
        self.latest_odom = self.pending_goal = None
        self.tf_ready = False
        self.anchor = None
        self.fatal = self.worker_error = None
        self.received = self.accepted = self.dropped = self.wrong_frame = 0
        self.callback_s = 0.
        self.start_wall = time.perf_counter()
        self.first_scan_stamp = None
        self.last_snapshot = self.last_result = None
        self.snapshot_future = self.worker_future = None
        self.last_requested_revision = 0
        self.model_ready = False
        self.cost_trigger=config.get('cost_trigger','continuous')
        self.cost_queue=deque()
        self.prediction_history=[]
        self.persisted_revision=None
        self.geometry_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='entry_snapshot')
        self.disk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='entry_storage')
        self.cost_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='entry_cost')
        self.disk_futures = []
        self.worker = None
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud_pub = self.create_publisher(PointCloud2, '/entry/cloud', durable)
        self.cost_pub = self.create_publisher(PointCloud2, '/entry/cost_cloud', durable)
        self.blocks_pub = self.create_publisher(MarkerArray, '/entry/blocks', durable)
        self.status_pub = self.create_publisher(String, '/entry/status', durable)
        self.rviz_display=DisplayPublisher(self,config.get('display_voxel_m',.1),config.get('display_max_points',200000),
            config.get('display_hz',1.),self.directory/'display.json')
        self.last_full_cloud_revision=0
        sensor = QoSProfile(depth=config['input_qos_depth'], reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscriptions_ = [
            self.create_subscription(PointCloud2, config['cloud_topic'], self.cloud, sensor),
            self.create_subscription(Odometry, config['odom_topic'], self.odometry, sensor),
            self.create_subscription(PoseStamped, config['goal_topic'], self.goal, 10)]
        # Background work must continue when bag /clock pauses or reaches EOF.
        # Arrival decisions still use the original odometry timestamps.
        self.background_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.entry_timers = [self.create_timer(.1, self.poll, clock=self.background_clock),
            self.create_timer(1/config['snapshot_hz'], self.request_snapshot, clock=self.background_clock),
            self.create_timer(1/config['visualization_hz'], self.publish_visualization, clock=self.background_clock)]
        self.blocks_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        if mode == 'cost':
            self.worker = WorkerClient(config, config_path, self.directory)
            self.worker_future = self.cost_pool.submit(self.worker.initialize)
        self.get_logger().info(f'Entry {mode}: {config["cloud_topic"]} -> map; output {self.directory}')

    def check_tf(self):
        identity = transform_value(self.buffer.lookup_transform('map', self.cfg['source_frame'], Time()))
        if np.linalg.norm(identity[0]) > 1e-7 or not np.allclose(rotation_matrix(identity[1]), np.eye(3), atol=1e-7):
            raise ValueError('map -> camera_init is not identity; restart the updated TF bridge')
        anchor = transform_value(self.buffer.lookup_transform(self.cfg['world_frame'], 'map', Time()))
        if self.anchor is not None and (not np.allclose(anchor[0], self.anchor[0], atol=1e-7, rtol=0) or
                not np.allclose(rotation_matrix(anchor[1]), rotation_matrix(self.anchor[1]), atol=1e-7, rtol=0)):
            raise ValueError('world -> map changed; restart Entry for the new LIO session')
        if self.anchor is None:
            self.anchor = anchor
            atomic_json(self.directory / 'frames.json', dict(root=self.cfg['world_frame'], frame='map',
                source=self.cfg['source_frame'], world_map_translation=anchor[0].tolist(),
                world_map_quaternion_xyzw=anchor[1].tolist(), map_camera_identity=True))
        self.tf_ready = True

    def in_map(self, header, pose):
        value = pose_value(pose)
        if header.frame_id != 'map':
            transform = transform_value(self.buffer.lookup_transform('map', header.frame_id, Time.from_msg(header.stamp)))
            value = compose(transform, value)
        return value

    def cloud(self, message):
        started = time.perf_counter()
        self.received += 1
        if self.fatal:
            self.dropped += 1
            return
        if message.header.frame_id != self.cfg['source_frame']:
            self.wrong_frame += 1
            return
        if not self.tf_ready:
            if len(self.pending_scans) >= self.cfg['pending_scan_limit']:
                self.pending_scans.popleft()
                self.dropped += 1
            self.pending_scans.append(message)
        else:
            self.ingest(message)
        self.callback_s += time.perf_counter()-started

    def ingest(self, message):
        try:
            stamp = seconds(message.header.stamp)
            if self.map.add(cloud_xyz(message), stamp):
                self.accepted += 1
                if self.first_scan_stamp is None:
                    self.first_scan_stamp = stamp
                self.observe_pose()
        except (ValueError, KeyError) as error:
            self.fatal = str(error)
            self.get_logger().error(self.fatal)

    def odometry(self, message):
        self.latest_odom = message
        if self.tf_ready and not self.fatal:
            self.observe_pose()

    def observe_pose(self):
        if self.latest_odom is None or not self.map.count:
            return
        try:
            position, _ = self.in_map(self.latest_odom.header, self.latest_odom.pose.pose)
            request=self.goals.observe(position,seconds(self.latest_odom.header.stamp),self.map.revision,self.map.count)
            if request:
                snapshot = self.map.snapshot()
                if self.cost_trigger=='goal' and self.mode=='cost':self.cost_queue.append((snapshot,dict(request)))
                atomic_json(self.directory/'goal_events.json',self.goals.report())
                self.get_logger().info(f'Whole PCD {request["trigger"]}: revision {snapshot.revision}, {len(snapshot.coord):,} points')
        except TransformException:
            return
        except ValueError as error:
            self.fatal = str(error)

    def goal(self, message):
        self.pending_goal = message
        self.apply_goal()

    def apply_goal(self):
        if self.pending_goal is None or not self.tf_ready:
            return
        try:
            position, _ = self.in_map(self.pending_goal.header, self.pending_goal.pose)
            self.goals.set_goal(position, seconds(self.pending_goal.header.stamp))
            atomic_json(self.directory/'goal_events.json',self.goals.report())
            self.pending_goal = None
        except (TransformException, ValueError):
            return

    def request_snapshot(self):
        if not self.tf_ready or self.fatal or not self.map.count or self.snapshot_future is not None:
            return
        # Goal requests already freeze their own exact snapshot. Do not copy an
        # entire cumulative map at 2 Hz when no geometry display consumes it.
        if (self.cost_trigger!='continuous' and self.cloud_pub.get_subscription_count()==0 and
                self.rviz_display.publishers['cloud'].get_subscription_count()==0):
            return
        if self.last_snapshot is None or self.last_snapshot.revision != self.map.revision:
            self.snapshot_future = self.geometry_pool.submit(self.map.snapshot)

    def persist_result(self, snapshot, prediction, report):
        temporary = self.directory / 'latest_prediction.tmp.pcd'
        write_pcd(temporary, snapshot.coord, prediction)
        os.replace(temporary, self.directory / 'latest_prediction.pcd')
        atomic_npy(self.directory / 'latest_cost.npy', prediction)
        self.persist_geometry(snapshot)
        atomic_json(self.directory / 'latest_prediction.json', dict(frame_id='map', **report))
        self.prediction_history.append({k:report.get(k) for k in ['prediction_id','goal_id','trigger','revision','stamp','points','input_scope','force_full','timing']})
        atomic_json(self.directory/'prediction_history.json',self.prediction_history)
        self.persisted_revision=snapshot.revision

    def persist_geometry(self,snapshot):
        atomic_npy(self.directory/'latest_geometry.npy',snapshot.coord)
        temporary=self.directory/'latest_geometry.tmp.pcd'
        write_pcd(temporary,snapshot.coord);os.replace(temporary,self.directory/'latest_geometry.pcd')
        atomic_json(self.directory/'geometry.json',dict(frame_id='map',revision=snapshot.revision,
            stamp=snapshot.stamp,points=len(snapshot.coord),xyz='latest_geometry.npy',pcd='latest_geometry.pcd'))

    def accept_result(self, result):
        snapshot, prediction, report = result
        self.last_result = result
        self.rviz_display.queue('cost_cloud',snapshot.coord,ros_stamp(snapshot.stamp),prediction)
        self.rviz_display.poll()
        self.cost_pub.publish(cloud_message(snapshot.coord, ros_stamp(snapshot.stamp), prediction))
        self.disk_futures.append(self.disk_pool.submit(self.persist_result, snapshot, prediction, report))

    def predict(self,snapshot,request=None):
        result=predict_map(self.worker,snapshot,self.anchor,self.cfg.get('inference_frame','map'),
            force_full=self.cfg.get('inference_recompute','full')=='full')
        result[2].update(dict(trigger='final' if request is None else request['trigger'],
            prediction_id=request['prediction_id'] if request else None,goal_id=request['goal_id'] if request else None))
        return result

    def poll(self):
        if not self.fatal:
            try:
                self.check_tf()
            except TransformException:
                self.tf_ready = False
            except ValueError as error:
                self.fatal = str(error)
                self.get_logger().error(self.fatal)
            if self.tf_ready:
                while self.pending_scans and not self.fatal:
                    self.ingest(self.pending_scans.popleft())
                self.apply_goal()
                self.observe_pose()
        if self.snapshot_future is not None and self.snapshot_future.done():
            self.last_snapshot = self.snapshot_future.result()
            self.snapshot_future = None
            self.rviz_display.queue('cloud',self.last_snapshot.coord,ros_stamp(self.last_snapshot.stamp))
        if (self.last_snapshot is not None and self.cloud_pub.get_subscription_count()>0 and
                self.last_full_cloud_revision!=self.last_snapshot.revision):
            self.cloud_pub.publish(cloud_message(self.last_snapshot.coord,ros_stamp(self.last_snapshot.stamp)))
            self.last_full_cloud_revision=self.last_snapshot.revision
        self.rviz_display.poll()
        if self.worker_future is not None and self.worker_future.done():
            future, self.worker_future = self.worker_future, None
            try:
                result = future.result()
                if not self.model_ready:
                    self.model_ready = True
                else:
                    self.accept_result(result)
                self.worker_error = None
            except Exception as error:
                self.worker_error = str(error)
                self.get_logger().error(f'Cost worker: {error}')
        if self.worker and self.model_ready and self.worker_future is None and not self.fatal:
            snapshot,request=(self.cost_queue.popleft() if self.cost_trigger=='goal' and self.cost_queue else
                (self.last_snapshot,None) if self.cost_trigger=='continuous' else (None,None))
            if snapshot is not None and (self.cost_trigger=='goal' or snapshot.revision > self.last_requested_revision):
                self.last_requested_revision = snapshot.revision
                self.worker_future = self.cost_pool.submit(self.predict, snapshot,request)
        remaining = []
        for future in self.disk_futures:
            if future.done():
                try:
                    future.result()
                except Exception as error:
                    self.fatal = f'Storage failed: {error}'
            else:
                remaining.append(future)
        self.disk_futures = remaining

    def status(self):
        current = self.map.metadata()
        elapsed = (self.map.stamp-self.first_scan_stamp) if self.first_scan_stamp is not None else 0.
        result = self.last_result[2] if self.last_result else None
        return dict(state='error' if self.fatal or self.worker_error else ('waiting_tf' if not self.tf_ready else 'running'),
            mode=self.mode, frame_id='map', source_frame=self.cfg['source_frame'],
            revision=current['revision'], points=current['points'], blocks=len(current['blocks']),
            last_scan_stamp=self.map.stamp,
            completed_goals=self.goals.completed_goals,initial_ready=self.goals.initial_ready,
            goal_sequence=self.goals.goal_sequence,received_frames=self.received, accepted_frames=self.accepted,
            dropped_pending_frames=self.dropped, wrong_frame_messages=self.wrong_frame,
            invalid_points=self.map.invalid_points, duplicate_frames=self.map.duplicate_frames,
            received_sim_hz=(self.accepted-1)/elapsed if elapsed > 0 else None,
            callback_total_s=self.callback_s, pending_scans=len(self.pending_scans),
            worker_ready=self.model_ready, worker_busy=self.worker_future is not None,
            cost_revision=result['revision'] if result else None,
            cost_trigger=self.cost_trigger,inference_frame=self.cfg.get('inference_frame','map'),
            pending_cost_requests=len(self.cost_queue),input_scope='whole_pcd',
            latest_request_id=self.goals.report()['latest_request_id'],
            rviz_display=self.rviz_display.stats,
            last_request_revision=self.goals.report()['last_request_revision'],
            pending_revisions=current['revision']-(result['revision'] if result else 0),
            cost_report=result, error=self.fatal, worker_error=self.worker_error,
            wall_s=time.perf_counter()-self.start_wall, output=str(self.directory))

    def publish_visualization(self):
        stamp = self.get_clock().now().to_msg()
        self.blocks_pub.publish(block_markers(self.map.metadata()['blocks'], self.cfg['core_size_m'], stamp))
        value = self.status()
        self.status_pub.publish(String(data=json.dumps(value, allow_nan=False)))
        atomic_json(self.directory / 'status.json', value)

    def close(self):
        for timer in self.entry_timers:
            timer.cancel()
        self.rviz_display.close()
        self.geometry_pool.shutdown(wait=True)
        self.cost_pool.shutdown(wait=True)
        self.disk_pool.shutdown(wait=True)
        for future in self.disk_futures:
            try:
                future.result()
            except Exception as error:
                self.fatal = f'Storage failed: {error}'
        snapshot = self.map.snapshot()
        if self.worker:
            try:
                self.last_result = final_prediction(self, self.worker_future,
                    self.model_ready, self.last_result, snapshot)
                self.model_ready = True
                self.worker_error = None
                if self.last_result is not None and self.persisted_revision!=snapshot.revision:
                    self.persist_result(*self.last_result)
            except Exception as error:
                self.worker_error = f'Final cost incomplete: {error}'
        self.worker_future = None
        if self.persisted_revision!=snapshot.revision:self.persist_geometry(snapshot)
        atomic_json(self.directory/'goal_events.json',self.goals.report())
        complete = bool(self.last_result is not None and self.last_result[0].revision == snapshot.revision
                        and len(self.last_result[1]) == len(snapshot.coord) and not self.worker_error)
        atomic_json(self.directory / 'status.json', dict(self.status(), state='stopped',
                    final_cost_complete=complete if self.mode == 'cost' else None))
        if self.worker:
            self.worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=(CONFIG_DIR / 'whole_map.yaml'))
    parser.add_argument('--mode', choices=['blocks', 'cost'], default='cost')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rviz', action='store_true')
    args, ros_args = parser.parse_known_args()
    # Fixed public topics mean only one Entry may own a given local ROS domain.
    lock_path = Path('/tmp') / f'pure3d_entry_{os.getuid()}_domain_{int(os.environ.get("ROS_DOMAIN_ID", "0"))}.lock'
    instance_lock = lock_path.open('a')
    try:
        fcntl.flock(instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1, 'Entry is already running in this ROS domain; use the existing Entry terminal.\n')
    config = load_config(args.config)
    for key in ('checkpoint', 'output_root'):
        config[key] = str((ROOT / config[key]).resolve())
    session = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    output = (args.output or Path(config['output_root']) / session).resolve()
    rclpy.init(args=ros_args)
    node = EntryNode(config, args.mode, output)
    viewer = subprocess.Popen(['rviz2', '-d', str((CONFIG_DIR / 'whole_map.rviz')),
                              '--ros-args', '-r', '__node:=entry_rviz',
                              '-p', 'use_sim_time:='+str(config['use_sim_time']).lower()]) if args.rviz else None
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            viewer.wait(timeout=10)
        if rclpy.ok():
            rclpy.shutdown()
        instance_lock.close()


if __name__ == '__main__':
    main()
