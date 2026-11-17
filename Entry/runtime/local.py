"""Incremental box-local LIO voxels -> asynchronous timer cost. Independent of navigation."""
from Entry.utils.paths import CONFIG_DIR
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker
import yaml
from common.unitree_gazebo.scripts.frame_config import rotation_matrix
from Entry.utils.client import WorkerClient
from Entry.utils.io import atomic_json, atomic_npy
from Entry.runtime.display import DisplayPublisher
from Entry.utils.local_map import LatestOnlyTimer, ContextVoxelMap, LocalLatency, load_local_config
from Entry.utils.knn import LocalKNNFilter, FilteredSnapshot
from Entry.utils.predict import predict_filtered
from Entry.utils.io import cloud_xyz, seconds, transform_value
from Entry.utils.io import pose_value
from Entry.utils.box import LioBox, box_marker
from Entry.utils.io import cloud_message


class LocalCostNode(Node):
    def __init__(self, cfg, directory, mode='cost'):
        super().__init__('entry_local_cost', parameter_overrides=[Parameter('use_sim_time', value=cfg['use_sim_time'])])
        self.cfg, self.directory, self.mode = cfg, Path(directory), mode
        self.directory.mkdir(parents=True, exist_ok=False)
        config_path = self.directory/'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.map = ContextVoxelMap(cfg['voxel_m'], cfg['max_voxels'], cfg['core_size_m'],
                                   cfg['context_m'], cfg['context_mode'])
        self.schedule = LatestOnlyTimer(cfg['period_s'])
        self.latency = LocalLatency()
        self.buffer = Buffer(); self.listener = TransformListener(self.buffer, self)
        self.anchor = None; self.ready = mode == 'cloud'
        self.worker = None; self.future = None; self.job = None
        self.knn = LocalKNNFilter(cfg.get('knn_filter'))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='entry_local_infer')
        self.received = self.accepted = self.tf_dropped = self.pose_gate_dropped = self.outside_scans = 0
        self.input_wall = self.pose_wall = self.pose_stamp = None
        self.cloud_received_wall = None
        self.pose_received = self.pose_accepted = self.pose_dropped = 0
        self.lio_box = LioBox(cfg['box_size_m'], cfg['voxel_m'], cfg['box_shift_m'])
        self.last_box_publish_wall = self.last_cost_publish_wall = 0.
        self.cache_refreshes = 0
        self.last_result = None; self.result_started_wall = None
        self.last_error = None; self.invalidated = False
        self.ingest_s = 0.; self.last_status_wall = 0.
        self.jobs = 0; self.busy_ticks = 0
        self.phase_samples = deque(maxlen=4096); self.ingest_samples = deque(maxlen=4096)
        self.maintenance_samples = deque(maxlen=4096)
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cloud_pub = self.create_publisher(PointCloud2, '/entry/local/cloud', durable)
        self.cost_pub = self.create_publisher(PointCloud2, '/entry/local/cost_cloud', durable)
        self.removed_pub = self.create_publisher(PointCloud2, '/entry/local/knn_removed', durable)
        self.status_pub = self.create_publisher(String, '/entry/local/status', durable)
        self.box_pub = self.create_publisher(Marker, '/entry/local/box', durable)
        self.display = DisplayPublisher(self, cfg['display_voxel_m'], cfg['display_max_points'],
            cfg['display_hz'], self.directory/'display.json', '/entry/local/rviz')
        self.cloud_sub = self.create_subscription(PointCloud2, cfg['cloud_topic'], self.cloud, sensor)
        self.odom_sub = self.create_subscription(Odometry, cfg['odom_topic'], self.odometry, sensor)
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.poll_timer = self.create_timer(.02, self.poll, clock=self.steady)
        if mode == 'cost':
            self.worker = WorkerClient(cfg, config_path, self.directory)
            self.future = self.pool.submit(self.worker.initialize)
        self.get_logger().info(f'Local {mode}: LIO-owned {cfg["box_size_m"]} m box; output {self.directory}')

    def odometry(self, message):
        self.pose_received += 1
        try:
            if message.header.frame_id != self.cfg['source_frame'] or message.child_frame_id != self.cfg['body_frame']:
                raise ValueError('Unexpected LIO odometry frames')
            stamp = seconds(message.header.stamp)
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            if not np.isfinite(stamp) or age < -.1 or age > self.cfg['max_pose_age_s']:
                raise ValueError('LIO odometry timestamp is stale or ahead of the clock')
            if self.pose_stamp is not None and stamp <= self.pose_stamp:
                return
            position, _ = pose_value(message.pose.pose)
            translation, quaternion = transform_value(self.buffer.lookup_transform(
                self.cfg['frame'], message.header.frame_id, Time.from_msg(message.header.stamp)))
            center = rotation_matrix(quaternion) @ position + translation
            if self.lio_box.update(center):
                self.map.set_bounds(self.lio_box.lower, self.lio_box.upper)
            self.pose_stamp, self.pose_wall = stamp, time.perf_counter()
            self.pose_accepted += 1
        except (ValueError, TransformException):
            self.pose_dropped += 1

    def pose_fresh(self, now):
        return self.pose_wall is not None and now-self.pose_wall <= self.cfg['max_pose_age_s']

    def cloud(self, message):
        started = time.perf_counter(); self.received += 1
        self.cloud_received_wall = started
        if self.last_error is not None:
            return
        if not self.pose_fresh(started):
            self.pose_gate_dropped += 1; return
        if message.header.frame_id != self.cfg['source_frame']:
            self.tf_dropped += 1; return
        try:
            transform = transform_value(self.buffer.lookup_transform(
                self.cfg['frame'], message.header.frame_id, Time.from_msg(message.header.stamp)))
            if self.anchor is not None and (not np.allclose(transform[0], self.anchor[0], rtol=0, atol=1e-7) or
                    not np.allclose(rotation_matrix(transform[1]), rotation_matrix(self.anchor[1]), rtol=0, atol=1e-7)):
                raise ValueError('world <- camera_init changed; restart the local cost session')
            if self.anchor is None:
                self.anchor = transform
                atomic_json(self.directory/'frames.json', dict(frame=self.cfg['frame'], source=message.header.frame_id,
                    translation=transform[0].tolist(), quaternion_xyzw=transform[1].tolist()))
            xyz = (cloud_xyz(message).astype(np.float64) @ rotation_matrix(transform[1]).T + transform[0]).astype(np.float32)
            changed = self.map.add(xyz, seconds(message.header.stamp)); self.accepted += 1
            self.maintenance_samples.append(self.map.last_maintenance_s)
            if self.map.last_in_box_observations:
                self.input_wall = time.perf_counter()
                self.latency.received(seconds(message.header.stamp), started, self.input_wall, changed)
            else:
                self.outside_scans += 1
        except TransformException:
            self.tf_dropped += 1
        except (ValueError, KeyError) as error:
            self.last_error = str(error); self.get_logger().error(self.last_error)
        finally:
            elapsed = time.perf_counter()-started
            self.ingest_s += elapsed; self.ingest_samples.append(elapsed)

    def fresh(self, now):
        return (self.input_wall is not None and now-self.input_wall <= self.cfg['max_input_age_s'] and
                self.pose_fresh(now))

    def result_fresh(self, now):
        if self.last_result is not None and self.last_result[0].revision == self.map.stats()['revision']:
            return self.fresh(now)
        return self.result_started_wall is not None and now-self.result_started_wall <= self.cfg['max_result_age_s']

    def finish(self, publish=True):
        future, self.future = self.future, None
        try:
            result = future.result()
            if not self.ready:
                self.ready = True; return
            snapshot, cost, report = result
            result_ready = time.perf_counter()
            duration = result_ready-self.job['started_wall']
            report.update(input_scope='local_voxel_box', frame_id=self.cfg['frame'], model=self.cfg['model'],
                lower=self.job['lower'], upper=self.job['upper'], pipeline_s=duration,
                queue_depth=0, max_concurrent_jobs=1, voxel_m=self.cfg['voxel_m'])
            report.update(snapshot.metadata)
            self.last_result = (snapshot, cost, report); self.jobs += 1
            self.result_started_wall = self.job['started_wall']
            self.schedule.finish()
            published = None
            if (publish and self.last_error is None and self.map.stats()['points'] and
                    self.fresh(time.perf_counter()) and self.result_fresh(time.perf_counter())):
                stamp = Time(seconds=snapshot.stamp).to_msg()
                self.cost_pub.publish(cloud_message(snapshot.coord, stamp, cost, self.cfg['frame']))
                published = time.perf_counter()
                self.last_cost_publish_wall = published
                removed=(snapshot.raw_context_coord[~snapshot.context_keep] if isinstance(snapshot,FilteredSnapshot) else np.empty((0,3),np.float32))
                maximum=self.cfg.get('knn_filter',{}).get('removed_display_max_points',2000)
                if len(removed)>maximum:removed=removed[np.linspace(0,len(removed)-1,maximum,dtype=int)]
                self.removed_pub.publish(cloud_message(removed,stamp,frame=self.cfg['frame']))
                if not len(cost):self.display.invalidate('cost_cloud')
                self.display.queue('cost_cloud', snapshot.coord, stamp, cost, frame=self.cfg['frame'])
                self.invalidated = not len(cost)
            report['latency'] = self.latency.finish(self.job['latency'], self.job['started_wall'], result_ready, published)
            if published is not None:
                sample = dict(context_copy_s=report['context_copy_s'], query_extract_s=report['query_extract_s'],
                    pipeline_s=duration, result_to_publish_s=published-result_ready,
                    **report.get('timing', {}), **report.get('transport_timing', {}))
                self.phase_samples.append(sample)
                knn=report.get('knn_filter',{})
                if knn.get('enabled'):
                    sample['knn_filter_s']=knn['filter_s'];sample['knn_statistics_s']=knn['statistics_s']
            with (self.directory/'prediction_history.jsonl').open('a') as stream:
                stream.write(json.dumps(report, allow_nan=False)+'\n')
        except Exception as error:
            self.schedule.finish(False); self.last_error = str(error)
            self.get_logger().error(self.last_error)
        finally:
            self.job = None

    def poll(self):
        now = time.perf_counter()
        if self.future is not None and self.future.done():
            self.finish()
        self.display.poll()
        fresh = self.fresh(now)
        info = self.map.stats()
        if self.pose_fresh(now) and self.map.lower is not None and now-self.last_box_publish_wall >= .5:
            self.box_pub.publish(box_marker(self.map.lower, self.map.upper, self.get_clock().now().to_msg(), self.cfg['frame']))
            self.last_box_publish_wall = now
        if (not fresh or not info['points'] or not self.result_fresh(now) or self.last_error is not None) and self.last_result is not None and not self.invalidated:
            stamp = self.get_clock().now().to_msg(); empty = np.empty((0,3), np.float32)
            self.cost_pub.publish(cloud_message(empty, stamp, np.empty(0, np.float32), self.cfg['frame']))
            self.removed_pub.publish(cloud_message(empty,stamp,frame=self.cfg['frame']))
            self.display.invalidate('cost_cloud')
            self.display.queue('cost_cloud', empty, stamp, np.empty(0, np.float32), frame=self.cfg['frame'])
            self.invalidated = True; self.schedule.completed = None
        if (fresh and self.future is None and self.last_result is not None and not self.invalidated and not self.last_error and
                self.last_result[0].revision == info['revision'] and
                now-self.last_cost_publish_wall >= 1/self.cfg['cache_publish_hz']):
            snapshot, cost, _ = self.last_result
            started = time.perf_counter()
            batch = self.latency.snapshot(started, started)
            stamp = Time(seconds=info['stamp']).to_msg()
            self.cost_pub.publish(cloud_message(snapshot.coord, stamp, cost, self.cfg['frame']))
            published = time.perf_counter()
            self.display.queue('cost_cloud', snapshot.coord, stamp, cost, frame=self.cfg['frame'])
            self.last_cost_publish_wall = published
            self.cache_refreshes += 1
            refresh = dict(kind='cache_refresh', revision=snapshot.revision, stamp=info['stamp'],
                           prediction_stamp=snapshot.stamp, points=len(cost),
                           latency=self.latency.finish(batch, started, published, published))
            with (self.directory/'cache_refresh_history.jsonl').open('a') as stream:
                stream.write(json.dumps(refresh, allow_nan=False)+'\n')
        if self.future is not None:
            self.busy_ticks += 1
        elif self.ready and fresh and not self.last_error and info['points'] and self.schedule.start(now, info['revision']):
            snapshot_started = time.perf_counter()
            snapshot = self.map.snapshot()
            timing = self.latency.snapshot(snapshot_started, time.perf_counter())
            stamp = Time(seconds=snapshot.stamp).to_msg()
            self.cloud_pub.publish(cloud_message(snapshot.coord, stamp, frame=self.cfg['frame']))
            self.display.queue('cloud', snapshot.coord, stamp, frame=self.cfg['frame'])
            if self.mode == 'cloud':
                self.schedule.finish()
            else:
                self.job = dict(started_wall=time.perf_counter(), lower=self.map.lower.tolist(), upper=self.map.upper.tolist(), latency=timing)
                self.future = self.pool.submit(predict_filtered,self.worker,self.knn,snapshot)
        if now-self.last_status_wall >= .5:
            self.last_status_wall = now; self.status()

    def status(self, state=None):
        now = time.perf_counter(); info = self.map.stats()
        value = dict(state=state or ('error' if self.last_error else 'waiting_for_model' if not self.ready else
            'waiting_for_pose' if not self.pose_fresh(now) else 'waiting_for_input' if not self.fresh(now) else 'predicting' if self.future else 'idle'),
            **info, model=self.cfg['model'], input_scope='local_voxel_box', frame_id=self.cfg['frame'],
            received_scans=self.received, accepted_scans=self.accepted, dropped_tf=self.tf_dropped,
            dropped_without_pose=self.pose_gate_dropped, scans_without_in_box_points=self.outside_scans,
            box_source='lio_odometry', box_size_m=self.cfg['box_size_m'],
            pose_received=self.pose_received, pose_accepted=self.pose_accepted, pose_dropped=self.pose_dropped,
            pose_stamp=self.pose_stamp, pose_age_s=None if self.pose_wall is None else now-self.pose_wall,
            cloud_received_age_s=None if self.cloud_received_wall is None else now-self.cloud_received_wall,
            cache_refreshes=self.cache_refreshes,
            ingest_s=self.ingest_s, predictions=self.jobs,
            period_s=self.cfg['period_s'], in_flight=int(self.future is not None), pending_snapshots=0,
            input_age_s=None if self.input_wall is None else now-self.input_wall,
            result_age_s=None if self.result_started_wall is None else now-self.result_started_wall,
            lower=None if self.map.lower is None else self.map.lower.tolist(),
            upper=None if self.map.upper is None else self.map.upper.tolist(),
            cost_revision=None if self.last_result is None else self.last_result[0].revision,
            last_prediction=None if self.last_result is None else self.last_result[2],
            cost_points=0 if self.last_result is None else len(self.last_result[0].coord),
            knn_filter=dict(enabled=self.knn.enabled) if self.last_result is None else self.last_result[2].get('knn_filter',dict(enabled=False)),
            valid=bool(state != 'stopped' and info['points'] and self.last_result is not None and len(self.last_result[0].coord) and not self.invalidated and self.fresh(now) and self.result_fresh(now) and not self.last_error),
            phase_timing_ms=self.timing_summary(),
            scan_cost_integration=False, error=self.last_error)
        if rclpy.ok():
            self.status_pub.publish(String(data=json.dumps(value, allow_nan=False)))
        atomic_json(self.directory/'status.json', value)
        return value

    def timing_summary(self):
        def stats(values):
            a = np.asarray(values, dtype=float)*1000
            return dict(samples=len(a), median=None if not len(a) else float(np.median(a)),
                        p95=None if not len(a) else float(np.percentile(a,95)))
        names = {key for row in self.phase_samples for key in row}
        return dict(ingest_s=stats(self.ingest_samples), context_maintenance_s=stats(self.maintenance_samples),
                    **{key:stats([row[key] for row in self.phase_samples if key in row]) for key in sorted(names)})

    def close(self):
        self.destroy_timer(self.poll_timer)
        if self.future is not None:
            self.finish(publish=False)
        self.pool.shutdown(wait=True)
        snapshot = self.map.snapshot()
        atomic_npy(self.directory/'latest_geometry.npy', snapshot.coord)
        atomic_npy(self.directory/'latest_context.npy', snapshot.context_coord)
        atomic_json(self.directory/'context_geometry.json', dict(**snapshot.metadata, revision=snapshot.revision,
                    stamp=snapshot.stamp, frame_id=self.cfg['frame']))
        atomic_json(self.directory/'geometry.json', dict(frame_id=self.cfg['frame'], revision=snapshot.revision,
            stamp=snapshot.stamp, points=len(snapshot.coord)))
        if self.last_result is not None:
            predicted, cost, report = self.last_result
            atomic_npy(self.directory/'prediction_xyz.npy', predicted.coord)
            atomic_npy(self.directory/'latest_cost.npy', cost)
            if isinstance(predicted,FilteredSnapshot):
                atomic_npy(self.directory/'prediction_raw_context.npy',predicted.raw_context_coord)
                atomic_npy(self.directory/'prediction_context_keep.npy',predicted.context_keep)
                atomic_npy(self.directory/'prediction_raw_xyz.npy',predicted.raw_coord)
                atomic_npy(self.directory/'prediction_query_keep.npy',predicted.query_keep)
            atomic_json(self.directory/'latest_prediction.json', dict(**report,
                final_geometry_revision=snapshot.revision, final_cost_complete=predicted.revision==snapshot.revision,
                prediction_scope='knn_retained_roi_points' if isinstance(predicted,FilteredSnapshot) else 'raw_roi_points'))
        self.status('stopped'); self.display.close()
        if self.worker is not None:
            self.worker.close()
        self.map.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=(CONFIG_DIR / 'local.yaml'))
    parser.add_argument('--model', choices=['small', 'tiny'])
    parser.add_argument('--mode', choices=['cloud', 'cost'], default='cost')
    parser.add_argument('--output', type=Path); parser.add_argument('--rviz', action='store_true')
    args, ros_args = parser.parse_known_args()
    cfg = load_local_config(args.config, args.model)
    lock = open(f'/tmp/pure3d_entry_local_{os.getuid()}_domain_{int(os.environ.get("ROS_DOMAIN_ID", "0"))}.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1, 'Local Entry already owns these topics in this ROS domain.\n')
    session = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    output = args.output or Path(cfg['output_root'])/session
    rclpy.init(args=ros_args); node = LocalCostNode(cfg, output, args.mode)
    viewer = subprocess.Popen(['rviz2','-d',str((CONFIG_DIR / 'local.rviz')),
        '--ros-args','-r','__node:=entry_local_rviz','-p','use_sim_time:='+str(cfg['use_sim_time']).lower()]) if args.rviz else None
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close(); node.destroy_node()
        if viewer is not None and viewer.poll() is None:
            viewer.terminate(); viewer.wait(timeout=10)
        if rclpy.ok(): rclpy.shutdown()
        lock.close()


if __name__ == '__main__':
    main()
