"""ROS 2 real-time replay into an unmodified external Faster-LIO process.

The local LIO's PointBodyToWorld drops curvature. Its dense output retains the
input's strictly time-sorted row order. We join by unique input offsets and
verify exact row count AND the preserved synthetic reflectivity sequence.
There is no nearest-neighbour provenance assignment.
Run with system Python after sourcing ROS and the livox_ws underlay.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from livox_ros_driver2.msg import CustomMsg, CustomPoint


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def stamp(seconds):
    ns = int(round(seconds * 1e9))
    return dict(sec=ns // 1000000000, nanosec=ns % 1000000000)


def seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def cloud_fields(msg):
    fields = [f for f in msg.fields if f.name in ['x', 'y', 'z', 'intensity', 'curvature']]
    dtype = np.dtype(dict(names=[f.name for f in fields], formats=['<f4'] * len(fields),
                         offsets=[f.offset for f in fields], itemsize=msg.point_step))
    rows = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    return {name: rows[name].copy() for name in dtype.names}


class Replay(Node):
    def __init__(self, folder, config):
        super().__init__('pure3d_scan_replay')
        self.folder, self.config = folder, config
        self.records = json.loads((folder / 'scans.json').read_text())
        self.end_times = np.array([r['end_s'] for r in self.records])
        self.registered = folder / 'registered'
        self.registered.mkdir(exist_ok=True)
        self.frames, self.odometry, self.callback_s = {}, [], 0.
        qos = QoSProfile(depth=2048, reliability=ReliabilityPolicy.RELIABLE)
        self.lidar = self.create_publisher(CustomMsg, '/pure3d_scan/lidar', qos)
        self.imu = self.create_publisher(Imu, '/pure3d_scan/imu', qos)
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.subscribers = [
            self.create_subscription(PointCloud2, '/pure3d_lio/cloud_registered', self.cloud, 1024),
            self.create_subscription(Odometry, '/pure3d_lio/Odometry', self.odom, 1024)]
        self.raw = {}

    def odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odometry.append([seconds(msg.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def cloud(self, msg):
        started = time.perf_counter()
        end = seconds(msg.header) - self.config['time_origin_s']
        frame = int(np.argmin(abs(self.end_times - end)))
        assert abs(self.end_times[frame] - end) < 1e-6, (frame, end)
        raw = self.raw.pop(frame)
        fields = cloud_fields(msg)
        assert len(fields['x']) == len(raw['offset_ns']), (frame, len(fields['x']), len(raw['offset_ns']))
        assert np.all(np.diff(raw['offset_ns'].astype(np.int64)) > 0)
        np.testing.assert_array_equal(fields['intensity'], raw['reflectivity'])
        xyz = np.column_stack([fields[k] for k in ['x', 'y', 'z']])
        assert np.isfinite(xyz).all(), frame
        np.savez(self.registered / f'{frame:05d}.npz', coord=xyz, ray_id=raw['ray_id'],
                 offset_ns=raw['offset_ns'], stamp_s=seconds(msg.header))
        self.frames[frame] = dict(frame=frame, stamp_s=seconds(msg.header), points=len(xyz),
            point_mapping='unique input offset_ns order; dense count and reflectivity sequence checked')
        self.callback_s += time.perf_counter() - started

    def run(self):
        imu = dict(np.load(self.folder / 'imu.npz'))
        origin = self.config['time_origin_s']
        nimu = 0
        start = time.perf_counter()
        maximum_lag = 0.
        for row in self.records:
            raw = dict(np.load(self.folder / row['path']))
            frame = row['frame']
            self.raw[frame] = raw
            message = CustomMsg()
            message.header.frame_id = 'scan_lidar'
            for name, value in stamp(origin + row['start_s']).items():
                setattr(message.header.stamp, name, value)
            message.timebase = int(round((origin + row['start_s']) * 1e9))
            message.point_num = len(raw['coord'])
            message.points = [CustomPoint(offset_time=int(offset), x=float(p[0]), y=float(p[1]),
                z=float(p[2]), reflectivity=int(reflectivity), tag=16, line=0)
                for p, offset, reflectivity in zip(raw['coord'], raw['offset_ns'], raw['reflectivity'])]
            # Deliver the IMU bracket before the completed LiDAR frame.
            end = row['end_s'] + .006
            while nimu < len(imu['time']) and imu['time'][nimu] <= end:
                event_time = float(imu['time'][nimu])
                delay = event_time - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)
                maximum_lag = max(maximum_lag, -delay)
                item = Imu()
                item.header.frame_id = 'scan_imu'
                for name, value in stamp(origin + event_time).items():
                    setattr(item.header.stamp, name, value)
                for axis, value in zip('xyz', imu['acceleration'][nimu]):
                    setattr(item.linear_acceleration, axis, float(value))
                for axis, value in zip('xyz', imu['angular_velocity'][nimu]):
                    setattr(item.angular_velocity, axis, float(value))
                item.orientation_covariance[0] = -1.
                self.imu.publish(item)
                clock = Clock()
                clock.clock = item.header.stamp
                self.clock.publish(clock)
                rclpy.spin_once(self, timeout_sec=0.)
                nimu += 1
            self.lidar.publish(message)
            rclpy.spin_once(self, timeout_sec=0.)
            if frame % 50 == 0:
                print('REPLAY', frame, 'registered', len(self.frames), flush=True)
        # Flush the final IMU bracket, then drain outputs without republishing scans.
        for _ in range(3):
            item = Imu()
            item.header.frame_id = 'scan_imu'
            for name, value in stamp(origin + float(imu['time'][nimu])).items():
                setattr(item.header.stamp, name, value)
            for axis, value in zip('xyz', imu['acceleration'][nimu]):
                setattr(item.linear_acceleration, axis, float(value))
            for axis, value in zip('xyz', imu['angular_velocity'][nimu]):
                setattr(item.angular_velocity, axis, float(value))
            self.imu.publish(item)
            nimu += 1
        drain = time.perf_counter()
        while time.perf_counter() - drain < 5:
            rclpy.spin_once(self, timeout_sec=.05)
        wall = time.perf_counter() - start
        np.save(self.folder / 'lio_odometry.npy', np.asarray(self.odometry))
        save_json(self.folder / 'registered.json', [self.frames[k] for k in sorted(self.frames)])
        missing = [row['frame'] for row in self.records if row['frame'] not in self.frames]
        save_json(self.folder / 'replay.json', dict(input_frames=len(self.records),
            registered_frames=len(self.frames), missing_frames=missing, imu_messages=nimu,
            intended_playback_rate=1., wall_s=wall, maximum_schedule_lag_s=maximum_lag,
            registered_export_s=self.callback_s, odometry_messages=len(self.odometry),
            provenance='unmodified dense LIO preserves strictly time-sorted input rows; output curvature is dropped',
            external_pcd_save_enabled=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    folder = Path(config['output']).resolve() / args.profile
    ws = Path(config['lio_workspace'])
    parameters = yaml.safe_load((ws / 'src/faster-lio/config/mid360_sim.yaml').read_text())
    p = parameters['/**']['ros__parameters']
    p['common'].update(lid_topic='/pure3d_scan/lidar', imu_topic='/pure3d_scan/imu', time_sync_en=False)
    p['mapping'].update(extrinsic_T=[0., 0., 0.], extrinsic_R=np.eye(3).ravel().tolist(), extrinsic_est_en=False)
    p['preprocess']['blind'] = max(config['min_range_m'], config['blind_zone_m'])
    p.update(point_filter_num=1, filter_size_surf=.1, filter_size_map=.1, path_save_en=False)
    p['publish'].update(dense_publish_en=True, scan_publish_en=True, scan_bodyframe_pub_en=False,
                        path_publish_en=False, tf_world_frame='lio_odom', tf_imu_frame='lio_imu')
    p['pcd_save']['pcd_save_en'] = False
    param_path = folder / 'lio.actual.yaml'
    param_path.write_text(yaml.safe_dump(parameters))
    rclpy.init()
    node = Replay(folder, config)
    command = [str(ws / 'install/faster_lio/lib/faster_lio/run_mapping_online'), '--ros-args',
               '-r', '__ns:=/pure3d_lio', '--params-file', str(param_path)]
    log_path = folder / 'faster_lio.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.perf_counter() + 20
            while node.lidar.get_subscription_count() == 0 or node.imu.get_subscription_count() == 0:
                if process.poll() is not None:
                    raise RuntimeError(f'Faster-LIO exited {process.returncode}: {log_path}')
                if time.perf_counter() > deadline:
                    raise TimeoutError(f'Faster-LIO subscriptions not ready: {log_path}')
                rclpy.spin_once(node, timeout_sec=.1)
            node.run()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=15)
            node.destroy_node()
            rclpy.shutdown()
    save_json(folder / 'lio_process.json', dict(returncode=process.returncode, command=command,
        log=str(log_path), source_workspace=str(ws)))
    print('REPLAY_COMPLETE', args.profile, flush=True)


if __name__ == '__main__':
    main()
