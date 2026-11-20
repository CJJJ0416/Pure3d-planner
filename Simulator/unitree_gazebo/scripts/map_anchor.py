#!/usr/bin/env python3
"""Publish world -> map -> camera_init -> body, or the legacy real-robot chain."""
import argparse
from collections import deque
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMessage
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformException, TransformListener

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
from Simulator.unitree_gazebo.scripts.frame_config import DEFAULT_CONFIG, anchor_edges, compose, inverse, load_config, rotation_matrix, transform_from_config


def pose_transform(pose):
    p, q = pose.position, pose.orientation
    t, r = np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])
    if not np.isfinite(np.r_[t, r]).all() or np.linalg.norm(r) < 1e-6:
        raise ValueError('Invalid odometry pose')
    return t, r / np.linalg.norm(r)


def tf_transform(message):
    t, q = message.transform.translation, message.transform.rotation
    return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])


def stamped_transform(parent, child, stamp, transform):
    message = TransformStamped()
    message.header.frame_id, message.child_frame_id = parent, child
    message.header.stamp = stamp
    t, q = transform
    for axis, value in zip('xyz', t):
        setattr(message.transform.translation, axis, float(value))
    for axis, value in zip('xyzw', q):
        setattr(message.transform.rotation, axis, float(value))
    return message


class MapAnchor(Node):
    def __init__(self, config_path):
        super().__init__('faster_lio_frame_bridge')
        self.config = load_config(config_path)
        self.config_path = str(config_path)
        self.frames = self.config['frames']
        self.anchor_parent = self.frames.get('world', self.frames['map'])
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.static = StaticTransformBroadcaster(self)
        self.dynamic = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/faster_lio/Odometry', 20)
        self.path_pub = self.create_publisher(PathMessage, '/faster_lio/path', 5)
        self.path = deque(maxlen=10000)
        self.pending = None
        self.anchored = False
        self.warned = False
        self.source_gids = set()
        self.source_missing_since = None
        self.imu_to_body = None
        if self.config['mount']['source'] == 'parameters':
            mount = self.config['mount']
            self.imu_to_body = inverse(compose(transform_from_config(mount['body_to_lidar']), transform_from_config(mount['lidar_to_imu'])))
        self.subscription = self.create_subscription(
            Odometry, '/faster_lio/raw_odometry', self.receive, qos_profile_sensor_data)
        self.timer = self.create_timer(.1, self.update)
        self.get_logger().info(f'Frame config: {config_path}; anchor parent={self.anchor_parent}; {self.frames["map"]} -> {self.frames["camera"]} -> {self.frames["body"]}')

    def receive(self, message):
        if message.header.frame_id != self.frames['camera']:
            if not self.warned:
                self.get_logger().error('Restart Faster-LIO with the new frame configuration: expected camera_init raw odometry')
                self.warned = True
            return
        if self.imu_to_body is None:
            try:
                self.imu_to_body = tf_transform(self.buffer.lookup_transform(
                    self.frames['imu'], self.frames['body_reference'], rclpy.time.Time()))
            except TransformException:
                return
        camera_imu = pose_transform(message.pose.pose)
        camera_body = compose(camera_imu, self.imu_to_body)
        output = copy.deepcopy(message)
        output.child_frame_id = self.frames['body']
        t, q = camera_body
        for axis, value in zip('xyz', t):
            setattr(output.pose.pose.position, axis, float(value))
        for axis, value in zip('xyzw', q):
            setattr(output.pose.pose.orientation, axis, float(value))
        # Native IEKF covariance: world translation error and local attitude error.
        # Shift the origin to the body and rotate the local attitude error basis.
        lever, rotation = self.imu_to_body
        x, y, z = lever
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        jacobian = np.eye(6)
        jacobian[:3, 3:] = -rotation_matrix(camera_imu[1]) @ skew
        jacobian[3:, 3:] = rotation_matrix(rotation).T
        covariance = np.asarray(message.pose.covariance).reshape(6, 6)
        output.pose.covariance = (jacobian @ covariance @ jacobian.T).ravel().tolist()
        self.odom_pub.publish(output)
        self.dynamic.sendTransform(stamped_transform(self.frames['camera'], self.frames['body'], message.header.stamp, camera_body))
        pose = PoseStamped()
        pose.header, pose.pose = copy.deepcopy(output.header), copy.deepcopy(output.pose.pose)
        self.path.append(pose)
        path = PathMessage()
        path.header, path.poses = copy.deepcopy(output.header), list(self.path)
        self.path_pub.publish(path)
        if not self.anchored and self.pending is None:
            self.pending = output

    def update(self):
        gids = {bytes(p.endpoint_gid) for p in self.get_publishers_info_by_topic('/faster_lio/raw_odometry')}
        if gids and not self.source_gids:
            self.source_gids = gids
        if self.source_gids and not gids & self.source_gids:
            if self.source_missing_since is None:
                self.source_missing_since = time.monotonic()
            elif time.monotonic() - self.source_missing_since > 3.:
                self.get_logger().info('Faster-LIO stopped; exiting frame publisher')
                rclpy.shutdown()
            return
        self.source_missing_since = None
        if self.anchored or self.pending is None:
            return
        message = self.pending
        if self.config['anchor']['source'] == 'fixed':
            transform = transform_from_config(self.config['anchor'])
        elif self.config['anchor']['source'] == 'initial_body':
            transform = compose(transform_from_config(self.config['anchor']), inverse(pose_transform(message.pose.pose)))
        else:
            try:
                truth = self.buffer.lookup_transform(self.anchor_parent, self.frames['body_reference'],
                                                     rclpy.time.Time.from_msg(message.header.stamp))
            except TransformException:
                if self.buffer.can_transform(self.anchor_parent, self.frames['body_reference'], rclpy.time.Time()):
                    newest = self.buffer.lookup_transform(self.anchor_parent, self.frames['body_reference'], rclpy.time.Time())
                    old = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
                    now = newest.header.stamp.sec + newest.header.stamp.nanosec*1e-9
                    if now - old > 5.:
                        self.pending = None
                return
            transform = compose(tf_transform(truth), inverse(pose_transform(message.pose.pose)))
        edges = anchor_edges(self.frames, transform)
        self.static.sendTransform([stamped_transform(parent, child, message.header.stamp, value)
                                   for parent, child, value in edges])
        self.anchored = True
        output = Path(os.environ.get('PURE3D_LIO_OUTPUT', str(WORKSPACE / 'outputs/gazebo_lio')))
        output.mkdir(parents=True, exist_ok=True)
        t, q = transform
        report = dict(parent=self.anchor_parent, child=self.frames['map'] if 'world' in self.frames else self.frames['camera'], body_frame=self.frames['body'],
                      translation=t.tolist(), quaternion_xyzw=q.tolist(), config=self.config_path,
                      anchor_source=self.config['anchor']['source'], mount_source=self.config['mount']['source'],
                      imu_to_body_translation=self.imu_to_body[0].tolist(), imu_to_body_quaternion_xyzw=self.imu_to_body[1].tolist(),
                      anchor_stamp_s=message.header.stamp.sec + message.header.stamp.nanosec*1e-9,
                      static_edges=[dict(parent=p, child=c, translation=v[0].tolist(), quaternion_xyzw=v[1].tolist())
                                    for p, c, v in edges])
        (output / 'map_anchor.json').write_text(json.dumps(report, indent=2) + '\n')
        self.get_logger().info(f'Static anchor from {self.anchor_parent} ready; body pose includes mounting rotation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(os.environ.get('PURE3D_LIO_FRAMES_CONFIG', DEFAULT_CONFIG)))
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = MapAnchor(args.config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
