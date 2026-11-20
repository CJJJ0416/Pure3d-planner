#!/usr/bin/env python3
"""Wait for a stationary, gravity-bearing MID360 IMU before LIO initialization."""
import argparse
from collections import deque
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=120.,help='Wall-time limit in seconds; 0 waits until stationary or Ctrl+C')
    parser.add_argument('--stable-seconds', type=float, default=1.)
    parser.add_argument('--topic', default='/livox/imu')
    parser.add_argument('--gravity-norm', type=float, default=9.81, help='Stationary |acc| in the driver units; 9.81 for m/s^2, 1 for g')
    args = parser.parse_args()
    if not math.isfinite(args.gravity_norm) or args.gravity_norm <= 0:
        parser.error('--gravity-norm must be finite and positive')
    if not math.isfinite(args.timeout) or args.timeout<0:parser.error('--timeout must be finite and nonnegative')
    rclpy.init()
    node = Node('wait_for_stationary_mid360')
    samples = deque()

    def receive(message):
        a, g = message.linear_acceleration, message.angular_velocity
        acceleration = (a.x, a.y, a.z)
        norm = math.sqrt(sum(v*v for v in acceleration))
        gyro_norm = math.sqrt(g.x*g.x + g.y*g.y + g.z*g.z)
        stamp = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
        if not (.948 * args.gravity_norm < norm < 1.05 * args.gravity_norm and gyro_norm < .08):
            samples.clear()
            return
        if samples and (stamp <= samples[-1][0] or stamp - samples[-1][0] > .1):
            samples.clear()
        samples.append((stamp, acceleration))
        while samples and stamp - samples[0][0] > args.stable_seconds + .1:
            samples.popleft()

    subscription = node.create_subscription(Imu, args.topic, receive, qos_profile_sensor_data)
    deadline = None if args.timeout==0 else time.monotonic()+args.timeout
    print(f'Waiting for {args.topic}: {args.stable_seconds:g} s stationary, |acc| near {args.gravity_norm:g} in driver units.', flush=True)
    if deadline is None:print('No startup timeout; Ctrl+C stops waiting.',flush=True)
    success = False
    try:
        while deadline is None or time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if len(samples) >= 20 and samples[-1][0] - samples[0][0] >= args.stable_seconds:
                spread = max(max(s[1][axis] for s in samples) - min(s[1][axis] for s in samples) for axis in range(3))
                if spread < .04078 * args.gravity_norm:
                    print(f'MID360 ready at sensor time {samples[-1][0]:.3f}s; starting Faster-LIO.', flush=True)
                    success = True
                    break
        if not success:
            print('MID360 did not become stationary with valid gravity before the timeout.', flush=True)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
