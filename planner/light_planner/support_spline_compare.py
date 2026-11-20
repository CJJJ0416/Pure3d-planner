"""Compare cubic XYZ fitting with native SCAN on stored stair PCD crops."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from scipy.interpolate import BSpline

from common.pointcloud_io import fields_xyz, read_pcd_fields
from planner.light_planner import PlannerConfig, SparseSurfacePlanner
from planner.light_planner.graph.candidates import LocalCandidateSearch
from planner.light_planner.optimization.support_spline import (
    TerrainLimits, arc_path, fit_support_spline, project_path, sample_path, slice_path,
    scan_bspline_message,
)

ROOT = Path(__file__).resolve().parents[2]
SCAN_CONFIG = ROOT/'Simulator/SCAN-Planner/src/planner/plan_manage/config/a1.yaml'
CASES = {'lower_up': 1., 'upper_up': 0., 'short_down': 0., 'same_height': 0.}
HORIZON = 1.5


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def duration(spline):
    return float(spline.t[-spline.k-1] - spline.t[spline.k])


def evaluate(spline, t, derivative=0):
    return spline(np.clip(t, 0., duration(spline)) + spline.t[spline.k], derivative)


def curve_metrics(curve, reference, limits, folder, name, graph):
    t = np.linspace(0, duration(curve), max(201, int(duration(curve)/.01)+1))
    position, velocity, acceleration = [evaluate(curve, t, k) for k in range(3)]
    points, arc = arc_path(reference)
    s = project_path(points, arc, position, horizontal=True)
    matched = sample_path(points, arc, s) + [0., 0., limits.body_height_m]
    z_error = position[:, 2] - matched[:, 2]
    xy_error = np.linalg.norm(position[:, :2]-matched[:, :2], axis=1)
    xy_speed = np.linalg.norm(velocity[:, :2], axis=1)
    np.savetxt(folder/(name+'.csv'), np.column_stack((t, position, velocity, acceleration, s, matched[:, 2])),
               header='t_s,x,y,z,vx_world,vy_world,vz_world,ax,ay,az,projected_arc_m,reference_body_z',
               delimiter=',', comments='')
    return dict(duration_s=duration(curve), z_rmse_m=float(np.sqrt(np.mean(z_error**2))),
        z_max_abs_error_m=float(np.max(np.abs(z_error))), xy_max_error_m=float(xy_error.max()),
        max_xy_speed_mps=float(xy_speed.max()), max_abs_vz_mps=float(np.max(np.abs(velocity[:, 2]))),
        max_acc_mps2=float(np.max(np.linalg.norm(acceleration, axis=1))),
        z_change_m=float(position[-1, 2]-position[0, 2]),
        graph_check=graph.validate_trajectory(position-[0., 0., limits.body_height_m], check_cost=False))


def seam(old, new, elapsed):
    jumps = [evaluate(new, 0., k)-evaluate(old, elapsed, k) for k in range(3)]
    return dict(change_time_s=elapsed, position_jump_m=float(np.linalg.norm(jumps[0])),
                velocity_jump_mps=float(np.linalg.norm(jumps[1])),
                velocity_xy_jump_mps=float(np.linalg.norm(jumps[1][:2])),
                velocity_z_jump_mps=abs(float(jumps[1][2])),
                acceleration_jump_mps2=float(np.linalg.norm(jumps[2])))


class ScanReplay:
    """Send a static PCD and prescribed odometry to an unmodified SCAN executable."""
    def __init__(self):
        import rclpy
        from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry, Path as PathMsg
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import PointCloud2
        from scan_planner_msgs.msg import Bspline

        self.ros = rclpy
        self.node = rclpy.create_node('support_spline_comparison')
        self.clock = self.node.create_publisher(Clock, '/clock', 10)
        self.body = self.node.create_publisher(Odometry, '/body_pose', qos_profile_sensor_data)
        self.sensor = self.node.create_publisher(Odometry, '/sensor_pose', qos_profile_sensor_data)
        self.cloud = self.node.create_publisher(PointCloud2, '/cloud', qos_profile_sensor_data)
        self.path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_type = PathMsg
        self.scan_runs = 0
        self.light_curve = self.node.create_publisher(Bspline, '/light_probe/bspline', 10)
        self.splines = []
        self.commands = []
        self.node.create_subscription(Bspline, '/planning/bspline', self.splines.append, 10)
        self.node.create_subscription(Twist, '/light_probe/cmd_vel', self.commands.append, 20)
        self.sim = 100.

    def stamp(self):
        from builtin_interfaces.msg import Time
        ns = round(self.sim*1e9)
        return Time(sec=ns//1000000000, nanosec=ns%1000000000)

    def step(self, position, velocity, heading, cloud=None):
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from common.io import cloud_message
        self.sim += .025
        stamp = self.stamp()
        self.clock.publish(Clock(clock=stamp))
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'base'
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position)
        odom.pose.pose.orientation.z = float(np.sin(heading/2))
        odom.pose.pose.orientation.w = float(np.cos(heading/2))
        c, s = np.cos(heading), np.sin(heading)
        odom.twist.twist.linear.x = float(c*velocity[0]+s*velocity[1])
        odom.twist.twist.linear.y = float(-s*velocity[0]+c*velocity[1])
        odom.twist.twist.linear.z = float(velocity[2])
        self.body.publish(odom)
        odom.pose.pose.position.z += .18
        self.sensor.publish(odom)
        if cloud is not None:
            self.cloud.publish(cloud_message(cloud, stamp, frame='world'))
        wall = time.monotonic()+.019
        while time.monotonic()<wall:
            self.ros.spin_once(self.node, timeout_sec=.003)

    def send_path(self, publisher, support):
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path as PathMsg
        path = PathMsg()
        path.header.frame_id = 'world'
        path.header.stamp = self.stamp()
        for xyz in support:
            pose = PoseStamped(header=path.header)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, xyz)
            pose.pose.orientation.w = 1.
            path.poses.append(pose)
        publisher.publish(path)

    @staticmethod
    def decode(msg):
        coefficients = np.array([[p.x, p.y, p.z] for p in msg.pos_pts])
        stamp = msg.start_time.sec + msg.start_time.nanosec*1e-9
        return BSpline(msg.knots, coefficients, msg.order), stamp

    def run_scan(self, support, body_height, cloud, folder):
        self.splines.clear()
        self.scan_runs += 1
        path_topic = f'/support_spline_comparison/path_{self.scan_runs}'
        publisher = self.node.create_publisher(self.path_type, path_topic, self.path_qos)
        position = support[0] + [0., 0., body_height]
        delta = support[1]-support[0]
        yaw = np.arctan2(delta[1], delta[0])
        cmd = ['ros2', 'run', 'scan_planner', 'scan_planner_node', '--ros-args',
               '--params-file', str(SCAN_CONFIG), '-p', 'fsm.navi_mode:=4', '-p', 'interface.rolling_reference:=true',
               '-r', 'initial_path:='+path_topic]
        with (folder/'scan.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                for i in range(250):
                    self.step(position, np.zeros(3), yaw, cloud if i%4==0 else None)
                    if publisher.get_subscription_count() and i>55:
                        break
                else:
                    raise RuntimeError("SCAN path subscriber not ready")
                requested = self.sim
                self.send_path(publisher, support)
                for i in range(230):
                    self.step(position, np.zeros(3), yaw, cloud if i%4==0 else None)
                    self.splines[:] = [m for m in self.splines if
                        m.start_time.sec + m.start_time.nanosec*1e-9 >= requested-1e-8]
                    if self.splines:
                        break
                else:
                    raise RuntimeError("SCAN produced no initial B-spline")
                first, first_stamp = self.decode(self.splines[0])
                if np.linalg.norm(evaluate(first, 0.)-position)>.20:
                    raise RuntimeError("SCAN first sample does not match the requested initial position")
                for i in range(130):
                    elapsed = self.sim + .025 - first_stamp
                    position, velocity = evaluate(first, elapsed), evaluate(first, elapsed, 1)
                    if np.linalg.norm(velocity[:2])>.01:
                        yaw = np.arctan2(velocity[1], velocity[0])
                    self.step(position, velocity, yaw, cloud if i%4==0 else None)
                    if len(self.splines)>=2:
                        second, second_stamp = self.decode(self.splines[1])
                        if second_stamp>first_stamp:
                            return first, second, second_stamp-first_stamp
                raise RuntimeError("SCAN produced no subsequent B-spline")
            finally:
                self.node.destroy_publisher(publisher)
                self.stop(proc)

    def run_controller(self, light, folder):
        """Use the unchanged controller with the light cubic knots and ideal odometry."""
        self.commands.clear()
        cmd = ['ros2', 'run', 'scan_planner', 'closed_loop_controller', '--ros-args',
               '--params-file', str(SCAN_CONFIG), '-r', 'planning/bspline:=/light_probe/bspline',
               '-r', 'cmd_vel:=/light_probe/cmd_vel']
        with (folder/'controller.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                pos = light.state(0.).position
                dx = light.state(min(.2, light.duration)).position-pos
                yaw = np.arctan2(dx[1], dx[0])
                for _ in range(160):
                    self.step(pos, np.zeros(3), yaw)
                    if self.light_curve.get_subscription_count():
                        break
                else:
                    raise RuntimeError("SCAN control subscriber not ready")
                self.light_curve.publish(scan_bspline_message(light, self.stamp(), 1))
                for t in np.arange(0., min(1.6, light.duration), .025):
                    state = light.state(t)
                    self.step(state.position, state.velocity, yaw)
                data = np.array([[m.linear.x, m.linear.y, m.linear.z, m.angular.z] for m in self.commands])
                if not len(data) or not np.isfinite(data).all() or not np.any(np.linalg.norm(data[:, :2], axis=1)>.02):
                    raise RuntimeError("Controller produced no finite nonzero XY command")
                np.savetxt(folder/'light_controller_twist.csv', data, delimiter=',',
                           header='vx,vy,vz,omega_z', comments='')
                return dict(messages=len(data), max_abs_cmd_vz=float(np.abs(data[:, 2]).max()),
                            max_cmd_xy_mps=float(np.linalg.norm(data[:, :2], axis=1).max()))
            finally:
                self.stop(proc)

    @staticmethod
    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=4)


def compare(source, folder, case, offset, replay, limits, reference_mode):
    folder.mkdir()
    rows = read_pcd_fields(source/'local_input.pcd')
    xyz = fields_xyz(rows)
    config = PlannerConfig(cost_threshold=.8, dc_input_cost_max=.8)
    graph = SparseSurfacePlanner(config, folder/'native')
    try:
        graph.load_arrays(xyz, rows['trav_cost'])
        metadata = json.loads((source/'result.json').read_text())['case']
        start, _ = graph.match_endpoint(np.asarray(metadata['start']), 1.)
        if start is None:
            raise ValueError("Cannot snap the saved start")
        search = LocalCandidateSearch(graph.step_graph, graph.node_xyz, graph.node_cost,
                                      astar_cost_weight=config.cost_weight, cost_limit=config.cost_threshold)
        result = search.query(start, metadata['goal'])
        errors = []
        for candidate in result['candidates']:
            route = graph.node_xyz[candidate['node_path']]
            try:
                _, full_arc = arc_path(route)
                route = slice_path(route, offset, full_arc[-1])
                reference = slice_path(route, 0., HORIZON)
                light = fit_support_spline(reference, limits)
                samples = light.curve(np.linspace(0., light.duration, int(np.ceil(light.duration/.02))+1))
                check = graph.validate_trajectory(samples-[0., 0., limits.body_height_m], check_cost=False)
                if not check['valid']:
                    raise ValueError('Candidate spline failed existing support graph check: '+check['reason'])
            except ValueError as err:
                errors.append(str(err))
                continue
            break
        else:
            raise ValueError("No usable ranked route: "+" | ".join(errors))
        # Replay voxel centroids from the same PCD. Prescribed sensor poses are
        # only used to update SCAN's map and do not reconstruct original rays.
        cloud = graph.full_xyz
        timings = [fit_support_spline(reference, limits).profile['fit_wall_ms'] for _ in range(20)]
        np.savetxt(folder/'reference_support.csv', reference, header='x,y,z', delimiter=',', comments='')
        report = dict(case=case, source_pcd=str((source/'local_input.pcd').resolve()),
                      points=len(xyz), scan_input_voxels=len(cloud), reference_offset_m=offset,
                      graph_searches=result['metrics']['graph_search_calls'], candidate_id=candidate['candidate_id'],
                      candidate_score=candidate['score'], candidate_fit_rejections=errors,
                      fit_p50_ms=float(np.median(timings)), fit_p95_ms=float(np.percentile(timings, 95)),
                      light=curve_metrics(light.curve, reference, limits, folder, 'light', graph),
                      terrain={k:v for k,v in light.profile.items() if np.isscalar(v)})
        try:
            scan_input = route if reference_mode == 'rolling' else reference
            scan, scan_next, elapsed = replay.run_scan(scan_input, limits.body_height_m, cloud, folder)
            report.update(scan=curve_metrics(scan, reference, limits, folder, 'scan', graph),
                          scan_seam=seam(scan, scan_next, elapsed))
            np.savez_compressed(folder/'scan_curves.npz', first_knots=scan.t, first_points=scan.c,
                                next_knots=scan_next.t, next_points=scan_next.c, handoff_elapsed_s=elapsed)
        except RuntimeError as err:
            report['scan_error'] = str(err)
            log = (folder/'scan.log').read_text(errors='replace')
            if 'Rejected rolling reference with occupied endpoint' in log:
                report['scan_rejection'] = 'occupied_reference_endpoint'
            elif 'First three control points are in obstacles' in log:
                report['scan_rejection'] = 'initial_control_points_occupied'
            elapsed = .65
        progress = np.interp(elapsed, light.profile['profile_t_s'], light.profile['profile_arc_m'])
        next_end = progress+HORIZON if reference_mode == 'rolling' else HORIZON
        successor = fit_support_spline(slice_path(route, progress, next_end), limits,
                                       start=light.state(elapsed))
        report['light_seam'] = seam(light.curve, successor.curve, elapsed)
        try:
            report['yaw_only_controller'] = replay.run_controller(light, folder)
        except RuntimeError as err:
            report['controller_error'] = str(err)
        np.savez_compressed(folder/'light_curves.npz', first_knots=light.curve.t, first_points=light.curve.c,
                            next_knots=successor.curve.t, next_points=successor.curve.c,
                            handoff_elapsed_s=elapsed, **{k:v for k,v in light.profile.items() if k.startswith('profile_')})
        save_json(folder/'result.json', report)
        return report
    finally:
        graph.close()


def plot_comparison(output, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [report for report in reports if 'light' in report]
    if not valid:
        return
    fig, axes = plt.subplots(len(valid), 2, figsize=(10, 2.6*len(valid)), squeeze=False,
                             constrained_layout=True)
    for report, (height_ax, speed_ax) in zip(valid, axes):
        folder = output/report['case']
        data = np.loadtxt(folder/'reference_support.csv', delimiter=',', skiprows=1)
        _, arc = arc_path(data)
        height_ax.plot(arc, data[:, 2]+.4, color='.25', lw=2, ls=':', label='Support + body height')
        for name, color in (('scan', '#e58b35'), ('light', '#1c789d')):
            if name not in report:
                continue
            values = np.genfromtxt(folder/(name+'.csv'), delimiter=',', names=True)
            height_ax.plot(values['projected_arc_m'], values['z'], color=color, label=name.upper())
            speed_ax.plot(values['t_s'], np.hypot(values['vx_world'], values['vy_world']),
                          color=color, label=name.upper())
        with np.load(folder/'light_curves.npz') as path:
            speed_ax.plot(path['profile_t_s'], path['profile_xy_cap_mps'], '--',
                          color='#246750', alpha=.7, label='Light terrain cap')
        title = report['case'].replace('_', ' ').capitalize()
        if 'scan' not in report:
            title += ' · SCAN did not publish a curve'
        height_ax.set(xlabel='Projected support arc (m)', ylabel='Body Z (m)', title=title)
        speed_ax.set(xlabel='Time within first trajectory (s)', ylabel='XY speed (m/s)')
        for ax in (height_ax, speed_ax):
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=.2)
    fig.savefig(output/'height_and_speed.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory with saved Light ROI PCD cases')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', choices=list(CASES), nargs='+', default=list(CASES))
    parser.add_argument('--domain', type=int, default=186)
    parser.add_argument('--reference-mode', choices=['rolling', 'window'], default='rolling',
                        help='Shared long reference with 1.5m lookahead, or identical fixed 1.5m endpoint')
    args = parser.parse_args()
    if not 1 <= args.domain <= 232:
        parser.error('Use an isolated nonzero ROS domain')
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    import rclpy
    rclpy.init(args=[])
    replay = ScanReplay()
    limits = TerrainLimits()
    try:
        for _ in range(16):
            rclpy.spin_once(replay.node, timeout_sec=.03)
        if (replay.node.count_publishers('/clock')>1 or replay.node.count_publishers('/planning/bspline') or
                replay.node.count_publishers('/light_probe/cmd_vel')):
            parser.error('ROS domain already has a clock or planner')
        args.output.mkdir(parents=True, exist_ok=False)
        reports = []
        for name in args.cases:
            try:
                report = compare(args.input/name, args.output/name, name, CASES[name], replay, limits,
                                 args.reference_mode)
            except (ValueError, RuntimeError) as err:
                report = dict(case=name, error=str(err))
            reports.append(report)
            print(json.dumps(report), flush=True)
        save_json(args.output/'summary.json', dict(limits=asdict(limits), domain=args.domain,
            reference_mode=args.reference_mode, cases=reports,
            scope='Saved LIO PCD, actual native SCAN and controller, prescribed ideal odometry. No physical motion.'))
        plot_comparison(args.output, reports)
    finally:
        replay.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
