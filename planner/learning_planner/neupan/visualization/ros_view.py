# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.visualization.ros_view --help
"""Standalone ROS/RViz adapter. Planning and training run without ROS."""
import json
from math import atan2, cos, sin
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


PREFIX = '/entry/learning'
CANDIDATE_COLORS = ((.15, .85, 1.), (1., .64, .16), (.92, .42, .94))


def candidate_metrics(result, rank):
    z = result.support_xyz[:, 2]
    return dict(rank=rank, length_m=result.length_m, nodes=len(z), stop_reason=result.reason,
                delta_z_m=float(z[-1] - z[0]), z_min_m=float(z.min()), z_max_m=float(z.max()),
                z_span_m=float(np.ptp(z)))


def candidate_markers(paths, records, header):
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    markers = [Marker(header=header, action=Marker.DELETEALL)]
    if not paths:
        return MarkerArray(markers=markers)
    xyz = np.concatenate(paths)
    legend = tuple(map(float, xyz[:, :2].mean(axis=0)))
    legend_z = float(xyz[:, 2].max()) + 1.55
    for points, record, color in zip(paths, records, CANDIDATE_COLORS):
        rank = record['rank']
        rgba = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.)
        line = Marker(header=header, ns='beam_paths', id=rank, type=Marker.LINE_STRIP, color=rgba)
        line.pose.orientation.w = 1.
        line.scale.x = .034
        line.points = [Point(x=float(x), y=float(y), z=float(z) + .045) for x, y, z in points]
        end = Marker(header=header, ns='beam_endpoints', id=rank, type=Marker.SPHERE, color=rgba)
        end.pose.orientation.w = 1.
        end.pose.position = line.points[-1]
        end.scale.x = end.scale.y = end.scale.z = .072
        label = Marker(header=header, ns='beam_labels', id=rank, type=Marker.TEXT_VIEW_FACING, color=rgba)
        label.pose.orientation.w = 1.
        label.pose.position = Point(x=legend[0], y=legend[1], z=legend_z - .19 * (rank - 1))
        label.scale.z = .105
        label.text = (f'#{rank}|L={record["length_m"]:.2f}m|dZ={record["delta_z_m"]:+.2f}m'
                      f'|span={record["z_span_m"]:.2f}m|{record["stop_reason"]}')
        markers.extend((line, end, label))
    return MarkerArray(markers=markers)


def goal_pose(record, body_height=0.):
    from geometry_msgs.msg import Pose
    x, y, z = record['support_xyz']
    message = Pose()
    message.position.x, message.position.y, message.position.z = x, y, z + body_height
    message.orientation.z = sin(record['yaw'] / 2.)
    message.orientation.w = cos(record['yaw'] / 2.)
    return message


def local_goal_markers(records, header, rollout, start, goal):
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    markers = [Marker(header=header, action=Marker.DELETEALL)]
    for rank, name, xyz in ((0, 'Start', start), (1, 'Target', goal)):
        if xyz is None:
            continue
        record = dict(support_xyz=xyz, yaw=0.)
        rgba = ColorRGBA(r=.95, g=.95, b=.95, a=.9)
        anchor = Marker(header=header, ns='goal_anchors', id=rank, type=Marker.CUBE,
                        color=rgba, pose=goal_pose(record))
        anchor.scale.x = anchor.scale.y = anchor.scale.z = .075
        label = Marker(header=header, ns='goal_anchor_labels', id=rank, type=Marker.TEXT_VIEW_FACING,
                       color=rgba, pose=goal_pose(record, .38))
        label.scale.z = .11
        label.text = name
        markers.extend((anchor, label))
    for record in records:
        rank = record['rank']
        color = CANDIDATE_COLORS[0] if rollout else CANDIDATE_COLORS[rank - 1]
        rgba = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.)
        ball = Marker(header=header, ns='local_goals', id=rank, type=Marker.SPHERE,
                      color=rgba, pose=goal_pose(record))
        ball.scale.x = ball.scale.y = ball.scale.z = .12
        label = Marker(header=header, ns='local_goal_labels', id=rank, type=Marker.TEXT_VIEW_FACING,
                       color=rgba, pose=goal_pose(record, .23))
        label.scale.z = .11
        label.text = f'G{rank}|Z={record["support_xyz"][2]:.2f}m'
        markers.extend((ball, label))
    return MarkerArray(markers=markers)


def path_message(points, header):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    message = Path(header=header)
    for i, (x, y, z) in enumerate(points):
        if len(points) > 1:
            start = min(i, len(points) - 2)
            dx, dy = points[start + 1, :2] - points[start, :2]
            yaw = atan2(float(dy), float(dx))
        else:
            yaw = 0.
        pose = PoseStamped(header=header)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
        pose.pose.orientation.z, pose.pose.orientation.w = sin(yaw / 2), cos(yaw / 2)
        message.poses.append(pose)
    return message


def slice_message(slices, header):
    from sensor_msgs.msg import PointCloud2, PointField
    cloud = slices.astype('<f4', copy=False)
    names = ('x', 'y', 'z', 'trav_cost', 'slice_id')
    fields = [PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
              for index, name in enumerate(names)]
    return PointCloud2(header=header, height=1, width=len(cloud), fields=fields,
                       is_bigendian=False, point_step=20, row_step=cloud.nbytes,
                       is_dense=True, data=cloud.tobytes())


class PreviewPublisher:
    def __init__(self, node, publish_pure_path=False, beam=False, local=False):
        from nav_msgs.msg import Path
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        self.node = node
        self.qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=ReliabilityPolicy.RELIABLE)
        self.slices = node.create_publisher(PointCloud2, PREFIX + '/slice_cost', self.qos)
        self.expert = node.create_publisher(Path, PREFIX + '/expert_path', self.qos)
        self.learned = None if local else node.create_publisher(Path, PREFIX + '/path', self.qos)
        self.status = node.create_publisher(String, PREFIX + '/status', self.qos)
        self.candidates = None
        self.support_goals = self.preview_goal = None
        if local:
            from geometry_msgs.msg import PoseArray, PoseStamped
            self.support_goals = node.create_publisher(PoseArray, PREFIX + '/support_goals', self.qos)
            self.preview_goal = node.create_publisher(PoseStamped, PREFIX + '/local_goal', self.qos)
        if beam or local:
            from visualization_msgs.msg import MarkerArray
            self.candidates = node.create_publisher(MarkerArray, PREFIX + '/candidates', self.qos)
        self.control = node.create_publisher(Path, '/pure_path', self.qos) if publish_pure_path else None

    def publish(self, result, slices, expert, *, frame, metadata=None, candidates=None):
        from std_msgs.msg import Header, String
        if self.control is not None:
            others = [info for info in self.node.get_publishers_info_by_topic('/pure_path')
                      if info.node_name != self.node.get_name()
                      or info.node_namespace != self.node.get_namespace()]
            if others:
                raise RuntimeError('Another /pure_path publisher exists in the validation domain')
        header = Header(frame_id=frame, stamp=self.node.get_clock().now().to_msg())
        self.slices.publish(slice_message(slices, header))
        self.expert.publish(path_message(expert, header))
        if self.learned is not None:
            self.learned.publish(path_message(result.support_xyz, header))
        if self.support_goals is not None:
            from geometry_msgs.msg import PoseArray, PoseStamped
            goals = metadata['local_goals']
            records = goals['points']
            self.candidates.publish(local_goal_markers(records, header, goals['rollout'],
                                                       metadata['start_support_xyz'], metadata['goal']))
            self.support_goals.publish(PoseArray(header=header, poses=[goal_pose(record) for record in records]))
            if records and not goals['rollout']:
                self.preview_goal.publish(PoseStamped(header=header, pose=goal_pose(records[0], .4)))
        elif self.candidates is not None:
            self.candidates.publish(candidate_markers(candidates, metadata['beam']['candidates'], header))
        data = dict(metadata or {}, frame=frame, cost_version=result.cost_version, nodes=len(result.node_chain),
                    length_m=result.length_m, stop_reason=result.reason, build_ms=result.build_ms,
                    decode_ms=result.decode_ms, pure_path_enabled=self.control is not None)
        self.status.publish(String(data=json.dumps(data)))
        if self.control is not None:
            self.control.publish(path_message(result.body_path(), header))
        return data


def view(args):
    import sys

    import torch

    from planner.learning_planner.neupan.inference.path_generator import PathGenerator
    from planner.learning_planner.neupan.data.supervision import Windows, read_cloud

    torch.set_num_threads(args.threads)
    samples = Windows(args.dataset, args.split, args.pairs)
    planner = PathGenerator(args.checkpoint, args.device)
    if args.route_id is None:
        if not 0 <= args.index < len(samples):
            raise ValueError(f'Available {args.split} window indices: 0 through {len(samples) - 1}')
        window = samples[args.index]
        graph = window.graph
        path = graph.xyz[window.chain]
        goal = window.goal
        start = path[0] if len(path) else np.array([np.inf, np.inf, np.inf])
        rid, offset, scenario, cloud_path = samples.index[args.index]
        version = (args.snapshot_version or samples.versions.get(args.index)
                   or f'{samples.meta["sha256"][:12]}/{rid}/{offset}/{scenario}')
        status = dict(route_id=rid, window=args.index, scenario=scenario,
                      expert_stop_reason=window.end, goal=goal.tolist())
        build_ms = window.build_ms
        visual_map = graph.tomo
        if args.rollout:
            path = samples.tomo.xyz(samples.routes[rid][offset:])
            status.update(expert_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
                          expert_delta_z_m=float(path[-1, 2] - path[0, 2]))
    else:
        rid = args.route_id
        if not 0 <= rid < len(samples.routes) or samples.meta['routes'][rid]['split'] != args.split:
            raise ValueError(f'Route {rid} is not in the requested test split')
        route = samples.routes[rid]
        path = samples.tomo.xyz(route)
        start, goal = path[0], np.asarray(samples.meta['routes'][rid]['goal'], dtype=np.float32)
        version = args.snapshot_version or f'{samples.meta["sha256"][:12]}/{rid}/rollout'
        status = dict(route_id=rid, scenario='test_route', split=args.split, goal=goal.tolist(),
                      expert_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
                      expert_delta_z_m=float(path[-1, 2] - path[0, 2]))
        visual_map, _ = samples.tomo.crop(route[len(route) // 2], width=max(6., 2. * abs(
            path[:, :2] - path[len(path) // 2, :2]).max() + 2.))
    if args.local_goals:
        from planner.learning_planner.neupan.inference.path_generator import PathResult
        from planner.learning_planner.neupan.inference.stair_rollout import rollout_local_goals, rollout_observed_goals

        if args.rollout:
            selected, reason, build_ms, search_ms = rollout_observed_goals(
                planner, read_cloud(cloud_path), start, goal, args.lookahead, args.beam_width)
        elif args.route_id is None:
            before = perf_counter()
            selected = planner.local_goals(graph, start, goal, lookahead=args.lookahead,
                                           width=args.beam_width)
            search_ms = (perf_counter() - before) * 1000.
            reason = 'local_goals_ready' if selected else 'no_local_goal'
        else:
            selected, reason, build_ms, search_ms = rollout_local_goals(
                planner, samples.tomo, route[0], goal, args.lookahead, args.beam_width)
        status['start_support_xyz'] = start.tolist() if np.isfinite(start).all() else None
        status['local_goals'] = dict(rollout=args.rollout or args.route_id is not None, lookahead_m=args.lookahead,
                                    found=len(selected), reason=reason, search_ms=search_ms,
                                    points=[dict(rank=i, support_xyz=point.support_xyz.tolist(),
                                                 support_cell=point.support_cell.tolist(), yaw=point.yaw,
                                                 approach_m=point.approach_m, delta_z_m=point.delta_z_m,
                                                 stop_reason=point.reason)
                                            for i, point in enumerate(selected, 1)])
        result = PathResult(np.empty((0, 3), dtype=int), np.empty((0, 3)), reason, version,
                            build_ms, search_ms)
    else:
        result = planner.plan_graph(graph, start, goal, cost_version=version, build_ms=build_ms)
    points, offsets = [], [0]
    if args.beam:
        from planner.learning_planner.neupan.inference.beam import DIVERSITY_ARC, DIVERSITY_SEPARATION, exclusive_arcs

        before = perf_counter()
        candidates = planner.plan_candidates_graph(graph, start, window.goal, cost_version=version,
                                                   width=args.beam_width, build_ms=window.build_ms)
        elapsed = (perf_counter() - before) * 1000.
        points = [candidate.support_xyz for candidate in candidates]
        offsets += np.cumsum([len(path) for path in points]).tolist()
        pairs = [dict(ranks=[a + 1, b + 1], exclusive_arc_m=exclusive_arcs(points[a], points[b]))
                 for a in range(len(points)) for b in range(a + 1, len(points))]
        status['beam'] = dict(requested=3, found=len(points), width=args.beam_width,
                              separation_m=DIVERSITY_SEPARATION, minimum_exclusive_arc_m=DIVERSITY_ARC,
                              search_ms=elapsed, pairs=pairs,
                              candidates=[candidate_metrics(c, i) for i, c in enumerate(candidates, 1)])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = dict(status, cost_version=version, start_support_xyz=start.tolist() if np.isfinite(start).all() else None,
                      build_ms=result.build_ms)
        if args.local_goals:
            report['expert_support_xyz'] = path.tolist()
        else:
            report['greedy'] = dict(length_m=result.length_m, stop_reason=result.reason,
                                    support_xyz=result.support_xyz.tolist(), decode_ms=result.decode_ms)
            report['candidate_support_xyz'] = [xyz.tolist() for xyz in points]
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.plan_only:
        print(json.dumps(status, allow_nan=False))
        return
    with TemporaryDirectory(prefix='support-path-ros-') as tmp:
        snapshot = Path(tmp) / 'view.npz'
        cloud = visual_map.cloud()
        if args.route_id is not None:
            cloud = cloud[(cloud[:, 2] >= path[:, 2].min() - .28) & (cloud[:, 2] <= path[:, 2].max() + .32)]
        np.savez_compressed(snapshot, slices=cloud, expert=path, learned=result.support_xyz,
                            node_chain=result.node_chain,
                            beam_xyz=np.concatenate(points) if points else np.empty((0, 3)),
                            beam_offsets=np.asarray(offsets),
                            status=json.dumps(dict(status, cost_version=result.cost_version, reason=result.reason,
                                                   build_ms=result.build_ms, decode_ms=result.decode_ms)))
        env = dict(os.environ, ROS_LOCALHOST_ONLY='1', ROS_DOMAIN_ID=str(args.domain))
        env.setdefault('ROS_LOG_DIR', str(Path(tmp) / 'ros_logs'))
        command = ['bash', '-c',
                   'source /opt/ros/humble/setup.bash && exec /usr/bin/python3 -m planner.light_planner.learning_planner.neupan.visualization.ros_view "$@"',
                   'support_path_view', str(snapshot)]
        if args.rviz:
            command.append('--rviz')
        if args.publish_pure_path:
            command.append('--publish-pure-path')
        # Humble uses the system Python ABI. The child has no PyTorch dependency.
        if not sys.prefix.startswith('/usr'):
            ld_path = env.get('LD_LIBRARY_PATH', '').split(':')
            env['LD_LIBRARY_PATH'] = ':'.join(item for item in ld_path if item and not item.startswith(sys.prefix))
        try:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[4], env=env, check=True)
        except KeyboardInterrupt:
            pass


def publish_snapshot(snapshot, rviz=False, pure_path=False):
    from types import SimpleNamespace

    import rclpy
    import yaml
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import ExternalShutdownException
    from tf2_ros import StaticTransformBroadcaster

    with np.load(snapshot, allow_pickle=False) as store:
        slices, path, learned, chain = (store[key] for key in ('slices', 'expert', 'learned', 'node_chain'))
        status = json.loads(str(store['status']))
        candidates = None
        if 'beam' in status:
            positions, offsets = store['beam_xyz'], store['beam_offsets']
            candidates = [positions[a:b] for a, b in zip(offsets, offsets[1:])]
    body = learned.copy()
    if len(body) > 1:
        body[:, 2] += .4
    else:
        body = np.empty((0, 3))
    result = SimpleNamespace(support_xyz=learned, node_chain=chain, body_path=lambda: body,
                             length_m=float(np.linalg.norm(np.diff(learned[:, :2], axis=0), axis=1).sum()),
                             **{key: status.pop(key) for key in ('cost_version', 'reason', 'build_ms', 'decode_ms')})

    rclpy.init()
    node = rclpy.create_node(f'learned_path_preview_{os.getpid()}')
    viewer = None
    with TemporaryDirectory(prefix='support-path-rviz-') as tmp:
        try:
            local = 'local_goals' in status
            rollout = local and status['local_goals']['rollout']
            publisher = PreviewPublisher(node, pure_path, beam=candidates is not None, local=local)
            static_tf = StaticTransformBroadcaster(node)
            origin = TransformStamped()
            origin.header.frame_id = 'world'
            origin.header.stamp = node.get_clock().now().to_msg()
            origin.child_frame_id = node.get_name() + '_origin'
            origin.transform.rotation.w = 1.
            static_tf.sendTransform(origin)
            rclpy.spin_once(node, timeout_sec=.5)
            data = publisher.publish(result, slices, path, frame='world', metadata=status, candidates=candidates)
            print(json.dumps(data), flush=True)
            if rviz:
                executable = shutil.which('rviz2')
                if executable is None:
                    raise RuntimeError('rviz2 not found. Source ROS Humble.')
                config = yaml.safe_load(Path(__file__).with_name('preview.rviz').read_text())
                manager = config['Visualization Manager']
                focal = learned.mean(axis=0) if len(learned) else status['goal']
                if local or candidates is not None:
                    for display in manager['Displays']:
                        if display['Class'] == 'rviz_default_plugins/Path':
                            display['Enabled'] = local and display['Name'] == 'VST expert prefix'
                            if display['Enabled']:
                                name = 'VST reference route' if rollout else 'VST expert prefix'
                                display.update({'Name': name, 'Line Width': .047})
                                display['Offset']['Z'] = .04
                        if display['Class'] == 'rviz_default_plugins/PointCloud2':
                            display.update({'Style': 'Points', 'Size (Pixels)': 3, 'Alpha': .6,
                                            'Max Color': '116; 83; 91', 'Min Color': '174; 194; 170'})
                    manager['Displays'].append({'Class': 'rviz_default_plugins/MarkerArray', 'Name': 'Network local goals' if local else 'Beam candidates',
                                               'Enabled': True, 'Topic': {
                                                   'Value': PREFIX + '/candidates', 'Reliability Policy': 'Reliable',
                                                   'Durability Policy': 'Transient Local'}})
                    manager['Views']['Current'].update({'Distance': 5.9, 'Pitch': .78, 'Yaw': 1.1})
                    if candidates:
                        focal = np.concatenate(candidates).mean(axis=0)
                        focal[2] += .25
                    if local and len(path):
                        focal = path.mean(axis=0)
                        extent = float(np.ptp(path[:, :2], axis=0).max())
                        manager['Views']['Current'].update({'Distance': max(5.6, 1.65 * extent),
                                                            'Pitch': .77, 'Yaw': .65})
                manager['Views']['Current']['Focal Point'] = dict(
                    zip(('X', 'Y', 'Z'), map(float, focal)))
                config_name = 'local_goal_preview.rviz' if local else ('beam_preview.rviz' if candidates is not None else 'preview.rviz')
                output = Path(tmp) / config_name
                output.write_text(yaml.safe_dump(config, sort_keys=False))
                viewer = subprocess.Popen([executable, '-d', str(output)])
            while rclpy.ok() and (viewer is None or viewer.poll() is None):
                rclpy.spin_once(node, timeout_sec=.1)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            if viewer is not None and viewer.poll() is None:
                viewer.terminate()
                viewer.wait(timeout=5)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    if viewer is not None and viewer.returncode not in (0, -15):
        raise RuntimeError(f'RViz exited with code {viewer.returncode}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--rviz', action='store_true')
    parser.add_argument('--publish-pure-path', action='store_true')
    options = parser.parse_args()
    publish_snapshot(options.snapshot, options.rviz, options.publish_pure_path)
