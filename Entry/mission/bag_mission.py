"""Independent commands for VST goals, measured following, recording and replay."""
from Entry.utils.paths import ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo
import numpy as np
import yaml
from Entry.mission.bag_io import PLAY_TOPICS, RECORD_TOPICS, inspect_bag
from Entry.utils.io import atomic_json
from Entry.mission.mission_core import FollowState


DEFAULT_SCENE = ROOT/'datasets/wfc_capability_dr_v2/scene_20261109'
DEFAULT_SPAWN = ROOT/'datasets/simulation/gazebo_terrain/spawn.json'
DEFAULT_NATIVE = ROOT/'datasets/cache/vst_native_py310'


def mission_path(path):
    path = Path(path).resolve()
    return path/'mission.json' if path.is_dir() else path


def read_mission(path):
    path = mission_path(path)
    data = json.loads(path.read_text())
    if data['state'] != 'complete' and not (data.get('planning_mode')=='stepwise_short_goals' and data['state']=='ready'):
        raise ValueError('Mission planning is incomplete')
    return path, data


def native_plan(scene, spawn, native, output, base, seed=821111, segments=4, goal_index=None,
                distance_range=None,max_path=None,heading=None,visited=None):
    command = ['/usr/bin/python3', '-m', 'Entry.mission.mission_plan', '--scene', str(scene), '--spawn', str(spawn),
        '--native', str(native), '--output', str(output), '--start-base', *map(str, base),
        '--seed', str(seed), '--segments', str(segments)]
    if goal_index is not None:
        command += ['--goal-index', *map(str, goal_index)]
    if distance_range is not None:command+=['--distance-min',str(distance_range[0]),'--distance-max',str(distance_range[1])]
    if max_path is not None:command+=['--max-path',str(max_path)]
    if heading is not None:command+=['--heading',str(heading)]
    if visited is not None:
        visited_file=output.parent/(output.name+'_visited.json')
        atomic_json(visited_file,visited)
        command+=['--visited',str(visited_file)]
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent/(output.name+'.log')
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, LD_LIBRARY_PATH=str(native)+':'+os.environ.get('LD_LIBRARY_PATH', '')))
    if result.returncode:
        raise RuntimeError(f'Native planning failed; see {log_path}')
    return json.loads((output/'mission.json').read_text())


def query_base(timeout):
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener, TransformException
    rclpy.init(args=[])
    node = Node('vst_mission_base_query')
    buffer = Buffer(); listener = TransformListener(buffer, node)
    try:
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            try:
                tf = buffer.lookup_transform('world', 'base', Time())
                p, q = tf.transform.translation, tf.transform.rotation
                return dict(position=[p.x, p.y, p.z], quaternion_xyzw=[q.x, q.y, q.z, q.w],
                            stamp=tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9, frame_id='world')
            except TransformException:
                pass
        raise TimeoutError('Current world -> base unavailable; start Gazebo and stand the robot first')
    finally:
        node.destroy_node(); rclpy.shutdown()


def command_plan(args):
    base = query_base(args.timeout)
    output = args.output.resolve() if args.output else ROOT/'outputs/entry_bag_mission'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    if args.stepwise:
        from Entry.mission.stepwise import initialize
        data=initialize(args,base,output)
        print(json.dumps(data,ensure_ascii=False,indent=2))
        return
    data = native_plan(args.scene.resolve(), args.spawn.resolve(), args.native.resolve(), output,
                       base['position'], args.seed, args.segments)
    atomic_json(output/'planning_base.json', base)
    print(json.dumps(dict(mission=str(output/'mission.json'), goals=[r['goal_body_xyz'] for r in data['segments']],
                         start_policy='read current base again before every executed segment'), ensure_ascii=False, indent=2))


def command_follow(args):
    _,metadata=read_mission(args.mission)
    if metadata.get('planning_mode')=='stepwise_short_goals':
        from Entry.mission.stepwise import follow
        return follow(args)
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, Path as PathMessage
    from std_msgs.msg import String
    from Entry.utils.io import pose_value, seconds, ros_stamp

    path, mission = read_mission(args.mission)
    directory = path.parent
    if (directory/'execution.json').exists():
        raise ValueError('execution.json already exists; use a new mission for a new capture')
    rclpy.init(args=[])
    node = Node('vst_mission_follow', parameter_overrides=[Parameter('use_sim_time', value=True)])
    state = FollowState(mission['segments'])
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    path_pub = node.create_publisher(PathMessage, '/pct_path', qos)
    goal_pub = node.create_publisher(PoseStamped, '/scan/goal', 10)
    status_pub = node.create_publisher(String, '/entry/mission_status', qos)
    latest = dict(pose=None, stamp=None, wall=None)
    def odometry(message):
        if message.header.frame_id != 'world' or message.child_frame_id != 'base':
            state.fail('Expected Gazebo world -> base odometry', None)
            return
        position, _ = pose_value(message.pose.pose)
        latest.update(pose=position, stamp=seconds(message.header.stamp), wall=time.monotonic())
    subscription = node.create_subscription(Odometry, '/Odometry_gazebo', odometry,
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    base_for_plan = None
    requests = 0
    last_save = 0.
    try:
        discovery = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.05)
            if time.monotonic()-discovery < 1.:
                continue
            owners = node.get_publishers_info_by_topic('/pct_path')
            if len(owners) != 1 or owners[0].node_name != node.get_name():
                state.fail('Another /pct_path publisher exists; stop that planner before following', latest['stamp'])
            if state.state in ('planning', 'following'):
                capture_file = directory/'capture.json'
                capture = json.loads(capture_file.read_text()) if capture_file.exists() else {}
                if capture.get('state') != 'recording' or time.time()-capture_file.stat().st_mtime > 3.:
                    state.fail('Mission recorder stopped; use the keyboard to pause the robot', latest['stamp'])
            if latest['pose'] is None:
                if time.monotonic()-discovery > args.timeout:
                    state.fail('Gazebo odometry unavailable', None)
            elif time.monotonic()-latest['wall'] > 2.:
                state.fail('Odometry stream stopped; use the keyboard to pause the robot', latest['stamp'])
            else:
                if state.state == 'waiting':
                    capture_file = directory/'capture.json'
                    capture = json.loads(capture_file.read_text()) if capture_file.exists() else {}
                    beginning = capture.get('first_sensor_stamp')
                    if capture.get('state') != 'recording' or beginning is None or latest['stamp']-beginning < 10.5:
                        if time.monotonic()-last_save > 2:
                            print('Waiting for this mission recorder and at least 10 s of stationary sensors...', flush=True)
                        index = None
                    elif path_pub.get_subscription_count() == 0:
                        index = None
                    else:
                        index = state.update(latest['pose'], latest['stamp'])
                else:
                    index = state.update(latest['pose'], latest['stamp'])
                if index is not None:
                    base_for_plan = latest['pose'].copy()
                    requests += 1
                    destination = directory/'executed_paths'/f'segment_{index:02d}_request_{requests:03d}'
                    atomic_json(destination.parent/(destination.name+'_base.json'),
                        dict(frame_id='world', stamp=latest['stamp'], base_xyz=base_for_plan.tolist()))
                    row = mission['segments'][index]
                    future = pool.submit(native_plan, Path(mission['scene']), Path(mission['spawn']),
                        Path(mission['native_dir']), destination, base_for_plan,
                        mission['settings']['seed'], 1, row['goal_vst_index'])
                if state.state == 'planning' and future is not None and future.done():
                    planned = future.result()
                    future = None
                    if np.linalg.norm(latest['pose']-base_for_plan) > .3:
                        state.fail('Base moved more than 0.3 m during planning; pause and restart with a new mission', latest['stamp'])
                    else:
                        row = planned['segments'][0]
                        body = np.load(row['body'])
                        message = PathMessage()
                        message.header.frame_id = 'world'
                        message.header.stamp = ros_stamp(latest['stamp'])
                        delta = np.diff(body[:, :2], axis=0)
                        yaw = np.r_[np.arctan2(delta[:, 1], delta[:, 0]), 0.]
                        yaw[-1] = yaw[-2]
                        for xyz, angle in zip(body, yaw):
                            pose = PoseStamped(); pose.header = message.header
                            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, xyz)
                            pose.pose.orientation.z, pose.pose.orientation.w = float(np.sin(angle/2)), float(np.cos(angle/2))
                            message.poses.append(pose)
                        goal = PoseStamped(); goal.header = message.header
                        target = mission['segments'][state.index]['goal_body_xyz']
                        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = map(float, target)
                        goal.pose.orientation.w = 1.
                        path_pub.publish(message); goal_pub.publish(goal)
                        mission['segments'][state.index]['length_m'] = row['length_m']
                        state.published(latest['stamp'], base_for_plan)
                        state.events[-1]['executed_plan'] = str(Path(row['body']).parent/'mission.json')
                        print(f'Published segment {state.index+1}/{len(state.segments)} from current base', flush=True)
            if time.monotonic()-last_save > .5 or state.state in ('complete', 'failed'):
                report = dict(state.report(), mission=str(path), frame_id='world', start_policy='current_base_each_segment')
                atomic_json(directory/'execution.json', report)
                status_pub.publish(String(data=json.dumps(report)))
                last_save = time.monotonic()
            if state.state in ('complete', 'failed'):
                if state.state == 'failed':
                    raise RuntimeError(state.error)
                print('Four measured arrivals complete. Keep recording stationary for 5 s, then stop record.', flush=True)
                break
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        state.fail('Operator interrupted following; execution incomplete', latest['stamp'])
    except Exception as error:
        state.fail(str(error), latest['stamp'])
        raise
    finally:
        pool.shutdown(wait=True)
        atomic_json(directory/'execution.json', dict(state.report(), mission=str(path), start_policy='current_base_each_segment'))
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


def command_record(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from livox_ros_driver2.msg import CustomMsg
    from Entry.mission.bag_io import stamp

    mission, _ = read_mission(args.mission)
    directory = mission.parent
    bag = (args.bag or directory/'bag').resolve()
    if bag.exists() or (directory/'capture.json').exists():
        raise ValueError('Bag/capture already exists; create a new mission')
    rclpy.init(args=[]); node = Node('vst_mission_record')
    capture = dict(state='starting', mission=str(mission), bag=str(bag), ros_domain_id=int(os.environ.get('ROS_DOMAIN_ID', '0')),
                   first_sensor_stamp=None, last_sensor_stamp=None, observed_scans=0)
    def scan(message):
        if capture['first_sensor_stamp'] is None:
            owners = node.get_subscriptions_info_by_topic('/livox/lidar2')
            if not any('recorder' in p.node_name for p in owners): return
            capture['first_sensor_stamp'] = stamp(message.header.stamp)
        capture.update(last_sensor_stamp=stamp(message.header.stamp), observed_scans=capture['observed_scans']+1)
    subscription = node.create_subscription(CustomMsg, '/livox/lidar2', scan,
        QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
    qos_file = directory/'record_qos.yaml'
    qos_file.write_text(yaml.safe_dump({topic: dict(history='keep_last', depth=100, reliability='reliable',
        durability='transient_local') for topic in ['/tf_static', '/robot_description']}))
    process = None
    try:
        if node.get_publishers_info_by_topic('/faster_lio/cloud_registered'):
            raise RuntimeError('Stop Faster-LIO while recording this mission')
        process = subprocess.Popen(['ros2', 'bag', 'record', '--use-sim-time', '-o', str(bag),
            '--qos-profile-overrides-path', str(qos_file), *RECORD_TOPICS], start_new_session=True)
        capture.update(state='recording', recorder_pid=process.pid)
        last_save = 0.
        while rclpy.ok() and process.poll() is None:
            rclpy.spin_once(node, timeout_sec=.1)
            if node.get_publishers_info_by_topic('/faster_lio/cloud_registered'):
                raise RuntimeError('Faster-LIO started during capture; keep recorded TF free of estimates')
            if time.monotonic()-last_save >= .5:
                atomic_json(directory/'capture.json', capture)
                last_save = time.monotonic()
        if process.poll() not in (None, 0):
            raise RuntimeError(f'Bag recorder exited {process.returncode}')
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=10)
        capture.update(state='stopped', returncode=process.returncode if process else None)
        atomic_json(directory/'capture.json', capture)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        if (bag/'metadata.yaml').exists():
            audit = inspect_bag(bag, mission)
            atomic_json(directory/'bag_check.json', audit)
            print(json.dumps(dict(bag=str(bag), audit=audit['state'], problems=audit['problems']), ensure_ascii=False))


def command_validate(args):
    report = inspect_bag(args.bag, args.mission)
    if args.report: atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['state'] != 'passed':
        raise SystemExit(1)


def command_play(args):
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    mission = mission_path(args.mission) if args.mission else None
    audit = inspect_bag(args.bag, mission)
    output = args.output.resolve() if args.output else Path(args.bag).resolve().parent/'offline'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'bag_check.json', audit)
    if audit['state'] != 'passed':
        raise RuntimeError('Bag preflight failed: '+ '; '.join(audit['problems']))
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from tf2_ros import Buffer, TransformListener
    from rclpy.time import Time
    from Entry.utils.io import seconds, cloud_xyz, pose_value, transform_value
    from Entry.mission.offline import ReconstructionArchive
    from common.unitree_gazebo.scripts.frame_config import rotation_matrix
    rclpy.init(args=[]); node = Node('vst_bag_play_monitor')
    observed = dict(first=None, last=None, messages=0, last_odom=None)
    archive = ReconstructionArchive(output/'reconstruction')
    buffer = Buffer(); listener = TransformListener(buffer, node)
    def cloud(message):
        try:
            t = seconds(message.header.stamp)
            archive.cloud(cloud_xyz(message), t, message.header.frame_id)
            if observed['first'] is None: observed['first'] = t
            observed.update(last=t, messages=observed['messages']+1)
        except Exception as error: archive.error = str(error)
    def odometry(message):
        if message.header.frame_id not in ('camera_init', 'map') or message.child_frame_id != 'body':
            archive.error = 'Expected camera_init/map -> body LIO odometry'
            return
        position, quaternion = pose_value(message.pose.pose)
        t = seconds(message.header.stamp)
        archive.odometry.append(dict(stamp=t, frame_id='map', position=position.tolist(), quaternion_xyzw=quaternion.tolist()))
        observed['last_odom']=t
    def goal(message):
        position, quaternion = pose_value(message.pose)
        archive.goals.append(dict(stamp=seconds(message.header.stamp), frame_id=message.header.frame_id,
            position=position.tolist(), quaternion_xyzw=quaternion.tolist()))
    sensor = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = [node.create_subscription(PointCloud2, '/faster_lio/cloud_registered', cloud,
        sensor), node.create_subscription(Odometry, '/faster_lio/Odometry', odometry, sensor),
        node.create_subscription(PoseStamped, '/scan/goal', goal, 10)]
    process = None
    transform = None
    report = dict(state='failed', bag=str(Path(args.bag).resolve()), domain=args.domain,
                  rate=args.rate, topics=PLAY_TOPICS, generates_clock=False, lio_drained=False,
                  nodes='not_run', cost='not_run', merge='not_run')
    try:
        deadline = time.monotonic()+args.ready_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if node.get_publishers_info_by_topic('/clock'):
                raise RuntimeError('Offline domain already has /clock; stop its Gazebo/player first')
            if node.get_publishers_info_by_topic('/entry/status'):
                raise RuntimeError('Stop online Entry before serial reconstruction; run nodes/cost only after PCD is saved')
            lidar = node.get_subscriptions_info_by_topic('/livox/lidar2')
            imu = node.get_subscriptions_info_by_topic('/livox/imu')
            lio_ready = any(p.node_namespace == '/faster_lio' for p in lidar) and any(p.node_namespace == '/faster_lio' for p in imu)
            bridge_ready = any(p.node_name == 'faster_lio_frame_bridge'
                              for p in node.get_subscriptions_info_by_topic('/faster_lio/raw_odometry'))
            if lio_ready and bridge_ready:
                if len(node.get_publishers_info_by_topic('/faster_lio/cloud_registered')) != 1:
                    raise RuntimeError('Expected exactly one Faster-LIO registered cloud publisher')
                break
        else:
            raise TimeoutError('Start bag-mode Faster-LIO and its TF bridge in this domain first')
        process = subprocess.Popen(['ros2', 'bag', 'play', str(Path(args.bag).resolve()), '-r', str(args.rate),
            '--disable-keyboard-controls', '--topics', *PLAY_TOPICS], start_new_session=True)
        while process.poll() is None:
            rclpy.spin_once(node, timeout_sec=.05)
            if len(node.get_publishers_info_by_topic('/clock')) > 1:
                raise RuntimeError('Multiple clock publishers detected')
            if archive.error: raise RuntimeError(archive.error)
        if process.returncode != 0:
            raise RuntimeError(f'Bag player exited {process.returncode}')
        deadline = time.monotonic()+args.drain_timeout
        target = audit['expected_final_lidar_end_s']
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            report['lio_drained'] = observed['last'] is not None and observed['last'] >= target-1e-4
            odom_drained = observed['last_odom'] is not None and observed['last_odom'] >= target-1e-4
            if report['lio_drained'] and odom_drained:
                identity = transform_value(buffer.lookup_transform('map', 'camera_init', Time()))
                if np.linalg.norm(identity[0]) > 1e-7 or not np.allclose(rotation_matrix(identity[1]), np.eye(3), atol=1e-7):
                    raise ValueError('map -> camera_init must be identity')
                anchor = transform_value(buffer.lookup_transform('world', 'map', Time()))
                transform = dict(frame_id='map', map_camera_identity=True,
                    world_map_translation=anchor[0].tolist(), world_map_quaternion_xyzw=anchor[1].tolist())
                report['state'] = 'complete'
                break
            if archive.error: raise RuntimeError(archive.error)
        else:
            raise TimeoutError('Faster-LIO EOF drain incomplete; inspect play.json and LIO logs')
        print('Faster-LIO drained. Saving PCD and scan timestamps; then run nodes, cost, and merge separately.', flush=True)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        report['error'] = 'Operator interrupted replay'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=5)
        try:
            report['reconstruction'] = archive.finish(report['state']=='complete', transform)
        except Exception as error:
            report.update(state='failed', error=f'Reconstruction save failed: {error}')
        report.update(lio_output=observed, expected_final_lidar_end_s=audit['expected_final_lidar_end_s'])
        atomic_json(output/'play.json', report)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    if report['state'] != 'complete': raise SystemExit(1)


def command_status(args):
    path = mission_path(args.mission)
    mission = json.loads(path.read_text())
    result = dict(planning=mission['state'], goals=len(mission['segments']), start_policy='current_base_each_segment')
    for label, name in [('execution', 'execution.json'), ('recording', 'capture.json'), ('bag', 'bag_check.json')]:
        file = path.parent/name
        result[label] = json.loads(file.read_text()) if file.exists() else dict(state='not_run')
    offline = args.output or path.parent/'offline'
    suffix='_'+args.variant if args.variant else ''
    for label, name in [('replay','play.json'), ('reconstruction','reconstruction/reconstruction.json'), ('nodes','nodes/nodes.json'),
                         ('cost',f'node_cost{suffix}/cost.json'), ('merge',f'merged{suffix}/merge.json')]:
        file = offline/name
        result[label] = json.loads(file.read_text()) if file.exists() else dict(state='not_run')
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_offline(args):
    from Entry.mission.offline import build_nodes, compute_node_cost, merge_node_cost
    mission = mission_path(args.mission)
    directory = (args.output or mission.parent/'offline').resolve()
    if args.command == 'nodes': report = build_nodes(directory, mission)
    elif args.command == 'cost': report = compute_node_cost(directory, args.config,
        inference_frame=args.inference_frame,variant=args.variant,context_source=args.context_source)
    else: report = merge_node_cost(directory,variant=args.variant)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_view(args):
    if args.compare_nodes:
        from Entry.rendering.node_compare_view import run
    elif args.nodes_eroi:
        from Entry.rendering.eroi_view import run
    elif args.compare:
        from Entry.rendering.compare_view import run
    else:
        from Entry.rendering.view import run
    run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan'); plan.set_defaults(function=command_plan)
    plan.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    plan.add_argument('--spawn', type=Path, default=DEFAULT_SPAWN)
    plan.add_argument('--native', type=Path, default=DEFAULT_NATIVE)
    plan.add_argument('--output', type=Path)
    plan.add_argument('--seed', type=int, default=821111)
    plan.add_argument('--segments', type=int, default=4)
    plan.add_argument('--timeout', type=float, default=10.)
    plan.add_argument('--stepwise',action='store_true',help='Select each short goal only after arriving and holding at the previous goal')
    plan.add_argument('--distance-min',type=float,default=1.5)
    plan.add_argument('--distance-max',type=float,default=3.)
    plan.add_argument('--hold-seconds',type=float,default=3.)
    plan.add_argument('--max-path',type=float,default=4.)
    follow = sub.add_parser('follow'); follow.set_defaults(function=command_follow)
    follow.add_argument('--mission', type=Path, required=True)
    follow.add_argument('--timeout', type=float, default=20.)
    record = sub.add_parser('record'); record.set_defaults(function=command_record)
    record.add_argument('--mission', type=Path, required=True)
    record.add_argument('--bag', type=Path)
    validate = sub.add_parser('validate'); validate.set_defaults(function=command_validate)
    validate.add_argument('--bag', type=Path, required=True)
    validate.add_argument('--mission', type=Path)
    validate.add_argument('--report', type=Path)
    play = sub.add_parser('play'); play.set_defaults(function=command_play)
    play.add_argument('--bag', type=Path, required=True)
    play.add_argument('--mission', type=Path)
    play.add_argument('--output', type=Path)
    play.add_argument('--domain', type=int, default=184)
    play.add_argument('--rate', type=float, default=1.)
    play.add_argument('--ready-timeout', type=float, default=60.)
    play.add_argument('--drain-timeout', type=float, default=180.)
    status = sub.add_parser('status'); status.set_defaults(function=command_status)
    status.add_argument('--mission', type=Path, required=True)
    status.add_argument('--output', type=Path)
    status.add_argument('--variant',default='')
    for stage in ('nodes', 'cost', 'merge'):
        command = sub.add_parser(stage); command.set_defaults(function=command_offline)
        command.add_argument('--mission', type=Path, required=True)
        command.add_argument('--output', type=Path)
        if stage in ('cost','merge'): command.add_argument('--variant',default='')
        if stage == 'cost':
            command.add_argument('--config', type=Path, default=ROOT/'Entry/configs/config.yaml')
            command.add_argument('--inference-frame',choices=['world','map'],default='world',
                help='world aligns gravity before encoding; output retains original map XYZ')
            command.add_argument('--context-source',choices=['final','node'],default='final',
                help='node restricts encoder context to that node motion-and-hold observations')
    view = sub.add_parser('view'); view.set_defaults(function=command_view)
    view.add_argument('--mission', type=Path, required=True)
    view.add_argument('--output', type=Path)
    view.add_argument('--variant',default='')
    view.add_argument('--rviz', action='store_true')
    view.add_argument('--compare',action='store_true',help='Show gravity model and native VST on shared support positions')
    view.add_argument('--compare-nodes',action='store_true',help='Compare each independently predicted node with native VST')
    view.add_argument('--nodes-eroi',action='store_true',help='Inspect complete point clouds of each motion-and-hold node')
    view.add_argument('--cycle-seconds',type=float,default=4.)
    view.add_argument('--max-points', type=int, default=1000000)
    args = parser.parse_args()
    if getattr(args, 'rate', 1.) <= 0: parser.error('rate must be positive')
    if getattr(args, 'max_points', 1) <= 0: parser.error('max-points must be positive')
    if getattr(args,'variant','') and not args.variant.replace('_','').isalnum():parser.error('variant must use letters, digits or underscores')
    args.function(args)


if __name__ == '__main__':
    main()
