"""Read-only rosbag2 preflight, sensor timing and replay topic contract."""
from collections import deque
from pathlib import Path
import json
import struct
import numpy as np
import yaml


REQUIRED = {
    '/livox/lidar2': 'livox_ros_driver2/msg/CustomMsg',
    '/livox/imu': 'sensor_msgs/msg/Imu',
    '/clock': 'rosgraph_msgs/msg/Clock',
    '/Odometry_gazebo': 'nav_msgs/msg/Odometry',
    '/tf': 'tf2_msgs/msg/TFMessage',
    '/tf_static': 'tf2_msgs/msg/TFMessage',
    '/robot_description': 'std_msgs/msg/String',
    '/scan/goal': 'geometry_msgs/msg/PoseStamped',
}
RECORD_ONLY = {'/pct_path': 'nav_msgs/msg/Path', '/control_input': 'control_input_msgs/msg/Inputs'}
RECORD_TOPICS = list(REQUIRED)+list(RECORD_ONLY)+['/livox/lidar', '/ground_truth/base_w']
PLAY_TOPICS = list(REQUIRED)+['/livox/lidar', '/ground_truth/base_w']


def stamp(message):
    return message.sec+message.nanosec*1e-9


def cdr_stamp(data):
    """Time at the start of a CDR Header (or Clock), without decoding LiDAR points."""
    if len(data) < 12 or bytes(data[:2]) not in (b'\0\0', b'\0\1'):
        raise ValueError('Expected ROS 2 CDR serialization')
    sec, ns = struct.unpack_from(('<' if data[1] == 1 else '>')+'iI', data, 4)
    if ns >= 1000000000:
        raise ValueError('Invalid CDR timestamp')
    return sec+ns*1e-9


def metadata(bag):
    bag = Path(bag).resolve()
    document = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    topics = {row['topic_metadata']['name']: dict(type=row['topic_metadata']['type'], count=row['message_count'])
              for row in document['topics_with_message_count']}
    files = [bag/p for p in document['relative_file_paths']]
    return bag, document, topics, files


def inspect_bag(bag, mission=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag, meta, topics, files = metadata(bag)
    problems = []
    for name, kind in {**REQUIRED, **RECORD_ONLY}.items():
        if name not in topics or not topics[name]['count']:
            problems.append(f'Missing required topic: {name}')
        elif topics[name]['type'] != kind:
            problems.append(f'{name}: expected {kind}, found {topics[name]["type"]}')
    if problems:
        return dict(state='failed', bag=str(bag), problems=problems, topics=topics)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=meta['storage_identifier']),
                rosbag2_py.ConverterOptions('', ''))
    selected = list(REQUIRED)
    reader.set_filter(rosbag2_py.StorageFilter(topics=selected))
    classes = {name: get_message(topics[name]['type']) for name in selected if name != '/livox/lidar2'}
    timing, frames = {}, {}
    goals, odometry = [], []
    tf_edges = set()
    initial_imu = deque()
    imu_origin = last_sample = None
    stable_imu = False
    last_lidar = None
    lidar_count = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in ('/livox/lidar2', '/livox/imu', '/Odometry_gazebo', '/scan/goal', '/clock'):
            t = cdr_stamp(data)
            info = timing.setdefault(topic, dict(first=t, last=t, count=0, backwards=0, duplicate_stamps=0, maximum_gap_s=0.))
            if info['count']:
                info['backwards'] += int(t < info['last'])
                info['duplicate_stamps'] += int(t == info['last'])
                info['maximum_gap_s'] = max(info['maximum_gap_s'], t-info['last'])
            info.update(last=t, count=info['count']+1)
        if topic == '/livox/lidar2':
            last_lidar = data
            lidar_count += 1
            continue
        if topic == '/robot_description':
            continue
        message = deserialize_message(data, classes[topic])
        if hasattr(message, 'header'):
            frames[topic] = message.header.frame_id
        if topic == '/livox/imu':
            if imu_origin is None:
                imu_origin = t
            if t-imu_origin <= 10. and (last_sample is None or t-last_sample >= .019):
                last_sample = t
                a, g = message.linear_acceleration, message.angular_velocity
                vector = np.array([a.x, a.y, a.z])
                if not (.948*9.81 < np.linalg.norm(vector) < 1.05*9.81 and
                        np.linalg.norm([g.x, g.y, g.z]) < .08):
                    initial_imu.clear()
                else:
                    if initial_imu and t-initial_imu[-1][0] > .1:
                        initial_imu.clear()
                    initial_imu.append((t, vector))
                    while initial_imu and t-initial_imu[0][0] > 1.1:
                        initial_imu.popleft()
                    if len(initial_imu) >= 20 and t-initial_imu[0][0] >= 1.:
                        stable_imu |= bool(np.ptp(np.array([v for _, v in initial_imu]), axis=0).max() < .4)
        elif topic == '/Odometry_gazebo':
            if message.header.frame_id != 'world' or message.child_frame_id != 'base':
                problems.append('Gazebo odometry must use world -> base')
            if not odometry or t-odometry[-1][0] >= .049:
                p = message.pose.pose.position
                odometry.append((t, np.array([p.x, p.y, p.z])))
        elif topic == '/scan/goal':
            p = message.pose.position
            point = np.array([p.x, p.y, p.z])
            if message.header.frame_id != 'world':
                problems.append('Recorded mission goals must use world')
            if not goals or np.linalg.norm(point-goals[-1]['position']) >= .01:
                goals.append(dict(stamp=t, position=point.tolist()))
        elif topic in ('/tf', '/tf_static'):
            for transform in message.transforms:
                parent, child = transform.header.frame_id, transform.child_frame_id
                tf_edges.add((parent, child))
                if child in ('map', 'camera_init', 'body'):
                    problems.append('Bag contains estimated map/camera_init/body TF; record with Faster-LIO stopped')
    for name, info in timing.items():
        if info['backwards']:
            problems.append(f'{name} timestamps move backwards')
        if name in ('/livox/lidar2', '/livox/imu') and info['duplicate_stamps']:
            problems.append(f'{name} contains duplicate sensor timestamps')
    for name, limit in [('/livox/lidar2', .5), ('/livox/imu', .1)]:
        if timing[name]['maximum_gap_s'] > limit:
            problems.append(f'{name} has a sensor gap larger than {limit} s')
    if not stable_imu:
        problems.append('No 1 s stationary IMU window in the first 10 s')
    if ('world', 'base') not in tf_edges:
        problems.append('Missing recorded world -> base TF')
    if ('base', 'laser_livox') not in tf_edges or ('laser_livox', 'livox_imu_link') not in tf_edges:
        problems.append('Missing recorded MID360 static mounting TF')
    parents = {}
    for parent, child in tf_edges:
        parents.setdefault(child, set()).add(parent)
    if any(len(value) != 1 for value in parents.values()):
        problems.append('Recorded TF has a child with multiple parents')
    for child in parents:
        seen, current = set(), child
        while current in parents and len(parents[current]) == 1:
            if current in seen:
                problems.append('Recorded TF contains a cycle')
                break
            seen.add(current)
            current = next(iter(parents[current]))
    lidar_time = timing.get('/livox/lidar2', {})
    first = max(timing['/livox/lidar2']['first'], timing['/livox/imu']['first'])
    head_s = goals[0]['stamp']-first if goals else 0.
    if head_s < 10.:
        problems.append('Need at least 10 s of initial sensor recording before the first goal')
    initial_positions = [p for t, p in odometry if first <= t <= first+10.]
    initial_motion = float(np.max(np.linalg.norm(np.asarray(initial_positions)-initial_positions[0], axis=1))) if initial_positions else None
    if initial_motion is None or initial_motion > .1:
        problems.append('Initial 10 s was not stationary within 0.1 m')
    expected_end = lidar_time.get('last')
    if last_lidar is not None:
        last = deserialize_message(last_lidar, get_message(REQUIRED['/livox/lidar2']))
        expected_end = stamp(last.header.stamp)+max((p.offset_time for p in last.points), default=0)*1e-9
        frames['/livox/lidar2'] = last.header.frame_id
        if expected_end > timing['/livox/imu']['last']:
            problems.append('Last LiDAR scan is not covered by the final IMU timestamp')
    execution = None
    tail_s = None
    if mission is None and (bag.parent/'mission.json').exists():
        mission = bag.parent/'mission.json'
    if mission is not None:
        mission = Path(mission).resolve()
        if mission.is_dir(): mission = mission/'mission.json'
        planned = json.loads(mission.read_text())
        execution_path = mission.parent/'execution.json'
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            arrived = [e for e in execution['events'] if e['event']=='arrived']
            arrivals = [e['stamp'] for e in arrived]
            expected = len(planned['segments'])
            sequence_ok = [e['segment'] for e in arrived] == list(range(expected))
            if execution['state'] == 'complete' and len(arrivals) == expected and sequence_ok:
                tail_s = min(timing['/livox/lidar2']['last'], timing['/livox/imu']['last'])-arrivals[-1]
                if tail_s < 5.:
                    problems.append('Need at least 5 s of sensors after the final measured arrival')
                tail_positions = [p for t, p in odometry if arrivals[-1] <= t <= arrivals[-1]+5.]
                if not tail_positions or np.max(np.linalg.norm(np.asarray(tail_positions)-tail_positions[0], axis=1)) > .2:
                    problems.append('Final 5 s was not stationary within 0.2 m')
                if not odometry or odometry[-1][0] < arrivals[-1]+4.9:
                    problems.append('Gazebo odometry does not cover the final stationary interval')
                for event, segment in zip(arrived, planned['segments']):
                    measured = [(t, p) for t, p in odometry if event['stamp']-.5 <= t <= event['stamp']+.01]
                    if (len(measured) < 2 or measured[-1][0]-measured[0][0] < .4 or
                            any(np.linalg.norm(p-segment['goal_body_xyz']) > .2+1e-6 for _, p in measured)):
                        problems.append(f'Segment {event["segment"]} arrival lacks 0.5 s of nearby recorded truth')
            else:
                problems.append('Mission execution did not report all measured arrivals')
            if len(goals) != expected or any(np.linalg.norm(np.asarray(a['position'])-b['goal_body_xyz']) > .01
                                             for a, b in zip(goals, planned['segments'])):
                problems.append('Recorded goals do not match the planned goal sequence')
            if planned.get('planning_mode')=='stepwise_short_goals':
                holds=[e for e in execution['events'] if e['event']=='hold_complete']
                requests=[e for e in execution['events'] if e['event']=='planning_requested']
                hold_s=planned['settings']['hold_seconds']
                if ([e['segment'] for e in holds]!=list(range(expected)) or
                        any(e['stationary_s']+1e-6<hold_s for e in holds)):
                    problems.append('Missing required stationary hold after a short goal')
                if len(requests)!=expected or any(requests[i]['stamp']<holds[i-1]['stamp'] for i in range(1,min(len(requests),len(holds)+1))):
                    problems.append('A new short goal was selected before the previous hold completed')
        else:
            problems.append('Missing execution.json; final 5 s cannot be verified')
    else:
        problems.append('Provide --mission to verify measured arrivals and the final 5 s')
    sensor_signature = [dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files]
    return dict(state='passed' if not problems else 'failed', bag=str(bag), topics=topics,
        files=sensor_signature, problems=sorted(set(problems)), sensor_timing=timing, frames=frames,
        stationary_imu_window=stable_imu, initial_stationary_displacement_m=initial_motion,
        initial_recording_s=head_s, final_stationary_recording_s=tail_s, distinct_goals=len(goals),
        goals=goals, expected_final_lidar_end_s=expected_end, lidar_frames=lidar_count,
        tf_edges=[list(e) for e in sorted(tf_edges)],
        execution_complete=execution.get('state')=='complete' if execution is not None else None)
