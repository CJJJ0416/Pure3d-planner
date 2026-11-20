"""Real-world sensor bags through the existing Faster-LIO and local Entry nodes."""
import argparse
from collections import Counter, OrderedDict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sqlite3
import struct
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from common.io import atomic_json, cloud_xyz
from common.paths import ROOT, CONFIG_DIR


SENSORS = {'/livox/lidar': 'livox_ros_driver2/msg/CustomMsg',
           '/livox/imu': 'sensor_msgs/msg/Imu'}


def preflight(bag):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    from livox_ros_driver2.msg import CustomMsg
    meta = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    if meta['storage_identifier'] != 'sqlite3' or len(meta['relative_file_paths']) != 1:
        raise ValueError('The real-world adapter expects one SQLite3 bag file')
    path = (bag/meta['relative_file_paths'][0]).resolve()
    result, initial = {}, []
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        topics = {name: (idx, kind) for idx, name, kind in db.execute('SELECT id,name,type FROM topics')}
        if '/clock' in topics:
            raise ValueError('Input already contains /clock; this adapter owns the replay clock')
        for topic, kind in SENSORS.items():
            if topic not in topics or topics[topic][1] != kind:
                raise ValueError(f'Expected {topic}: {kind}')
            stamps, offsets, records = [], [], []
            first_payload = last_payload = None
            for record, data in db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics[topic][0],)):
                endian = '<' if data[1] == 1 else '>'
                sec, ns = struct.unpack_from(endian+'iI', data, 4)
                stamp = sec*10**9+ns
                stamps.append(stamp); records.append(record); offsets.append((record-stamp)/1e9)
                first_payload = first_payload or data
                last_payload = data
                if topic == '/livox/imu' and stamp-stamps[0] < 3_000_000_000:
                    m = deserialize_message(data, Imu)
                    initial.append(([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
                                    [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]))
            a = np.asarray(stamps, np.int64)
            if len(a) < 2 or np.any(np.diff(a) <= 0):
                raise ValueError(f'{topic}: nonmonotonic or insufficient sensor timestamps')
            delta = np.diff(a)/1e9
            if delta.max() > (.1 if topic == '/livox/imu' else .5):
                raise ValueError(f'{topic}: excessive sensor gap')
            # rosbag2 --clock follows record time. Fail rather than hide an epoch mismatch.
            if min(offsets) < -.1 or max(offsets) > .5:
                raise ValueError(f'{topic}: record clock and sensor clock need explicit normalization')
            info = dict(count=len(a), first_ns=int(a[0]), last_ns=int(a[-1]),
                        first_record_ns=int(records[0]), last_record_ns=int(records[-1]),
                        max_gap_s=float(delta.max()), median_interval_s=float(np.median(delta)),
                        record_minus_header_s=[float(min(offsets)), float(np.median(offsets)), float(max(offsets))])
            if topic == '/livox/lidar':
                first = deserialize_message(first_payload, CustomMsg)
                last = deserialize_message(last_payload, CustomMsg)
                info.update(frame=first.header.frame_id, first_scan_lines=sorted({p.line for p in first.points}),
                            last_scan_end_ns=int(a[-1])+max(p.offset_time for p in last.points))
            result[topic] = info
    acc = np.array([x[0] for x in initial]); gyro = np.array([x[1] for x in initial])
    mean = acc.mean(0); norm = float(np.linalg.norm(mean))
    # The installed Faster-LIO normalizes acceleration by its initialized mean norm.
    # Accept the native Livox g unit and SI units; do not rescale messages twice.
    if not (.8 < norm < 1.2 or 8 < norm < 12) or np.max(np.std(acc, axis=0))/norm > .1 or np.max(np.linalg.norm(gyro, axis=1)) > .15:
        raise ValueError('Initial three seconds are unsuitable for static gravity initialization')
    if result['/livox/imu']['last_ns'] < result['/livox/lidar']['last_scan_end_ns']:
        raise ValueError('Final LiDAR scan is not fully covered by IMU')
    result['initial_imu'] = dict(mean_acc=mean.tolist(), mean_acc_norm=norm,
        std_acc=acc.std(0).tolist(), max_gyro_norm=float(np.linalg.norm(gyro, axis=1).max()),
        native_acceleration_unit='g' if norm < 2 else 'm/s^2', normalization='existing Faster-LIO IMU initialization')
    result['clock'] = dict(source='rosbag2 record timestamps', publisher='ros2 bag play --clock 100',
                           original_sensor_stamps_preserved=True)
    return result


def prepare(run, bag, model, report, profile='fastlio_official'):
    from scipy.spatial.transform import Rotation
    from Simulator.unitree_gazebo.scripts.frame_config import compose, inverse, transform_from_config, prepare_lio_config
    real_config = ROOT/'Entrance/config/real'
    frames = yaml.safe_load((real_config/(profile+'_frames.yaml')).read_text())
    mount = frames['mount']
    imu_body = inverse(compose(transform_from_config(mount['body_to_lidar']), transform_from_config(mount['lidar_to_imu'])))
    z = np.array(report['initial_imu']['mean_acc']); z /= np.linalg.norm(z)
    forward = Rotation.from_quat(imu_body[1]).apply([1., 0., 0.])
    x = forward-z*np.dot(z, forward); x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(z, x), z])
    translation = -rotation @ imu_body[0]
    frames['anchor'].update(translation=translation.tolist(), rpy=Rotation.from_matrix(rotation).as_euler('xyz').tolist())
    frame_path = run/'frames.yaml'; frame_path.write_text(yaml.safe_dump(frames, sort_keys=False))
    params = yaml.safe_load((real_config/'faster_lio.yaml').read_text())
    p = params['/**']['ros__parameters']; p['use_sim_time'] = True
    overrides = yaml.safe_load((real_config/(profile+'_lio_overrides.yaml')).read_text())
    for section, values in overrides.items():
        p[section].update(values)
    p['common'].update(lid_topic='/livox/lidar', imu_topic='/livox/imu')
    lio_path = run/'faster_lio.yaml'; lio_path.write_text(yaml.safe_dump(params))
    prepare_lio_config(frame_path, lio_path, lio_path)
    cfg = yaml.safe_load((CONFIG_DIR/'real_bag.yaml').read_text())
    cfg.update(model=model, use_sim_time=True)
    (run/'entry.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    report['alignment'] = dict(world_from_camera_rotation=rotation.tolist(), world_from_camera_translation=translation.tolist(),
        origin='initial IMU frame' if profile=='fastlio_official' else 'initial robot base', heading='initial body forward projected onto gravity plane',
        gravity_source='initial three-second sensor mean, no motion-capture input',
        gravity_residual=float(np.linalg.norm(rotation@z-[0., 0., 1.])))
    atomic_json(run/'preflight.json', report)
    atomic_json(run/'profile.json', dict(experiment_type='real-world', execution='recorded-sensor replay',
        entry_config=str(CONFIG_DIR/'real_bag.yaml'), lio_config=str(real_config/'faster_lio.yaml'),
        dataset_profile=profile, dataset_lio_overrides=str(real_config/(profile+'_lio_overrides.yaml')),
        calibration=str(real_config/(profile+'_frames.yaml')), model=model,
        blind_radius_m=float(p['preprocess']['blind']), blind_center='LiDAR sensor origin',
        clock_override=dict(use_sim_time=True, reason='Replay original real sensor timestamps using /clock'),
        gazebo_required=False, ground_truth_used_for_alignment=False))
    return frame_path, lio_path


def observe(run):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    rclpy.init(); node = Node('entry_bag_observer', parameter_overrides=[Parameter('use_sim_time', value=True)])
    durable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    counts, raw, latest, last_stamp = Counter(), OrderedDict(), {}, {}
    samples = run/'snapshots'; samples.mkdir(exist_ok=True)
    state = dict(clock_ns=0, last_saved_ns=0, last_odom_stamp=0., snapshots=[], errors=[], initial_clock_resets=[])
    status_log = (run/'status_history.jsonl').open('w', buffering=1)
    odom_log = (run/'odometry.jsonl').open('w', buffering=1)
    cost_log = (run/'cost_publications.jsonl').open('w', buffering=1)

    def stamp(m): return m.header.stamp.sec*10**9+m.header.stamp.nanosec

    def cloud(m, label):
        counts[label] += 1; t = stamp(m)
        # Empty publications explicitly clear stale output; their zero stamp is not sensor time.
        if m.width*m.height:
            if label in last_stamp and t < last_stamp[label]: state['errors'].append(label+': timestamp moved backwards')
            last_stamp[label] = t
        latest[label] = dict(stamp_ns=t, points=m.width*m.height, frame=m.header.frame_id)
        if label == 'raw':
            raw[t] = cloud_xyz(m)
            while len(raw) > 120: raw.popitem(last=False)
        elif label == 'cost' and m.width*m.height:
            xyz = cloud_xyz(m)
            offset = next(f.offset for f in m.fields if f.name == 'trav_cost')
            cost = np.ndarray((m.width*m.height,), dtype='<f4', buffer=m.data, offset=offset, strides=(m.point_step,)).copy()
            if not np.isfinite(xyz).all() or not np.isfinite(cost).all(): state['errors'].append('Nonfinite cost cloud')
            cost_log.write(json.dumps(dict(stamp_ns=t, points=len(xyz), min=float(cost.min()), max=float(cost.max()),
                mean=float(cost.mean()), ros_age_s=(state['clock_ns']-t)/1e9))+'\n')
            if t in raw and t-state['last_saved_ns'] >= 5_000_000_000:
                path = samples/f'snapshot_{len(state["snapshots"]):03d}.npz'
                np.savez_compressed(path, raw=raw[t], filtered=xyz, cost=cost, stamp_ns=np.int64(t))
                state['snapshots'].append(dict(path=str(path.relative_to(run)), stamp_ns=t, raw_points=len(raw[t]), kept_points=len(xyz)))
                state['last_saved_ns'] = t

    def odometry(m):
        counts['odometry'] += 1; p = m.pose.pose.position; q = m.pose.pose.orientation
        state['last_odom_stamp'] = stamp(m)/1e9
        odom_log.write(json.dumps(dict(stamp_ns=stamp(m), position=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w]))+'\n')

    def status(m):
        counts['status'] += 1; value = json.loads(m.data)
        status_log.write(json.dumps(value, allow_nan=False)+'\n')
        state['last_status'] = value

    def clock(m):
        counts['clock'] += 1; value = m.clock.sec*10**9+m.clock.nanosec
        if value < state['clock_ns']:
            event=dict(previous_ns=state['clock_ns'],current_ns=value,registered_scans=counts['registered'])
            if counts['registered']:state['errors'].append(dict(error='Clock moved backwards',**event))
            else:state['initial_clock_resets'].append(event)
        state['clock_ns'] = value

    node.create_subscription(Clock, '/clock', clock, qos_profile_sensor_data)
    node.create_subscription(Odometry, '/faster_lio/Odometry', odometry, qos_profile_sensor_data)
    from nav_msgs.msg import Path as PathMessage
    def path_message(m):
        counts['lio_path']+=1
        latest['lio_path']=dict(poses=len(m.poses),frame=m.header.frame_id,stamp_ns=stamp(m))
    node.create_subscription(PathMessage,'/faster_lio/path',path_message,qos_profile_sensor_data)
    node.create_subscription(PointCloud2, '/faster_lio/cloud_registered', lambda m: cloud(m,'registered'), qos_profile_sensor_data)
    for label, topic in [('raw','cloud'),('cost','cost_cloud'),('removed','knn_removed')]:
        node.create_subscription(PointCloud2, '/entry/local/'+topic, lambda m,label=label: cloud(m,label), durable)
    node.create_subscription(String, '/entry/local/status', status, durable)
    node.create_timer(.5, lambda: atomic_json(run/'audit.json', dict(**state, counts=dict(counts), latest=latest)))
    atomic_json(run/'observer_ready.json', dict(ready=True))
    try: rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): pass
    finally:
        atomic_json(run/'audit.json', dict(**state, counts=dict(counts), latest=latest))
        for stream in (status_log, odom_log, cost_log): stream.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


def replay(args):
    from common.visualization.bag_view import live_config
    bag = args.bag.resolve()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    run = (args.output or ROOT/'outputs/real_world'/args.profile/bag.name/(stamp+'-'+args.model)).resolve()
    run.mkdir(parents=True, exist_ok=False)
    report = preflight(bag); frames, lio = prepare(run, bag, args.model, report, args.profile)
    env = dict(os.environ, ROS_LOG_DIR=str(run/'ros_log'), PURE3D_LIO_OUTPUT=str(run/'lio'))
    (run/'lio').mkdir(); children = {}; logs = []
    mapping = Path(env.get('PURE3D_LIO_WS', str(Path.home()/'livox_ws')))/'install/faster_lio/lib/faster_lio/run_mapping_online'
    state = dict(state='starting', experiment_type='real-world', dataset_profile=args.profile, execution='bag replay', bag=str(bag), output=str(run), model=args.model, domain=int(env.get('ROS_DOMAIN_ID','186')), rate=args.rate)

    def spawn(name, command):
        stream = (run/(name+'.log')).open('w'); logs.append(stream)
        children[name] = subprocess.Popen(command, cwd=run, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)

    def check():
        for name, child in children.items():
            if name not in ('player','rviz') and child.poll() is not None:
                raise RuntimeError(f'{name} exited ({child.returncode}); see {run/(name+".log")}')
        path = run/'local/status.json'
        value = json.loads(path.read_text()) if path.exists() else {}
        if value.get('error'): raise RuntimeError(value['error'])
        return value

    def wait_for(predicate, timeout, label):
        until = time.monotonic()+timeout
        while not predicate():
            check()
            if time.monotonic() > until: raise TimeoutError(label)
            time.sleep(.2)

    def stop(name):
        child = children.pop(name, None)
        if child is None or child.poll() is not None: return
        os.killpg(child.pid, signal.SIGINT)
        try: child.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=10)

    try:
        spawn('observer', ['/usr/bin/python3','-u','-m','common.bag_tools.real_bag','observe','--run',str(run)])
        wait_for(lambda: (run/'observer_ready.json').exists(), 30, 'Observer startup')
        spawn('anchor', ['/usr/bin/python3',str(ROOT/'Simulator/unitree_gazebo/scripts/map_anchor.py'),'--config',str(frames),'--ros-args','-p','use_sim_time:=true'])
        spawn('lio', [str(mapping),'--ros-args','-r','__ns:=/faster_lio','--params-file',str(lio),'-r','Odometry:=raw_odometry','-r','/tf:=/faster_lio/raw_tf'])
        spawn('entry', ['/usr/bin/python3','-u','-m','common.cost.local','--config',str(run/'entry.yaml'),'--model',args.model,'--output',str(run/'local')])
        wait_for(lambda: (run/'local/model.json').exists(), 180, 'GPU model startup')
        if args.rviz:
            config = live_config(run)
            spawn('rviz', ['rviz2','-d',str(config),'--ros-args','-r','__node:=entry_bag_rviz','-p','use_sim_time:=true'])
        time.sleep(2)
        spawn('player', ['ros2','bag','play',str(bag),'--topics','/livox/lidar','/livox/imu','--clock','100','--rate',str(args.rate),'--delay','2','--disable-keyboard-controls'])
        state['state'] = 'replaying'; atomic_json(run/'run.json', state)
        print(f'Replaying {bag.name}, {args.model}, domain={state["domain"]}; {run}', flush=True)
        last_print = 0.
        while children['player'].poll() is None:
            value = check()
            if time.monotonic()-last_print > 10:
                print(f"scans={value.get('accepted_scans',0)} points={value.get('points',0)} predictions={value.get('predictions',0)} valid={value.get('valid',False)}", flush=True)
                last_print = time.monotonic()
            time.sleep(.5)
        if children['player'].returncode != 0: raise RuntimeError('Bag player failed')
        last_scan = report['/livox/lidar']['last_scan_end_ns']/1e9
        wait_for(lambda: check().get('pose_stamp',0) >= last_scan-.003, 15, 'LIO did not drain the final scan')
        state['state'] = 'complete'; state['last_scan_end_s'] = last_scan
    except BaseException as error:
        state.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        for name in ('player','entry','lio','anchor','observer','rviz'): stop(name)
        for stream in logs: stream.close()
        if (run/'rate_changes.json').exists():
            state['rate_history'] = json.loads((run/'rate_changes.json').read_text())
        atomic_json(run/'run.json', state)
        print(json.dumps(state), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--bag', type=Path, default=ROOT/'datasets/FAST_LIO2/ros2/HKU_MB_2020-09-20-13-34-51')
    run.add_argument('--profile',choices=['fastlio_official','jeplo'],default='fastlio_official')
    run.add_argument('--model', choices=['tiny','small'], default='tiny')
    run.add_argument('--rate', type=float, default=1.)
    run.add_argument('--output', type=Path); run.add_argument('--rviz', action='store_true')
    obs = commands.add_parser('observe'); obs.add_argument('--run', type=Path, required=True)
    view = commands.add_parser('view')
    view.add_argument('--run',type=Path,required=True);view.add_argument('--snapshot',type=int,default=-1)
    view.add_argument('--cycle-seconds',type=float,default=0.);view.add_argument('--rviz',action='store_true')
    view.add_argument('--plot',action='store_true')
    args = parser.parse_args()
    if args.command == 'observe': observe(args.run)
    elif args.command == 'view':
        from common.visualization.bag_view import plot,view
        if args.plot: print(plot(args.run.resolve(),args.snapshot))
        else: view(args.run.resolve(),args.snapshot,args.cycle_seconds,args.rviz)
    else:
        if not np.isfinite(args.rate) or args.rate <= 0: parser.error('--rate must be positive')
        replay(args)


if __name__ == '__main__': main()
