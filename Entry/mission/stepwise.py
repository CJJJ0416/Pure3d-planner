"""Near-goal selection after each measured arrival and stationary wait."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from Entry.utils.io import atomic_json
from Entry.mission.mission_core import ShortGoalState, sha256


def initialize(args,base,output):
    if (not 0<args.distance_min<args.distance_max or args.hold_seconds<=0 or
            args.max_path<args.distance_min or args.segments<1):
        raise ValueError('Invalid short-goal distance, hold or path limit')
    output.mkdir(parents=True,exist_ok=False)
    scene=args.scene.resolve();spawn=args.spawn.resolve()
    metadata=json.loads(spawn.read_text())
    report=dict(state='ready',planning_mode='stepwise_short_goals',frame_id='world',
        scene=str(scene),spawn=str(spawn),vst=str(scene/'vst_nominal.pickle'),native_dir=str(args.native.resolve()),
        planner='native VST A* + quintic GPMP; next endpoint selected only after stationary hold',
        source_sha256=dict(scene=sha256(scene/'scene.json'),vst=sha256(scene/'vst_nominal.pickle')),
        settings=dict(seed=args.seed,segments=args.segments,floor=metadata['floor'],floor_z=metadata['ground_z'],
            endpoint_distance_m=[args.distance_min,args.distance_max],maximum_path_m=args.max_path,
            hold_seconds=args.hold_seconds,cost_threshold=15.,body_height_m=.4,forward_half_cone_deg=70.),
        start_policy='current_robot_base',node_ownership='all scans during motion and hold; boundary at next path publication',
        requested_start_body_xyz=base['position'],segments=[])
    atomic_json(output/'mission.json',report);atomic_json(output/'planning_base.json',base)
    return report


def follow(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from nav_msgs.msg import Odometry,Path as PathMessage
    from geometry_msgs.msg import PoseStamped
    from control_input_msgs.msg import Inputs
    from std_msgs.msg import String
    from common.unitree_gazebo.scripts.frame_config import rotation_matrix
    from Entry.mission.bag_mission import read_mission, native_plan
    from Entry.utils.io import pose_value, seconds, ros_stamp

    path,mission=read_mission(args.mission);directory=path.parent
    if (directory/'execution.json').exists():raise ValueError('Use a new mission for a new capture')
    settings=mission['settings'];state=ShortGoalState(settings['segments'],settings['hold_seconds'])
    rclpy.init(args=[]);node=Node('entry_short_goal_follow',parameter_overrides=[Parameter('use_sim_time',value=True)])
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    path_pub=node.create_publisher(PathMessage,'/pct_path',qos)
    goal_pub=node.create_publisher(PoseStamped,'/scan/goal',10)
    mode_pub=node.create_publisher(Inputs,'/control_input',10)
    status_pub=node.create_publisher(String,'/entry/mission_status',qos)
    latest=dict(position=None,quaternion=None,stamp=None,wall=None)
    def odometry(message):
        if message.header.frame_id!='world' or message.child_frame_id!='base':
            state.fail('Expected Gazebo world/base odometry',latest['stamp']);return
        position,quaternion=pose_value(message.pose.pose)
        latest.update(position=position,quaternion=quaternion,stamp=seconds(message.header.stamp),wall=time.monotonic())
    subscription=node.create_subscription(Odometry,'/Odometry_gazebo',odometry,
        QoSProfile(depth=20,reliability=ReliabilityPolicy.BEST_EFFORT))
    pool=ThreadPoolExecutor(max_workers=1);future=None;planning_base=None;paused=False
    visited=[mission['requested_start_body_xyz']]
    discovery=time.monotonic();last_save=0.
    def pause(value):
        nonlocal paused
        if paused!=value:
            mode_pub.publish(Inputs(command=11));paused=value
    def save():
        report=dict(state.report(),mission=str(path),frame_id='world',planning_mode='stepwise_short_goals',
            node_ownership=mission['node_ownership'],paused=paused)
        atomic_json(directory/'execution.json',report);status_pub.publish(String(data=json.dumps(report)))
        mission['state']='complete' if state.state=='complete' else ('failed' if state.state=='failed' else 'ready')
        atomic_json(path,mission)
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.03)
            if time.monotonic()-discovery<1.:continue
            owners=node.get_publishers_info_by_topic('/pct_path')
            if len(owners)!=1 or owners[0].node_name!=node.get_name():state.fail('Another /pct_path publisher exists',latest['stamp'])
            if latest['position'] is None:
                if time.monotonic()-discovery>args.timeout:state.fail('Gazebo odometry unavailable',None)
            elif time.monotonic()-latest['wall']>2.:state.fail('Gazebo odometry stopped',latest['stamp'])
            elif state.state not in ('complete','failed'):
                capture_file=directory/'capture.json'
                capture=json.loads(capture_file.read_text()) if capture_file.exists() else {}
                if state.state=='waiting':
                    beginning=capture.get('first_sensor_stamp')
                    ready=(capture.get('state')=='recording' and beginning is not None and
                           latest['stamp']-beginning>=10.5 and path_pub.get_subscription_count()>0)
                    index=state.update(latest['position'],latest['stamp']) if ready else None
                elif capture.get('state')!='recording' or time.time()-capture_file.stat().st_mtime>3.:
                    state.fail('Mission recorder stopped',latest['stamp']);index=None
                else:
                    before=state.state;index=state.update(latest['position'],latest['stamp'])
                    if before=='following' and state.state=='holding':
                        pause(True)
                        print(f'Arrived at goal {state.index+1}; holding stationary {settings["hold_seconds"]} s',flush=True)
                if index is not None:
                    planning_base=latest['position'].copy()
                    heading=float(np.arctan2(rotation_matrix(latest['quaternion'])[1,0],rotation_matrix(latest['quaternion'])[0,0]))
                    destination=directory/'executed_paths'/f'short_{index:02d}'
                    atomic_json(destination.parent/(destination.name+'_base.json'),dict(frame_id='world',stamp=latest['stamp'],base_xyz=planning_base.tolist()))
                    future=pool.submit(native_plan,Path(mission['scene']),Path(mission['spawn']),Path(mission['native_dir']),
                        destination,planning_base,settings['seed']+index*997,1,None,
                        settings['endpoint_distance_m'],settings['maximum_path_m'],heading,visited)
                if state.state=='planning' and future is not None and future.done():
                    planned=future.result();future=None
                    if np.linalg.norm(latest['position']-planning_base)>.3:raise RuntimeError('Base moved during short-goal selection')
                    row=planned['segments'][0];row['id']=state.index
                    mission['segments'].append(row)
                    body=np.load(row['body']);delta=np.diff(body[:,:2],axis=0)
                    yaw=np.r_[np.arctan2(delta[:,1],delta[:,0]),0.];yaw[-1]=yaw[-2]
                    message=PathMessage();message.header.frame_id='world';message.header.stamp=ros_stamp(latest['stamp'])
                    for xyz,angle in zip(body,yaw):
                        pose=PoseStamped();pose.header=message.header
                        pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=map(float,xyz)
                        pose.pose.orientation.z,pose.pose.orientation.w=float(np.sin(angle/2)),float(np.cos(angle/2))
                        message.poses.append(pose)
                    goal=PoseStamped();goal.header=message.header
                    goal.pose.position.x,goal.pose.position.y,goal.pose.position.z=map(float,row['goal_body_xyz'])
                    goal.pose.orientation.w=1.
                    path_pub.publish(message);goal_pub.publish(goal);pause(False)
                    state.published(row,latest['stamp'],planning_base)
                    state.events[-1]['executed_plan']=str(Path(row['body']).parent/'mission.json')
                    visited.append(row['goal_body_xyz'])
                    print(f'Published short goal {state.index+1}/{state.total}: {row["length_m"]:.2f} m',flush=True)
            if time.monotonic()-last_save>.5 or state.state in ('complete','failed'):
                save();last_save=time.monotonic()
            if state.state=='failed':raise RuntimeError(state.error)
            if state.state=='complete':
                print('All short goals and holds complete. Robot remains paused; record another 5 s.',flush=True)
                break
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
        state.fail('Operator interrupted short-goal mission',latest['stamp'])
    except Exception as error:
        state.fail(str(error),latest['stamp']);raise
    finally:
        try:pause(True)
        except Exception:pass
        pool.shutdown(wait=True);save();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
