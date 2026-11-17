"""Clean this project's old simulation, then open four independent terminals."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from zoneinfo import ZoneInfo

from Entry.runtime.cost_session import ROOT,process_info,cost_module,alive,send,wait_exit
from Entry.utils.io import atomic_json


def project_process(info,domain):
    if not info or info['state']=='Z' or info['domain']!=str(domain):return False
    args=info['args']
    if not args:return False
    if cost_module(info,domain):return True
    context=info.get('project')==str(ROOT) or any(value.startswith(str(ROOT)+'/') for value in args)
    if not context:return False
    names={'gzserver','gzclient','rviz2','run_mapping_online','scan_planner_node',
           'scan_workspace_bridge','closed_loop_controller','keyboard_input','robot_state_publisher',
           'state_from_gazebo'}
    if Path(args[0]).name in names:return True
    if any(value.endswith('/unitree_gazebo/scripts/map_anchor.py') or
           value.endswith('/unitree_gazebo/sim/terminal_stage.py') for value in args):return True
    if '-m' in args and args.index('-m')+1<len(args) and args[args.index('-m')+1] in (
            'Entry.planning.local_astar_follow','Entry.experiments.astar_scan_compare'):return True
    return any(value.endswith('/ros2') for value in args) and (
        'unitree_gazebo' in args or 'keyboard_input' in args or 'scan_planner' in args)


def cleanup_previous(domain):
    processes={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=process_info(int(path.name))
            if row and row['pid']!=os.getpid():processes[row['pid']]=row
    owned={pid:row for pid,row in processes.items() if project_process(row,domain)}
    changed=True
    while changed:
        changed=False
        for pid,row in processes.items():
            if pid not in owned and row['ppid'] in owned:
                owned[pid]=row;changed=True
    # Signal outer launchers/stages once. Their normal cleanup drains workers.
    for row in owned.values():
        if row['ppid'] not in owned:send(row,signal.SIGINT)
        elif row['args'] and Path(row['args'][0]).name=='keyboard_input':
            # ros2 run may exit without forwarding a PID-only signal to its
            # interactive child. This is still a verified project process.
            send(row,signal.SIGINT)
    remaining=wait_exit(list(owned.values()),45)
    for row in remaining:send(row,signal.SIGTERM)
    remaining=wait_exit(remaining,5)
    if remaining:raise RuntimeError('Old project processes still active: '+str([r['pid'] for r in remaining]))
    return [dict(pid=row['pid'],args=row['args']) for row in owned.values()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['tiny','small'],default='tiny')
    parser.add_argument('--flight-type',type=int,choices=[1,4],default=4,
                        help='4: local-cost A* /pure_path + SCAN. 1: direct manual SCAN.')
    parser.add_argument('--cost-limit',type=float,default=.8,help='Mode-4 strict A* cost upper bound')
    parser.add_argument('--run-delay',type=float,default=3.,help='Wall seconds after run readiness')
    parser.add_argument('--crouch-delay',type=float,default=3.,help='Simulation seconds in crouch')
    parser.add_argument('--stand-delay',type=float,default=4.,help='Simulation seconds standing')
    parser.add_argument('--mode-delay',type=float,default=3.,help='Simulation seconds in RL before LIO')
    parser.add_argument('--lio-delay',type=float,default=3.,help='Wall seconds after first registered cloud')
    parser.add_argument('--timeout',type=float,default=180.,help='Wall timeout for each readiness check')
    parser.add_argument('--dry-run',action='store_true',help='Print commands without cleanup or startup')
    args=parser.parse_args()
    if not math.isfinite(args.cost_limit) or not 0<args.cost_limit<=1:
        parser.error('--cost-limit must lie in (0, 1]')
    if min(args.run_delay,args.crouch_delay,args.stand_delay,args.mode_delay,args.lio_delay)<0 or args.timeout<=0:
        parser.error('Delays must be nonnegative and timeout positive')
    commands=dict(gazebo=[str(ROOT/'run_gazebo.sh'),'planner:=scan',f'flight_type:={args.flight_type}','rviz:=false','gui:=true'],
        keyboard=[str(ROOT/'keyboard_gazebo.sh'),'-p','publish_cmd_vel:=false'],
        lio=[str(ROOT/'run_faster_lio.sh'),'-p','pcd_save.pcd_save_en:=false'],
        cost=[str(ROOT/'run_cost.sh'),'--model',args.model,'--rviz'])
    if args.flight_type==4:commands['gazebo'].append('cost_astar_limit:='+str(args.cost_limit))
    if args.dry_run:
        print(json.dumps(dict(commands=commands,delays=vars(args),mode_sequence=[2,2,7]),indent=2));return 0
    from Entry.utils.local_map import load_local_config
    cfg=load_local_config(ROOT/'Entry/configs/local.yaml',args.model)
    for path in [Path(cfg['checkpoint']),Path(cfg['network_python']),*(Path(v[0]) for v in commands.values())]:
        if not path.is_file():parser.error('Missing: '+str(path))
    if not shutil.which('gnome-terminal'):parser.error('gnome-terminal is required')
    domain=int(os.environ.get('ROS_DOMAIN_ID','0'))
    lock=open(f'/tmp/pure3d_total_terminal_{os.getuid()}_domain_{domain}.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('Another total startup is already in progress')
    def interrupt(_signum,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    run=None;node=None;stages={};ready=False
    try:
        print('Cleaning old simulation/cost processes in this workspace and ROS domain...',flush=True)
        cleaned=cleanup_previous(domain)
        run=ROOT/'outputs/entry_local'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('terminal_%Y%m%d_%H%M%S_%f')
        (run/'stages').mkdir(parents=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        master=f'http://127.0.0.1:{port}'
        if args.flight_type==4:commands['gazebo'].append('cost_astar_output:='+str(run/'astar'))
        manifest=dict(domain=domain,gazebo_master=master,flight_type=args.flight_type,model=args.model,
            manual_targets=True,scan_cost_integration=False,cost_astar_reference=args.flight_type==4,
            cost_limit=args.cost_limit,mode4_goal_fallback=False,
            pcd_save_en=False,cost_run=str(run/'cost'),
            cleaned=cleaned,processes={})
        import yaml
        planner=yaml.safe_load((ROOT/'common/SCAN-Planner/src/planner/plan_manage/config/a1.yaml').read_text())['scan_planner_node']['ros__parameters']
        frames=yaml.safe_load((ROOT/'common/unitree_gazebo/sim/config/frames.yaml').read_text())
        manifest.update(box_m=[planner['grid_map.sliding_map_size_'+a] for a in 'xyz'],
                        entry_box_source='lio_odometry',entry_box_m=cfg['box_size_m'],
                        planning_horizon_m=planner['fsm.planning_horizon'],
                        mount_base_to_lidar_xyz=frames['mount']['body_to_lidar']['translation'],
                        mount_rpy=frames['mount']['body_to_lidar']['rpy'])
        atomic_json(run/'processes.json',manifest)
        (ROOT/'outputs/entry_local/current_run.txt').write_text(str(run)+'\n')
        for src in ['a1.yaml','frames.yaml']:
            source=ROOT/('common/SCAN-Planner/src/planner/plan_manage/config/a1.yaml' if src=='a1.yaml'
                         else 'common/unitree_gazebo/sim/config/frames.yaml')
            shutil.copy(source,run/src)
        import rclpy
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from controller_manager_msgs.srv import ListControllers,SwitchController
        from std_srvs.srv import Empty
        from control_input_msgs.msg import Inputs
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
        from tf2_ros import Buffer,TransformListener
        from rclpy.time import Time
        rclpy.init(args=[])
        node=rclpy.create_node('pure3d_terminal_startup',parameter_overrides=[Parameter('use_sim_time',value=True)])
        observed=dict(odom=None,cloud=False,follow=None);buffer=Buffer();listener=TransformListener(buffer,node)
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String,'/entry/local_astar_follow/status',
            lambda m:observed.update(follow=json.loads(m.data)),durable)
        node.create_subscription(Odometry,'/Odometry_gazebo',lambda m:observed.update(odom=m),qos_profile_sensor_data)
        node.create_subscription(PointCloud2,'/faster_lio/cloud_registered',
            lambda m:observed.update(cloud=bool(m.width*m.height)),qos_profile_sensor_data)
        mode_pub=node.create_publisher(Inputs,'/control_input',10)

        def stage_record(name):
            path=run/'stages'/(name+'.json')
            if path.exists():return json.loads(path.read_text())
            return None

        def check_stages():
            for name in stages:
                row=stage_record(name)
                if row and (row['state']!='running' or not alive(row['process'])):
                    raise RuntimeError(f'{name} exited. See {run/(name+".log")}')

        def tick():
            check_stages();rclpy.spin_once(node,timeout_sec=.05)

        def wait_until(label,test,timeout=None):
            print(label,flush=True);end=time.monotonic()+(args.timeout if timeout is None else timeout)
            while time.monotonic()<end:
                tick()
                if test():return
            raise RuntimeError('Timeout: '+label)

        def delay(seconds,sim=False):
            start=node.get_clock().now().nanoseconds*1e-9 if sim else time.monotonic()
            wait_until(f'Waiting {seconds:g} '+('simulation seconds...' if sim else 'seconds...'),
                lambda:(node.get_clock().now().nanoseconds*1e-9 if sim else time.monotonic())-start>=seconds,
                max(args.timeout,seconds*5+10))

        def launch(name,command):
            print('Opening terminal: '+name,flush=True)
            environment=['env',f'ROS_DOMAIN_ID={domain}','ROS_LOCALHOST_ONLY='+os.environ.get('ROS_LOCALHOST_ONLY','1'),
                'GAZEBO_MASTER_URI='+master,'PURE3D_WS='+str(ROOT),'PYTHONPATH='+os.environ.get('PYTHONPATH',str(ROOT)),
                'PURE3D_LIO_OUTPUT='+str(run/'lio'),'/usr/bin/python3',str(Path(__file__).with_name('terminal_stage.py')),
                '--run',str(run),'--name',name,'--',*command]
            client_env=dict(os.environ)
            for key in ['LD_LIBRARY_PATH','PYTHONPATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH']:client_env.pop(key,None)
            result=subprocess.run(['gnome-terminal','--window','--title','Pure3D | '+name,'--',*environment],
                                  env=client_env,capture_output=True,text=True,timeout=15)
            if result.returncode:raise RuntimeError(result.stderr)
            stages[name]=True
            wait_until('Waiting for '+name+' terminal process...',lambda:stage_record(name) is not None,15)
            row=stage_record(name)
            manifest['processes'][name if name!='cost' else 'cost_session']=dict(
                pid=row['process']['pid'],start=row['process']['start'],command=command,log=str(run/(name+'.log')))
            manifest['processes'][name+'_terminal']=dict(pid=row['stage']['pid'],start=row['stage']['start'])
            atomic_json(run/'processes.json',manifest)

        launch('gazebo',commands['gazebo'])
        controllers=node.create_client(ListControllers,'/controller_manager/list_controllers')
        controller_order=['joint_state_broadcaster','imu_sensor_broadcaster','leg_pd_controller','unitree_guide_controller']
        required=set(controller_order)
        future=None;inactive_since=None;recovered=False
        def active():
            nonlocal future,inactive_since,recovered
            if not controllers.service_is_ready():return False
            if future is None:future=controllers.call_async(ListControllers.Request());return False
            if not future.done():return False
            values={v.name:v.state for v in future.result().controller};future=None
            if all(values.get(k)=='active' for k in required):return observed['odom'] is not None
            if required.issubset(values) and not recovered:
                if inactive_since is None:inactive_since=time.monotonic()
                if time.monotonic()-inactive_since>12:
                    print('Recovering stalled unpause/controller activation...',flush=True)
                    unpause=node.create_client(Empty,'/unpause_physics')
                    if unpause.wait_for_service(timeout_sec=1):
                        f=unpause.call_async(Empty.Request());wait_until('Unpausing Gazebo...',f.done,10)
                    switch=node.create_client(SwitchController,'/controller_manager/switch_controller')
                    if switch.wait_for_service(timeout_sec=1):
                        req=SwitchController.Request();req.activate_controllers=[k for k in controller_order if values.get(k)!='active'];req.strictness=2;req.timeout.sec=10
                        f=switch.call_async(req);wait_until('Activating controllers...',f.done,15)
                    recovered=True
            return False
        wait_until('Waiting for all four controllers and odometry...',active)
        if node.count_publishers('/clock')!=1 or node.count_publishers('/cmd_vel')!=1:
            raise RuntimeError('Expected exactly one clock and one SCAN velocity source')
        delay(args.run_delay)
        launch('keyboard',commands['keyboard'])
        wait_until('Waiting for keyboard and mode subscriber...',lambda:node.count_publishers('/control_input')>=2 and mode_pub.get_subscription_count()>0)
        def log_has(text):return (run/'gazebo.log').exists() and text in (run/'gazebo.log').read_text(errors='replace')
        def mode(command,expected):
            print(f'Robot mode command: {command}',flush=True)
            message=Inputs();message.command=command;mode_pub.publish(message)
            delay(.1);mode_pub.publish(Inputs())
            wait_until('Waiting for '+expected+'...',lambda:log_has(expected),15)
        mode(2,'Switched from passive to fixed down');delay(args.crouch_delay,True)
        mode(2,'Switched from fixed down to fixed stand');delay(args.stand_delay,True)
        mode(7,'Switched from fixed stand to RL');delay(args.mode_delay,True)
        launch('lio',commands['lio'])
        wait_until('Waiting for registered LIO cloud and world TF...',lambda:observed['cloud'] and buffer.can_transform('world','camera_init',Time()))
        delay(args.lio_delay)
        launch('cost',commands['cost']+['--output',str(run/'cost')])
        def cost_ready():
            path=run/'cost/local/status.json'
            if not path.exists():return False
            value=json.loads(path.read_text())
            if value.get('error'):raise RuntimeError(value['error'])
            return value.get('valid',False)
        wait_until('Waiting for first valid cost...',cost_ready)
        if args.flight_type==4:
            wait_until('Waiting for mode 4 cost A* and /pure_path...',lambda:
                observed['follow'] is not None and observed['follow'].get('scan_mode')==4 and
                observed['follow'].get('scan_mode_ready') and observed['follow'].get('cost_fresh') and
                node.count_publishers('/pure_path')==1)
        if node.count_publishers('/entry/local/cost_cloud')!=1:
            raise RuntimeError('Expected exactly one local cost publisher')
        local_manifest=json.loads((run/'cost/processes.json').read_text())
        manifest['processes'].update(local_manifest['processes'])
        atomic_json(run/'processes.json',manifest)
        atomic_json(run/'startup_status.json',dict(state='ready',domain=domain,mode_sequence=[2,2,7],flight_type=args.flight_type,
            cost_run=str(run/'cost'),delays=vars(args),
            publishers={t:node.count_publishers(t) for t in ['/clock','/cmd_vel','/entry/local/cost_cloud','/pure_path']}))
        ready=True
        print(f'READY: {run}\nFour terminals are independent. Publish end_pos in RViz.\n'
              'P is available in the keyboard terminal. This startup shell can now exit.',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        print('Startup interrupted or failed: '+str(exc),flush=True)
        # Close only the terminals/processes created by this incomplete startup.
        launched=[]
        for name in reversed(list(stages)):
            path=run/'stages'/(name+'.json')
            if path.exists():
                row=json.loads(path.read_text());send(row['stage'],signal.SIGINT);launched.append(row['stage'])
        remaining=wait_exit(launched,45)
        for row in remaining:send(row,signal.SIGTERM)
        wait_exit(remaining,5)
        if run:atomic_json(run/'startup_status.json',dict(state='failed',error=str(exc)))
        return 1
    finally:
        if node is not None:
            node.destroy_node()
            import rclpy
            if rclpy.ok():rclpy.shutdown()
        lock.close()
    return 0 if ready else 1


if __name__=='__main__':raise SystemExit(main())
