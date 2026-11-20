"""整体启动方案：清理本工作区旧会话，再依次打开四个独立终端。

Usage：./Entrance/shell/total_terminal.sh --scene terrain-prepared --dry-run
配置：Entrance/config/startup.yaml、Entrance/config/scenes/*.yaml、Entrance/config/local_cost.yaml。
"""
import argparse
from datetime import datetime
import fcntl
import json
import importlib.util
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from zoneinfo import ZoneInfo

from common.cost.session import ROOT,process_info,cost_module,alive,send,wait_exit
from common.io import atomic_json
_spec=importlib.util.spec_from_file_location('pure3d_scenes',ROOT/'Entrance/launch/sim/_internal/scenes.py')
_scenes=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scenes)
default_scene,scene_names,scene_plan=_scenes.default_scene,_scenes.scene_names,_scenes.scene_plan


def project_process(info,domain):
    # 排除僵尸进程、空命令行与其它 ROS 域；cost 会话由共享运行库的规则识别。
    if not info or info['state']=='Z' or info['domain']!=str(domain):return False
    args=info['args']
    if not args:return False
    if cost_module(info,domain):return True
    # 只处理命令行里出现本工作区路径、或进程记录中标注了本工作区的进程。
    context=info.get('project')==str(ROOT) or any(value.startswith(str(ROOT)+'/') for value in args)
    if not context:return False
    # 常见的直接可执行进程（Gazebo、RViz、LIO、SCAN、键盘、状态发布等）。
    names={'gzserver','gzclient','rviz2','run_mapping_online','scan_planner_node',
           'scan_workspace_bridge','closed_loop_controller','ego_local_trajectory_server',
           'keyboard_input','robot_state_publisher',
           'state_from_gazebo'}
    if Path(args[0]).name in names:return True
    # 通过脚本路径识别的内部工具（地图锚定与阶段托管）。
    if any(value.endswith('/unitree_gazebo/scripts/map_anchor.py') or
           value.endswith('/unitree_gazebo/sim/terminal_stage.py') or
           value.endswith('/Entrance/shell/_internal/terminal_stage.py') for value in args):return True
    # 以 python -m 模块方式启动的跟踪/对照实验进程。
    if '-m' in args and args.index('-m')+1<len(args) and args[args.index('-m')+1] in (
            'common.planning.local_astar_follow','common.planning.local_spline_follow',
            'common.experiments.astar_scan_compare'):return True
    # ros2 run 包装的仿真/键盘/SCAN 进程，以及直接 ros2 launch 本目录 launch 文件的进程。
    return any(value.endswith('/ros2') for value in args) and (
        'unitree_gazebo' in args or 'keyboard_input' in args or 'scan_planner' in args or
        any(value.startswith(str(ROOT/'Entrance/launch')+'/') and value.endswith('.launch.py')
            for value in args))


def cleanup_previous(domain):
    # 扫描 /proc 建立 PID → 进程信息表；只读取信息，不在这里发信号。
    processes={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=process_info(int(path.name))
            if row and row['pid']!=os.getpid():processes[row['pid']]=row
    owned={pid:row for pid,row in processes.items() if project_process(row,domain)}
    # 迭代补全子进程：父进程属于本会话时，其子进程也视为本次要清理的对象。
    changed=True
    while changed:
        changed=False
        for pid,row in processes.items():
            if pid not in owned and row['ppid'] in owned:
                owned[pid]=row;changed=True
    # Signal outer launchers/stages once. Their normal cleanup drains workers.
    # 只给最外层进程发 SIGINT，让原有清理链自行回收子进程；键盘子进程单独补发。
    for row in owned.values():
        if row['ppid'] not in owned:send(row,signal.SIGINT)
        elif row['args'] and Path(row['args'][0]).name=='keyboard_input':
            # ros2 run may exit without forwarding a PID-only signal to its
            # interactive child. This is still a verified project process.
            send(row,signal.SIGINT)
    # 先等待 45 s 正常退出，超时后升级为 SIGTERM；仍有残留则拒绝启动新会话。
    remaining=wait_exit(list(owned.values()),45)
    for row in remaining:send(row,signal.SIGTERM)
    remaining=wait_exit(remaining,5)
    if remaining:raise RuntimeError('Old project processes still active: '+str([r['pid'] for r in remaining]))
    return [dict(pid=row['pid'],args=row['args']) for row in owned.values()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    scenes=scene_names()
    # 场景与规划：场景来自 config/scenes；模式 4 为 cost 引导，模式 1 为人工目标。
    parser.add_argument('--scene',choices=scenes,default=default_scene(),
                        help='场景 YAML 名称；terrain-prepared 复用已有 world 和出生点')
    parser.add_argument('--model',choices=['tiny','small'],default='tiny')
    parser.add_argument('--flight-type',type=int,choices=[1,4],default=4,
                        help='4: cost-guided local planning. 1: direct manual SCAN.')
    parser.add_argument('--cost-limit',type=float,default=.8,help='Mode-4 strict A* cost upper bound')
    parser.add_argument('--planner-backend',choices=['light','light_scan','astar'],default='light_scan')
    parser.add_argument('--cost-config',type=Path,default=ROOT/'Entrance/config/local_cost.yaml')
    # 启动节奏：run_delay/lio_delay 为墙钟时间，其余 delay 在仿真时钟上计时。
    parser.add_argument('--run-delay',type=float,default=3.,help='Wall seconds after run readiness')
    parser.add_argument('--crouch-delay',type=float,default=3.,help='Simulation seconds in crouch')
    parser.add_argument('--stand-delay',type=float,default=4.,help='Simulation seconds standing')
    parser.add_argument('--mode-delay',type=float,default=3.,help='Simulation seconds in RL before LIO')
    parser.add_argument('--lio-delay',type=float,default=3.,help='Wall seconds after first registered cloud')
    parser.add_argument('--timeout',type=float,default=180.,help='Wall timeout for each readiness check')
    parser.add_argument('--dry-run',action='store_true',help='Print commands without cleanup or startup')
    args=parser.parse_args()
    if args.scene not in scenes:
        parser.error('PURE3D_SIM_SCENE must be one of: '+', '.join(scenes))
    # 场景选择通过环境变量传给 Gazebo 子进程；同时保留调用者已有的场景资源覆盖变量。
    scene_env={'PURE3D_SIM_SCENE':args.scene}
    for key in ('PURE3D_BUILDING_MODEL','PURE3D_TERRAIN_SCENE','PURE3D_TERRAIN_OUTPUT','PURE3D_TERRAIN_FLOOR'):
        if key in os.environ:
            scene_env[key]=os.environ[key]
    argument_values=dict(vars(args),cost_config=str(args.cost_config.resolve()))
    if not math.isfinite(args.cost_limit) or not 0<args.cost_limit<=1:
        parser.error('--cost-limit must lie in (0, 1]')
    if min(args.run_delay,args.crouch_delay,args.stand_delay,args.mode_delay,args.lio_delay)<0 or args.timeout<=0:
        parser.error('Delays must be nonnegative and timeout positive')
    # 整体方案只组合各单项 launch；场景参数全部从同一 YAML 读取。
    # four stages：Gazebo → 键盘（只切模式）→ LIO → cost；各自在独立终端中托管。
    # 仿真 launch 统一放在 Entrance/launch/sim，公共运行库在 Entrance/launch/_internal。
    launch_prefix=['/opt/ros/humble/bin/ros2','launch']
    commands=dict(gazebo=[*launch_prefix,str(ROOT/'Entrance/launch/sim/gazebo.launch.py'),
        'scene:='+args.scene,'planner:=scan',f'flight_type:={args.flight_type}','rviz:=false','gui:=true'],
        keyboard=[*launch_prefix,str(ROOT/'Entrance/launch/sim/keyboard.launch.py'),'publish_cmd_vel:=false'],
        lio=[*launch_prefix,str(ROOT/'Entrance/launch/sim/faster_lio.launch.py'),'args:=-p pcd_save.pcd_save_en:=false'],
        cost=[*launch_prefix,str(ROOT/'Entrance/launch/sim/cost.launch.py'),'model:='+args.model,
              'config:='+str(args.cost_config.resolve()),'rviz:=true'])
    # 模式 4 需要把代价上限与后端传给 Gazebo 内的 SCAN 组合。
    if args.flight_type==4:commands['gazebo'].extend(['cost_astar_limit:='+str(args.cost_limit),
                                                   'cost_planner_backend:='+args.planner_backend])
    try:
        # 启动前先解析场景资源，提前发现缺失的 world/模型/导出目录，而不是等到 Gazebo 报错。
        plan=scene_plan(args.scene,dict(planner='scan',flight_type=str(args.flight_type),
                        cost_astar_limit=str(args.cost_limit),cost_planner_backend=args.planner_backend,
                        gui='true',rviz='false'))
    except (OSError,ValueError,KeyError) as exc:
        parser.error(str(exc))
    if plan.get('terrain_dir'):
        # terrain 场景的导出目录要显式传给 Gazebo 子进程，保证它读取同一份 world。
        scene_env['PURE3D_TERRAIN_OUTPUT']=plan['terrain_dir']
    if args.dry_run:
        # 预览只打印将要执行的命令、环境与场景解析结果，不清理旧进程、不开终端。
        print(json.dumps(dict(commands=commands,gazebo_environment=scene_env,
                              scene_plan=plan,delays=argument_values,mode_sequence=[2,2,7]),
                              ensure_ascii=False,indent=2));return 0
    from common.mapping.local_map import load_local_config
    # 运行前同时确认 cost 权重、推理 Python、各 launch 文件存在，避免开到一半才发现缺文件。
    cfg=load_local_config(args.cost_config,args.model)
    for path in [Path(cfg['checkpoint']),Path(cfg['network_python']),
                 *(Path(v[0]) for v in commands.values()),*(Path(v[2]) for v in commands.values())]:
        if not path.is_file():parser.error('Missing: '+str(path))
    if not shutil.which('gnome-terminal'):parser.error('gnome-terminal is required')
    # 同一用户、同一 ROS 域只允许一个整体启动；文件锁在进程退出时自动释放。
    domain=int(os.environ.get('ROS_DOMAIN_ID','0'))
    lock=open(f'/tmp/pure3d_total_terminal_{os.getuid()}_domain_{domain}.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('Another total startup is already in progress')
    # SIGTERM/SIGINT 统一转成 KeyboardInterrupt，由下方清理分支关闭本次已开的终端。
    def interrupt(_signum,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    run=None;node=None;stages={};ready=False
    try:
        print('Cleaning old simulation/cost processes in this workspace and ROS domain...',flush=True)
        cleaned=cleanup_previous(domain)
        # 每次启动使用带时间戳的独立输出目录，保留 stages 状态、日志与配置快照。
        run=ROOT/'outputs/entry_local'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('terminal_%Y%m%d_%H%M%S_%f')
        (run/'stages').mkdir(parents=True)
        # 先占一个空闲端口作为本次 Gazebo master URI，避免与其它会话共用默认 11345。
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        master=f'http://127.0.0.1:{port}'
        # 模式 4 的 A* 输出也落在本次运行目录下，便于事后对照。
        if args.flight_type==4:commands['gazebo'].append('cost_astar_output:='+str(run/'astar'))
        # manifest 记录本次实验的完整上下文：场景、模式、后端、代价配置、清理结果与各进程 PID。
        manifest=dict(domain=domain,gazebo_master=master,flight_type=args.flight_type,model=args.model,
            scene_plan=plan,
            scene=args.scene,gazebo_environment=scene_env,
            manual_targets=True,scan_cost_integration=False,cost_astar_reference=args.flight_type==4,
            cost_limit=args.cost_limit,mode4_goal_fallback=False,
            trajectory_backend='support_spline' if args.flight_type==4 and args.planner_backend=='light' else 'scan',
            trajectory_server='ego_local_trajectory_server' if args.flight_type==4 and args.planner_backend=='light' else 'closed_loop_controller',
            planner_backend=args.planner_backend,
            cost_config=str(args.cost_config.resolve()),step_completion=cfg.get('step_completion',{}),
            pcd_save_en=False,cost_run=str(run/'cost'),
            cleaned=cleaned,processes={})
        import yaml
        # 从扫描器与安装外参中提取盒子尺寸、规划时域和雷达到机身的安装位姿，写入 manifest 便于复现。
        planner=yaml.safe_load((ROOT/'Entrance/config/scan.yaml').read_text())['scan_planner_node']['ros__parameters']
        frames=yaml.safe_load((ROOT/'Entrance/config/sim/frames.yaml').read_text())
        manifest.update(box_m=[planner['grid_map.sliding_map_size_'+a] for a in 'xyz'],
                        entry_box_source='lio_odometry',entry_box_m=cfg['box_size_m'],
                        planning_horizon_m=planner['fsm.planning_horizon'],
                        mount_base_to_lidar_xyz=frames['mount']['body_to_lidar']['translation'],
                        mount_rpy=frames['mount']['body_to_lidar']['rpy'])
        atomic_json(run/'processes.json',manifest)
        # current_run.txt 指向最近一次运行目录；同时保存本次使用的 SCAN 与安装外参快照。
        (ROOT/'outputs/entry_local/current_run.txt').write_text(str(run)+'\n')
        for src in ['a1.yaml','frames.yaml']:
            source=ROOT/('Entrance/config/scan.yaml' if src=='a1.yaml'
                         else 'Entrance/config/sim/frames.yaml')
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
        # 启动监控节点使用仿真时钟：所有 delay 与超时判断都和 Gazebo /clock 对齐。
        node=rclpy.create_node('pure3d_terminal_startup',parameter_overrides=[Parameter('use_sim_time',value=True)])
        observed=dict(odom=None,cloud=False,follow=None);buffer=Buffer();listener=TransformListener(buffer,node)
        # follow 状态使用 transient_local，晚加入的监控节点也能拿到最新一次发布。
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String,'/entry/local_astar_follow/status',
            lambda m:observed.update(follow=json.loads(m.data)),durable)
        node.create_subscription(Odometry,'/Odometry_gazebo',lambda m:observed.update(odom=m),qos_profile_sensor_data)
        node.create_subscription(PointCloud2,'/faster_lio/cloud_registered',
            lambda m:observed.update(cloud=bool(m.width*m.height)),qos_profile_sensor_data)
        mode_pub=node.create_publisher(Inputs,'/control_input',10)

        # 读取某个阶段终端写入的状态文件（terminal_stage.py 负责更新）。
        def stage_record(name):
            path=run/'stages'/(name+'.json')
            if path.exists():return json.loads(path.read_text())
            return None

        # 任一会话终端退出即视为启动失败，抛出异常并进入统一清理分支。
        def check_stages():
            for name in stages:
                row=stage_record(name)
                if row and (row['state']!='running' or not alive(row['process'])):
                    raise RuntimeError(f'{name} exited. See {run/(name+".log")}')

        # 每次等待前推进一次 ROS 回调并检查阶段进程，保证等待过程可被中断且不会“假等待”。
        def tick():
            check_stages();rclpy.spin_once(node,timeout_sec=.05)

        # 在墙钟超时内轮询条件；超时后报出当前等待的标签，便于定位卡在哪一步。
        def wait_until(label,test,timeout=None):
            print(label,flush=True);end=time.monotonic()+(args.timeout if timeout is None else timeout)
            while time.monotonic()<end:
                tick()
                if test():return
            raise RuntimeError('Timeout: '+label)

        # sim=True 时按 /clock 计时（用于蹲下/站立/RL 等待），否则按墙钟计时。
        def delay(seconds,sim=False):
            start=node.get_clock().now().nanoseconds*1e-9 if sim else time.monotonic()
            wait_until(f'Waiting {seconds:g} '+('simulation seconds...' if sim else 'seconds...'),
                lambda:(node.get_clock().now().nanoseconds*1e-9 if sim else time.monotonic())-start>=seconds,
                max(args.timeout,seconds*5+10))

        # 在独立 gnome-terminal 中启动一个阶段：环境变量显式传入 ROS 域、Gazebo master、
        # 工作区和输出目录；子进程由 terminal_stage.py 托管并记录 PID/日志。
        def launch(name,command):
            print('Opening terminal: '+name,flush=True)
            overrides=[f'{key}={value}' for key,value in scene_env.items()] if name=='gazebo' else []
            environment=['env',f'ROS_DOMAIN_ID={domain}','ROS_LOCALHOST_ONLY='+os.environ.get('ROS_LOCALHOST_ONLY','1'),
                'GAZEBO_MASTER_URI='+master,'PURE3D_WS='+str(ROOT),'PYTHONPATH='+os.environ.get('PYTHONPATH',str(ROOT)),
                'PURE3D_LIO_OUTPUT='+str(run/'lio'),*overrides,'/usr/bin/python3',str(Path(__file__).with_name('terminal_stage.py')),
                '--run',str(run),'--name',name,'--',*command]
            client_env=dict(os.environ)
            # 启动 gnome-terminal 的客户端环境去掉 ROS overlay 路径，避免新终端继承父进程后
            # 再被 terminal_stage.py 中的 sim_env.bash 重复叠加；真正的环境由上面的 env 参数提供。
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

        # 第一段：启动 Gazebo + SCAN；随后等待四个控制器全部 active 且里程计已发布。
        launch('gazebo',commands['gazebo'])
        controllers=node.create_client(ListControllers,'/controller_manager/list_controllers')
        controller_order=['joint_state_broadcaster','imu_sensor_broadcaster','leg_pd_controller','unitree_guide_controller']
        required=set(controller_order)
        future=None;inactive_since=None;recovered=False
        # 周期查询控制器状态；若四个控制器已加载但长时间未激活，则主动恢复物理并再次切换。
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
        # 时钟与速度控制源必须唯一，否则遥控/规划会互相打架。
        if node.count_publishers('/clock')!=1 or node.count_publishers('/cmd_vel')!=1:
            raise RuntimeError('Expected exactly one clock and one SCAN velocity source')
        delay(args.run_delay)
        # 第二段：打开键盘终端（publish_cmd_vel=false），仅用于后续模式切换与人工 P 键急停。
        launch('keyboard',commands['keyboard'])
        wait_until('Waiting for keyboard and mode subscriber...',lambda:node.count_publishers('/control_input')>=2 and mode_pub.get_subscription_count()>0)
        # 模式切换以 Gazebo 日志中的状态机输出为确认，而不是仅看命令是否发出。
        def log_has(text):return (run/'gazebo.log').exists() and text in (run/'gazebo.log').read_text(errors='replace')
        def mode(command,expected):
            print(f'Robot mode command: {command}',flush=True)
            message=Inputs();message.command=command;mode_pub.publish(message)
            # 发布一次后立即清空命令，等同键盘“按下再松开”，避免状态机重复触发。
            delay(.1);mode_pub.publish(Inputs())
            wait_until('Waiting for '+expected+'...',lambda:log_has(expected),15)
        # 原键盘流程：2 蹲下 → 2 站起 → 7 进入 RL；等待时间按仿真时钟计算。
        mode(2,'Switched from passive to fixed down');delay(args.crouch_delay,True)
        mode(2,'Switched from fixed down to fixed stand');delay(args.stand_delay,True)
        mode(7,'Switched from fixed stand to RL');delay(args.mode_delay,True)
        # 第三段：LIO；等到注册点云与 world→camera_init TF 同时可用后再进入 cost。
        launch('lio',commands['lio'])
        wait_until('Waiting for registered LIO cloud and world TF...',lambda:observed['cloud'] and buffer.can_transform('world','camera_init',Time()))
        delay(args.lio_delay)
        # 第四段：cost 会话；输出目录固定到本次运行下，便于后续检查。
        launch('cost',commands['cost']+['output:='+str(run/'cost')])
        def cost_ready():
            # cost 会话在 status.json 中写入 valid/error；出现 error 立即中止启动。
            path=run/'cost/local/status.json'
            if not path.exists():return False
            value=json.loads(path.read_text())
            if value.get('error'):raise RuntimeError(value['error'])
            return value.get('valid',False)
        wait_until('Waiting for first valid cost...',cost_ready)
        if args.flight_type==4:
            # 模式 4 还要求跟踪状态就绪、cost 新鲜且 /pure_path 恰好一个发布者。
            wait_until('Waiting for mode 4 planner, occupancy and cost...',lambda:
                observed['follow'] is not None and observed['follow'].get('scan_mode')==4 and
                observed['follow'].get('scan_mode_ready') and observed['follow'].get('cost_fresh') and
                node.count_publishers('/pure_path')==1)
        if node.count_publishers('/entry/local/cost_cloud')!=1:
            raise RuntimeError('Expected exactly one local cost publisher')
        # 把 cost 会话的子进程合并进总 manifest，并写出 ready 状态与关键话题发布者数量。
        local_manifest=json.loads((run/'cost/processes.json').read_text())
        manifest['processes'].update(local_manifest['processes'])
        atomic_json(run/'processes.json',manifest)
        atomic_json(run/'startup_status.json',dict(state='ready',domain=domain,mode_sequence=[2,2,7],flight_type=args.flight_type,
            cost_run=str(run/'cost'),delays=argument_values,
            publishers={t:node.count_publishers(t) for t in ['/clock','/cmd_vel','/entry/local/cost_cloud','/pure_path']}))
        ready=True
        # 四个终端相互独立；启动脚本退出后它们仍继续运行，由 keyboard/cost 终端负责后续交互。
        print(f'READY: {run}\nFour terminals are independent. Publish end_pos in RViz.\n'
              'P is available in the keyboard terminal. This startup shell can now exit.',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        print('Startup interrupted or failed: '+str(exc),flush=True)
        # Close only the terminals/processes created by this incomplete startup.
        # 只关闭本次已打开的终端，避免误杀之前已经就绪的会话；按打开顺序逆序发送 SIGINT。
        launched=[]
        for name in reversed(list(stages)):
            path=run/'stages'/(name+'.json')
            if path.exists():
                row=json.loads(path.read_text());send(row['stage'],signal.SIGINT);launched.append(row['stage'])
        remaining=wait_exit(launched,45)
        # 宽限期后仍存活的终端进程升级为 SIGTERM；状态文件记录失败原因。
        for row in remaining:send(row,signal.SIGTERM)
        wait_exit(remaining,5)
        if run:atomic_json(run/'startup_status.json',dict(state='failed',error=str(exc)))
        return 1
    finally:
        # 无论成功/失败都销毁监控节点并释放启动锁。
        if node is not None:
            node.destroy_node()
            import rclpy
            if rclpy.ok():rclpy.shutdown()
        lock.close()
    return 0 if ready else 1


if __name__=='__main__':raise SystemExit(main())
