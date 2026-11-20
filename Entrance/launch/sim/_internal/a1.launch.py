# 用途：A1 仿真内部实现，场景与初始位姿由 Entrance/launch/sim/gazebo.launch.py 从 YAML 传入。
# Usage：ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building
# 场景配置：Entrance/config/scenes/*.yaml；公共默认：Entrance/config/startup.yaml。
# SCAN 配置：Entrance/config/scan.yaml。
# 话题配置：Entrance/config/local_planners.yaml。
# EGO/SCAN 规划与跟踪节点在本文件直接定义；planner 选择 ego/scan/None。
# 本入口按所选模式组合机器人与局部规划，建图与全局工具使用各自的单项 launch。
# 启动顺序：生成模型 → 加载机器人 → 加载未激活的控制器 → 恢复物理 → 同时激活控制器。
"""A1 simulation with exclusive EGO or SCAN local planner selection."""
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription, LogInfo,
    OpaqueFunction, RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from scripts import GazeboRosPaths


# OpaqueFunction 在参数解析后执行，才能用命令行参数生成控制器 YAML 和机器人 URDF。
def setup(context):
    args = context.launch_configurations
    # 模式 4 的 cost 参考依赖 SCAN，先做组合约束检查，避免启动到一半才失败。
    if args['flight_type'] == '4' and args['planner'] != 'scan':
        raise ValueError('flight_type:=4 requires planner:=scan')
    # 三个包目录分别提供 world/模型、机器人描述与控制器/策略；entrance 用于读取集中配置。
    scene = get_package_share_directory('unitree_gazebo')
    description = get_package_share_directory('a1_description')
    controller = get_package_share_directory('unitree_guide_controller')
    # 本文件位于 Entrance/launch/sim/_internal，上溯三级才是 Entrance。
    entrance = Path(__file__).resolve().parents[3]
    # 规划器共用的话题映射集中在 Entrance/config/local_planners.yaml。
    topics = yaml.safe_load((entrance / 'config/local_planners.yaml').read_text())['topics']

    # 【四足控制配置】从 A1 原配置读取关节、PD 和 RL 参数，只覆盖策略路径、设备和调试输出。
    # Preserve the existing policy, controller and joint initialization contract.
    config = yaml.safe_load(Path(description, 'config/gazebo.yaml').read_text())
    params = config['unitree_guide_controller']['ros__parameters']
    params['rl.model_path'] = args['model_path']
    params['rl.device'] = args['device']
    params['rl.publish_trace'] = args['trace'].lower() == 'true'
    # [ROS2适配 R3] 仅给Gazebo时钟节点配置发布频率；控制器参数沿用原值。
    config['gazebo'] = {'ros__parameters': {'publish_rate': float(args['clock_rate'])}}
    # 写入本次启动的临时配置，不改动安装目录中的原始 YAML；Gazebo 与模型插件共用它。
    config_file = Path(tempfile.mkdtemp(prefix='eolp_native_'), 'controllers.yaml')
    config_file.write_text(yaml.safe_dump(config))
    # 【四足传感器配置】Classic 模式加载 gazebo_ros2_control，MID360 的开关/采样数/步长传给 xacro。
    xml = xacro.process_file(os.path.join(description, 'xacro/robot.xacro'), mappings={
        'GAZEBO': 'true', 'CLASSIC': 'true', 'MID360': args['mid360'],
        'MID360_SAMPLES': args['mid360_samples'],
        'MID360_DOWNSAMPLE': args['mid360_downsample'],
    }).toxml()
    root = ET.fromstring(xml)
    # 将生成的配置交给模型内的控制插件，控制器管理器随机器人插件一起创建。
    root.find("./gazebo/plugin[@name='gazebo_ros2_control']/parameters").text = str(config_file)
    # 【四足初始姿态】大腿 1.3 rad、小腿 -2.4 rad、髋关节 0 rad，避免小腿初始值超出限位。
    for joint in root.findall('./ros2_control/joint'):
        position = joint.find("state_interface[@name='position']")
        initial = ET.SubElement(position, 'param', {'name': 'initial_value'})
        initial.text = '1.3' if 'thigh' in joint.attrib['name'] else '-2.4' if 'calf' in joint.attrib['name'] else '0.0'
    # 【四足观测】追加 100 Hz 无噪声真值插件，输出 /ground_truth/base_w（world 系下的 base 状态）。
    gazebo = ET.SubElement(root, 'gazebo')
    plugin = ET.SubElement(gazebo, 'plugin', {
        'name': 'eolp_ground_truth', 'filename': 'libgazebo_ros_p3d.so',
    })
    ros = ET.SubElement(plugin, 'ros')
    ET.SubElement(ros, 'namespace').text = '/ground_truth'
    ET.SubElement(ros, 'remapping').text = 'odom:=base_w'
    for name, value in [('body_name', 'base'), ('frame_name', 'world'),
                        ('update_rate', '100.0'), ('gaussian_noise', '0.0')]:
        ET.SubElement(plugin, name).text = value
    robot_description = ET.tostring(root, encoding='unicode')

    # 从各包导出的 Gazebo 路径中查找模型/插件/资源，并保留用户原有环境变量。
    # Resolve the same exported model/plugin paths used by gazebo_ros.
    model, plugin_path, media = GazeboRosPaths.get_paths()
    scene_models = args['scene_models_path']
    if scene_models and not Path(scene_models).is_dir():
        raise FileNotFoundError(f'Scene model directory does not exist: {scene_models}')
    # 场景模型目录优先于包内默认模型；插件与媒体资源仍使用包导出路径。
    gazebo_env = {}
    for key, paths in {
        'GAZEBO_MODEL_PATH': [scene_models, os.path.join(scene, 'models'), model],
        'GAZEBO_PLUGIN_PATH': [plugin_path],
        'GAZEBO_RESOURCE_PATH': [media],
    }.items():
        gazebo_env[key] = os.pathsep.join(path for path in [*paths, os.environ.get(key, '')] if path)
    server_env = dict(gazebo_env)
    # CPU 策略预加载 libtorch_cpu，避免 Gazebo 与 Torch 的 BLAS 符号绑定冲突；只作用于服务器。
    if args['device'] == 'cpu':
        torch_root = Path(controller, 'libtorch_path.txt').read_text().strip()
        preload = str(Path(torch_root, 'lib/libtorch_cpu.so'))
        if os.environ.get('LD_PRELOAD'):
            preload += ':' + os.environ['LD_PRELOAD']
        server_env['LD_PRELOAD'] = preload

    # 服务器先暂停物理；加载 ROS 初始化、实体生成与外力插件，并通过参数文件设置 /clock 频率。
    server_cmd = ['gzserver', args['world'], '--pause',
                  '-slibgazebo_ros_init.so', '-slibgazebo_ros_factory.so',
                  '-slibgazebo_ros_force_system.so',
                  '--ros-args', '--params-file', str(config_file), '--']
    # debug=true 时用 gdb 启动服务器，异常退出可打印各线程调用栈。
    if args['debug'].lower() == 'true':
        server_cmd = ['gdb', '--batch', '-ex', 'run', '-ex', 'thread apply all bt 15',
                      '--args', *server_cmd]
    server = ExecuteProcess(cmd=server_cmd, additional_env=server_env, output='screen')
    # gui 只控制 Gazebo 图形客户端；即使关闭 GUI，服务器与机器人仍会启动。
    client = ExecuteProcess(
        cmd=['gzclient', '--gui-client-plugin=libgazebo_ros_eol_gui.so'],
        additional_env=gazebo_env, output='screen',
        condition=IfCondition(LaunchConfiguration('gui')),
    )
    # 发布 robot_description 与关节 TF；所有仿真节点使用 Gazebo 的 /clock。
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'use_sim_time': True, 'robot_description': robot_description}])
    # 从 robot_description 创建名为 a1 的实体；x/y/height 是世界坐标，yaw 单位为 rad。
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-topic', 'robot_description', '-entity', 'a1',
                            '-x', args['x'], '-y', args['y'], '-Y', args['yaw'],
                            '-z', args['height']])

    # 先加载控制器但不激活，等待 controller_manager 最多 60 s，避免初始化期间抢先输出关节命令。
    def spawner(names):
        return Node(package='controller_manager', executable='spawner', output='screen',
                    arguments=[*names, '--inactive', '-c', '/controller_manager',
                               '--controller-manager-timeout', '60'])

    # 第一组是关节状态、IMU 状态和关节 PD；第二组是包含原四足 FSM/RL 状态的引导控制器。
    sensors_pd = spawner(['joint_state_broadcaster', 'imu_sensor_broadcaster', 'leg_pd_controller'])
    guide = spawner(['unitree_guide_controller'])
    # 全部控制器加载后恢复物理，再通过 strict 切换同时激活四个控制器。
    unpause = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/unpause_physics', 'std_srvs/srv/Empty', '{}'],
        output='screen',
    )
    activate = ExecuteProcess(
        cmd=['ros2', 'control', 'switch_controllers', '-c', '/controller_manager',
             '--strict', '--activate', 'joint_state_broadcaster',
             'imu_sensor_broadcaster', 'leg_pd_controller', 'unitree_guide_controller'],
        output='screen',
    )

    # 只有上一步成功退出才执行下一步；加载/激活失败时停止整个 launch。
    def after(process, next_actions):
        def completed(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(
                    reason=f'Controller startup step failed ({event.returncode})'))]
            return next_actions
        return RegisterEventHandler(OnProcessExit(target_action=process, on_exit=completed))

    # RViz 显示现有 EGO/PCT 配置；rviz 与 gui 是相互独立的开关。
    rviz = Node(
        package='rviz2', executable='rviz2', name='ego_rviz', output='screen',
        arguments=['-d', args['rviz_config'] or str(entrance / 'config' / ('scan.rviz' if args['planner'] == 'scan' else 'ego.rviz'))], parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )
    # Gazebo truth owns world -> base; the LIO bridge owns world -> map -> camera_init -> body.
    state = Node(
        package='unitree_gazebo', executable='state_from_gazebo', output='screen',
        parameters=[{'use_sim_time': True, 'entity_name': 'a1', 'truth_frame': 'world'}],
    )
    # Exactly one local planner owns the velocity command topic; None runs simulation only.
    local_nodes = []
    if args['planner'] == 'ego':
        # EGO 分支：使用 ego_planner 自带的 quadruped 参数，并把里程计/点云/路径话题
        # 重映射到 local_planners.yaml 中的统一话题。
        ego_config = str(Path(get_package_share_directory('ego_planner'), 'config/quadruped.yaml'))
        ego_params = {'use_sim_time': True, 'fsm/flight_type': int(args['flight_type']),
                      'fsm/odometry_topic': topics['odom'], 'odometry_topic': topics['odom']}
        local_nodes = [
            Node(package='ego_planner', executable='ego_planner_node', name='ego_planner_node',
                 output='screen', parameters=[ego_config, ego_params],
                 remappings=[('grid_map/odom', topics['odom']), ('grid_map/cloud', topics['cloud']),
                             ('/pct_path', topics['path'])]),
            Node(package='ego_planner', executable='traj_server', name='traj_server',
                 output='screen', parameters=[ego_config, ego_params],
                 remappings=[('/pct_path', topics['path']), ('/cmd_vel', topics['cmd_vel'])]),
        ]
    elif args['planner'] == 'scan':
        # SCAN 分支：三个节点分别做工作区桥接、局部规划与轨迹跟踪；所有内部话题收拢到 /scan。
        use_sim_time, mode = True, int(args['flight_type'])
        scan_share = get_package_share_directory('scan_planner')
        scan_config = str(entrance / 'config/scan.yaml')
        # 先列出需要在多个节点间保持一致的话题重映射，避免节点间话题名不一致。
        internal = [('body_pose', '/scan/body_pose'), ('sensor_pose', '/scan/sensor_pose'),
                    ('cloud', '/scan/cloud'), ('move_base_simple/goal', '/scan/goal'),
                    ('initial_path', '/scan/initial_path'), ('planning/bspline', '/scan/planning/bspline'),
                    ('planning/data_display', '/scan/planning/data_display'),
                    ('planning/go2_execution_frozen', '/scan/planning/go2_execution_frozen')]
        for name in ('goal_point', 'global_list', 'init_list', 'optimal_list', 'a_star_list', 'self_inflation',
                     'grid_map/occupancy', 'grid_map/occupancy_inflate', 'grid_map/sliding_map_bbox',
                     'grid_map/unknown', 'grid_map/depth_cloud', 'grid_map/sensor_pose_extrinsic'):
            internal.append((name, '/scan/' + name))
        scan_nodes = [
            Node(package='scan_planner', executable='scan_workspace_bridge', name='scan_workspace_bridge',
                 output='screen', parameters=[scan_config, {'use_sim_time': use_sim_time, 'manual_markers': mode == 1}],
                 remappings=[('input_odom', topics['odom']), ('input_cloud', topics['cloud']),
                             ('input_path', topics['path']),
                             ('body_pose', '/scan/body_pose'), ('sensor_pose', '/scan/sensor_pose'),
                             ('cloud', '/scan/cloud'), ('goal', '/scan/goal'), ('initial_path', '/scan/initial_path')]),
            Node(package='scan_planner', executable='scan_planner_node', name='scan_planner_node',
                 output='screen', parameters=[scan_config, {'use_sim_time': use_sim_time, 'fsm.navi_mode': mode}],
                 remappings=internal),
            Node(package='scan_planner', executable='closed_loop_controller', name='closed_loop_controller',
                 output='screen', parameters=[scan_config, {'use_sim_time': use_sim_time}],
                 remappings=internal + [('cmd_vel', topics['cmd_vel'])]),
        ]
        if mode == 4:
            # 模式 4 把完整 SCAN 组合交给单项 launch，由它追加 cost 参考与 A* 跟踪节点；
            # 此时本文件不再重复创建 scan_nodes，避免同一话题出现两个发布者。
            local_nodes = [IncludeLaunchDescription(PythonLaunchDescriptionSource(
                str(entrance / 'launch/sim/scan.launch.py')), launch_arguments={
                    'flight_type': '4', 'cost_astar_output': args['cost_astar_output'],
                    'cost_astar_limit': args['cost_astar_limit'],
                    'cost_planner_backend': args['cost_planner_backend']}.items())]
        else:
            local_nodes = scan_nodes
    # 先注册退出事件再启动 spawn，确保串行启动链能接收退出通知；服务器退出则关闭整个 launch。
    return [
        LogInfo(msg=f'Gazebo world: {args["world"]}; scene models: {scene_models or "package defaults"}'),
        rsp,
        RegisterEventHandler(OnProcessExit(
            target_action=server,
            on_exit=[EmitEvent(event=Shutdown(reason='Gazebo server exited'))])),
        server, client, rviz, state, *local_nodes,
        after(spawn, [sensors_pd]), after(sensors_pd, [guide]),
        after(guide, [unpause]), after(unpause, [activate]),
        after(activate, [LogInfo(msg=(
            'A1 controllers active. Open another terminal: ros2 launch Entrance/launch/sim/keyboard.launch.py. '
            'Keys: 2 crouch, wait 3 s, 2 stand, wait 3 s, 7 RL, 1 passive.'))]),
        spawn,
    ]


# 公开参数均可通过 ros2 launch ... 参数名:=值 覆盖；下面按显示、传感器、控制与坐标分组。
def generate_launch_description():
    guide = get_package_share_directory('unitree_guide_controller')
    return LaunchDescription([
        # 显示与调试：Gazebo 窗口、RViz 窗口/配置、服务器 gdb 开关。
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo client'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz2 for the selected local planner'),
        DeclareLaunchArgument('rviz_config', default_value='', description='Empty selects the RViz config for the chosen planner'),
        DeclareLaunchArgument('debug', default_value='false'),
        # MID360：samples 为每帧扫描模式采样数，downsample 为取样步长；步长增大可降低仿真负载。
        DeclareLaunchArgument('mid360', default_value='true'),
        DeclareLaunchArgument('mid360_samples', default_value='10000'),
        DeclareLaunchArgument('mid360_downsample', default_value='1'),
        # RL 推理设备 cpu/cuda；trace 控制策略调试数据发布，不改变策略模型。
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('trace', default_value='false'),
        # [ROS2适配 R3] 规划/跟踪使用仿真时钟，避免默认10Hz /clock使时间采样过粗。
        DeclareLaunchArgument('clock_rate', default_value='100.0'),
        # 世界场景由 YAML 解析后传入，四足 RL 权重保留现有楼梯策略。
        DeclareLaunchArgument('world', description='由场景 YAML 解析后的 world 完整路径'),
        DeclareLaunchArgument('scene_models_path', default_value='', description='Scene models directory, searched before package models'),
        DeclareLaunchArgument('model_path', default_value=os.path.join(guide, 'policies', 'policy_act_inference_stair.pt')),
        # 机器人世界出生位姿：x/y/height 单位 m，yaw 单位 rad。
        DeclareLaunchArgument('x', description='由场景 YAML 传入的出生 X 坐标，米'),
        DeclareLaunchArgument('y', description='由场景 YAML 传入的出生 Y 坐标，米'),
        DeclareLaunchArgument('height', description='由场景 YAML 传入的机身初始高度，米'),
        DeclareLaunchArgument('yaw', description='由场景 YAML 传入的初始航向，弧度'),
        # Legacy CLI offsets are accepted; truth is now published directly in world.
        DeclareLaunchArgument('odom_x', default_value='0.0'),
        DeclareLaunchArgument('odom_y', default_value='0.0'),
        DeclareLaunchArgument('odom_z', default_value='0.0'),
        # Planner selection; local topics/map/horizon live in YAML, not launch arguments.
        # 规划与模式参数：planner 选择唯一的速度控制源；flight_type 决定 SCAN/EGO 的规划行为。
        # cost_astar_* 仅在 flight_type:=4 时透传给 scan.launch.py。
        DeclareLaunchArgument('planner', default_value='scan', choices=['ego', 'scan', 'None']),
        DeclareLaunchArgument('flight_type', default_value='1', choices=['1', '2', '3', '4']),
        DeclareLaunchArgument('cost_astar_output', default_value=''),
        DeclareLaunchArgument('cost_astar_limit', default_value='0.8'),
        DeclareLaunchArgument('cost_planner_backend', default_value='light_scan',choices=['light','light_scan','astar']),
        # 所有参数声明完成后，使用其最终值构造启动动作。
        OpaqueFunction(function=setup),
    ])
