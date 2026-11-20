# A1 平地控制器单项示例，导航整体入口见 Entrance。
# Usage（先加载对应 ROS 工作区）：ros2 launch unitree_guide_controller eolp_classic.launch.py
# 配置位置：Simulator/quadruped_ros2_control/descriptions/unitree/a1_description/config/gazebo.yaml。
# 用途：A1 平地 Gazebo Classic 控制入口，保留已迁移的四足 FSM 与 RL 策略。
# 与 unitree_gazebo 主入口分开：默认 flat.world，不在此启动导航规划器、里程计转换节点或 RViz。
# 启动链：机器人生成 → 未激活的状态/PD 控制器 → 引导控制器 → 恢复物理 → 激活控制器。
"""A1 Gazebo Classic + chained ros2_control + the migrated RL state."""
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler, EmitEvent, LogInfo, ExecuteProcess, GroupAction, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


# OpaqueFunction 解析命令行参数后，生成此次启动使用的模型与控制配置。
def setup(context):
    args = context.launch_configurations
    description = get_package_share_directory('a1_description')
    controller = get_package_share_directory('unitree_guide_controller')
    # 【原四足配置】沿用 A1 的关节与控制参数，只覆盖 RL 权重、推理设备和 trace 开关。
    config = yaml.safe_load(Path(description, 'config/gazebo.yaml').read_text())
    params = config['unitree_guide_controller']['ros__parameters']
    params['rl.model_path'] = args['model_path']
    params['rl.device'] = args['device']
    params['rl.publish_trace'] = args['trace'].lower() == 'true'
    # 写入临时 YAML 并交给 gazebo_ros2_control 插件，原始配置文件保持不变。
    config_file = Path(tempfile.mkdtemp(prefix='eolp_native_'), 'controllers.yaml')
    config_file.write_text(yaml.safe_dump(config))
    # Classic 模式加载控制插件；MID360 开关、每帧采样数与取样步长传给 xacro。
    xml = xacro.process_file(os.path.join(description, 'xacro/robot.xacro'),
                             mappings={'GAZEBO': 'true', 'CLASSIC': 'true', 'MID360': args['mid360'], 'MID360_SAMPLES': args['mid360_samples'], 'MID360_DOWNSAMPLE': args['mid360_downsample']}).toxml()
    # 解析/重新序列化会移除 XML 注释，避免控制插件把注释误当作 ROS 参数内容。
    root = ET.fromstring(xml)  # Drop XML comments before gazebo_ros2_control's ROS parameter parsing.
    root.find("./gazebo/plugin[@name='gazebo_ros2_control']/parameters").text = str(config_file)
    # 【四足初始姿态】大腿 1.3 rad、小腿 -2.4 rad、髋关节 0 rad，保证初值处于关节限位内。
    # Start within the legal joint limits instead of the Classic plugin's default calf=0.
    for joint in root.findall('./ros2_control/joint'):
        position = joint.find("state_interface[@name='position']")
        initial = ET.SubElement(position, 'param', {'name': 'initial_value'})
        initial.text = '1.3' if 'thigh' in joint.attrib['name'] else '-2.4' if 'calf' in joint.attrib['name'] else '0.0'
    # 【四足真值观测】追加 p3d 插件，以 100 Hz 发布 /ground_truth/base_w 的世界坐标状态。
    gazebo = ET.SubElement(root, 'gazebo')
    plugin = ET.SubElement(gazebo, 'plugin', {'name': 'eolp_ground_truth', 'filename': 'libgazebo_ros_p3d.so'})
    ros = ET.SubElement(plugin, 'ros')
    ET.SubElement(ros, 'namespace').text = '/ground_truth'
    ET.SubElement(ros, 'remapping').text = 'odom:=base_w'
    for name, value in [('body_name', 'base'), ('frame_name', 'world'), ('update_rate', '100.0'), ('gaussian_noise', '0.0')]:
        ET.SubElement(plugin, name).text = value
    xml = ET.tostring(root, encoding='unicode')
    # 发布机器人描述和 TF，并使用 Gazebo 仿真时钟。
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'use_sim_time': True, 'robot_description': xml}])
    # 默认包含 gazebo_ros 的 Classic 启动文件，先暂停物理以完成模型/控制器初始化。
    sim = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('gazebo_ros'), 'launch/gazebo.launch.py')),
        launch_arguments={'world': args['world'], 'gui': args['gui'], 'pause': 'true', 'server_required': 'true'}.items())
    # debug=true 改用 gdb 直接启动服务器并在异常时打印调用栈，此分支不包含图形客户端。
    if args['debug'].lower() == 'true':
        sim = ExecuteProcess(cmd=['gdb', '--batch', '-ex', 'run', '-ex', 'thread apply all bt 15', '--args',
            'gzserver', args['world'], '--pause', '-slibgazebo_ros_init.so', '-slibgazebo_ros_factory.so',
            '-slibgazebo_ros_force_system.so'], output='screen')
    # CPU 策略预加载 Torch 的库，避免 Gazebo 与 Torch 的 BLAS 符号冲突，环境仅作用于仿真组。
    # Load Torch before Gazebo's BLAS to keep its CPU kernels bound to their own BLAS symbols.
    # This environment is scoped to the simulator, not the ROS publisher/spawner processes.
    if args['device'] == 'cpu':
        torch_root = Path(controller, 'libtorch_path.txt').read_text().strip()
        preload = str(Path(torch_root, 'lib/libtorch_cpu.so'))
        if os.environ.get('LD_PRELOAD'):
            preload += ':' + os.environ['LD_PRELOAD']
        sim = GroupAction([SetEnvironmentVariable('LD_PRELOAD', preload), sim])
    # 生成 a1 实体；x/y/height 单位 m，yaw 单位 rad，均相对于世界坐标。
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                 arguments=['-topic', 'robot_description', '-entity', 'a1', '-x', args['x'], '-y', args['y'], '-Y', args['yaw'], '-z', args['height']], output='screen')
    # 控制器先以 inactive 加载，等待 controller_manager 最多 60 s。
    def spawner(names):
        return Node(package='controller_manager', executable='spawner',
            arguments=names + ['--inactive', '-c', '/controller_manager', '--controller-manager-timeout', '60'], output='screen')
    # 状态广播器与关节 PD 先加载，再加载包含四足 FSM/RL 状态的引导控制器。
    sensors_pd = spawner(['joint_state_broadcaster', 'imu_sensor_broadcaster', 'leg_pd_controller'])
    guide = spawner(['unitree_guide_controller'])
    # 加载完成后恢复物理，并以 strict 方式一起激活所有控制器。
    unpause = ExecuteProcess(cmd=['ros2', 'service', 'call', '/unpause_physics', 'std_srvs/srv/Empty', '{}'], output='screen')
    activate = ExecuteProcess(
        cmd=['ros2', 'control', 'switch_controllers', '-c', '/controller_manager', '--strict', '--activate',
                   'joint_state_broadcaster', 'imu_sensor_broadcaster', 'leg_pd_controller',
                   'unitree_guide_controller'], output='screen')
    # 上一步正常退出才启动下一步；非零退出码触发整个 launch 停止。
    def after(process, next_actions):
        def completed(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason=f'Controller startup step failed ({event.returncode})'))]
            return next_actions
        return RegisterEventHandler(OnProcessExit(target_action=process, on_exit=completed))
    # 先注册启动链再执行 spawn；激活后提示键盘操作，但本文件不自动切换 RL 模式。
    return [rsp, sim, after(spawn, [sensors_pd]), after(sensors_pd, [guide]), after(guide, [unpause]),
            after(unpause, [activate]), after(activate, [LogInfo(msg='EOLP A1 controllers active. Open another terminal, source this workspace and run: ros2 run keyboard_input keyboard_input --ros-args -p publish_cmd_vel:=true. Type keys in that terminal: 2 crouch/stand, 7 RL, 1 passive.')]), spawn]


def generate_launch_description():
    pkg = get_package_share_directory('unitree_guide_controller')
    return LaunchDescription([
        # 调试与传感器：debug 启用 gdb；MID360 samples 为每帧采样数，downsample 为取样步长。
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('mid360', default_value='true'),
        DeclareLaunchArgument('mid360_samples', default_value='24000'),
        DeclareLaunchArgument('mid360_downsample', default_value='1'),
        # 窗口开关与机器人出生位姿：height/x/y 单位 m，yaw 单位 rad。
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('height', default_value='0.22'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        # 【四足 RL 参数】默认 CPU；trace 发布策略调试数据；model_path 指定已有楼梯策略。
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('trace', default_value='false'),
        DeclareLaunchArgument('model_path', default_value=os.path.join(pkg, 'policies/policy_act_inference_stair.pt')),
        # 默认平地测试世界，可用 world:=绝对路径 替换场景。
        DeclareLaunchArgument('world', default_value=os.path.join(pkg, 'worlds/flat.world')),
        # 所有参数声明完成后再生成启动动作。
        OpaqueFunction(function=setup),
    ])
