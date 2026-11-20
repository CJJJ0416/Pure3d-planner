# 保留的通用 Gazebo Classic 控制器示例。
# Usage（先加载对应 ROS 工作区）：ros2 launch unitree_guide_controller gazebo_classic.launch.py pkg_description:=a1_description
# 配置位置：所选描述包的 config/robot_control.yaml。
# 用途：控制器包保留的通用 Gazebo Classic 示例，默认模型包为 go1_description。
# 当前 A1 Building 完整入口位于 Entrance/launch/sim/gazebo.launch.py。
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

# 在参数解析后选择描述包与出生高度，从包中加载 xacro 和 RViz 配置。
def launch_setup(context, *args, **kwargs):

    package_description = context.launch_configurations['pkg_description']
    init_height = context.launch_configurations['height']
    pkg_path = os.path.join(get_package_share_directory(package_description))

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    # GAZEBO/CLASSIC 开关使 xacro 选择 Classic 物理与 ros2_control 插件。
    robot_description = xacro.process_file(xacro_file, mappings={'GAZEBO': 'true', 'CLASSIC': 'true'}).toxml()

    rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    # 显示机器人模型；RViz 配置来自所选描述包。
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_ocs2',
        output='screen',
        arguments=["-d", rviz_config_file]
    )

    # 包含 gazebo_ros 的通用 Classic 启动文件，服务器与客户端由它创建。
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])]
        ),
        launch_arguments={"verbose": "false"}.items(),
    )

    # 从 robot_description 生成名为 robot 的实体；height 为世界坐标 z，单位 m。
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "robot", "-z", init_height],
        output="screen",
    )

    # 发布 URDF 与关节 TF，动态关节状态由后面的 joint_state_broadcaster 提供。
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'publish_frequency': 20.0,
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True
            }
        ],
    )

    # 名称沿用原示例，实际动作是 spawner 加载 joint_state_broadcaster（不是 GUI 发布器）。
    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    # 从 ros2_control 导出 IMU 状态供上层控制器使用。
    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    # 关节 PD 控制器负责四足关节命令接口。
    leg_pd_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["leg_pd_controller",
                   "--controller-manager", "/controller_manager"],
    )

    # 引导控制器包含四足控制状态，挂载到模型插件创建的 /controller_manager。
    unitree_guide_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["unitree_guide_controller", "--controller-manager", "/controller_manager"],
    )

    # 原示例先启动 PD spawner；它退出后并行启动 IMU、关节状态与引导控制器 spawner。
    # 此处使用进程退出事件触发后续动作，没有检查退出码，也没有串行等待 spawn_entity。
    return [
        rviz,
        robot_state_publisher,
        gazebo,
        spawn_entity,
        leg_pd_controller,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=leg_pd_controller,
                on_exit=[imu_sensor_broadcaster, joint_state_publisher, unitree_guide_controller],
            )
        ),
    ]


def generate_launch_description():
    # 可切换描述包；目标包需提供 xacro/robot.xacro 与 config/visualize_urdf.rviz。
    pkg_description = DeclareLaunchArgument(
        'pkg_description',
        default_value='go1_description',
        description='package for robot description'
    )

    # 模型生成时的初始离地高度（m），默认 0.5。
    height = DeclareLaunchArgument(
        'height',
        default_value='0.5',
        description='Init height in simulation'
    )

    return LaunchDescription([
        pkg_description,
        height,
        # 使用上述参数的最终值创建启动动作。
        OpaqueFunction(function=launch_setup),
    ])