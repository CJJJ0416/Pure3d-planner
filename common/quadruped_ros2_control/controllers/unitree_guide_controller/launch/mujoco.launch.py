# 用途：控制器包保留的 MuJoCo 控制示例，默认读取 go2_description。
# 本文件启动 ros2_control 与 RViz；实际后端由机器人 xacro 的 ros2_control 硬件插件决定。
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# 读取最终描述包参数并展开机器人 xacro，此入口不设置 Gazebo 模式开关。
def launch_setup(context, *args, **kwargs):
    package_description = context.launch_configurations['pkg_description']
    pkg_path = os.path.join(get_package_share_directory(package_description))

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    # robot_control.yaml 提供 controller_manager 与控制器参数；硬件插件定义来自机器人描述。
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare(package_description),
            "config",
            "robot_control.yaml",
        ]
    )

    rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    # RViz 用于显示所选描述包的机器人模型与状态。
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_ocs2',
        output='screen',
        arguments=["-d", rviz_config_file]
    )

    # 发布 robot_description 与关节 TF，供控制器管理器和 RViz 使用。
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

    # 独立启动 ros2_control_node；其私有 robot_description 输入接到公开模型话题。
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_controllers],
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        output="both",
    )

    # 首先加载 joint_state_broadcaster，提供实际关节状态（此变量不是 GUI 发布器）。
    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    # 随后加载 IMU 状态广播器。
    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    # 最后加载四足引导控制器。
    unitree_guide_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["unitree_guide_controller", "--controller-manager", "/controller_manager"],
    )

    # 顺序：关节状态 spawner 退出 → IMU spawner 退出 → 引导控制器。
    # 原示例通过退出事件串联，没有进一步检查前一步退出码。
    return [
        rviz,
        robot_state_publisher,
        controller_manager,
        joint_state_publisher,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[imu_sensor_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=imu_sensor_broadcaster,
                on_exit=[unitree_guide_controller],
            )
        ),
    ]


def generate_launch_description():
    # 目标描述包需提供 xacro/robot.xacro、config/robot_control.yaml 及 RViz 配置。
    pkg_description = DeclareLaunchArgument(
        'pkg_description',
        default_value='go2_description',
        description='package for robot description'
    )

    return LaunchDescription([
        pkg_description,
        # 参数解析后才构造模型和控制器节点。
        OpaqueFunction(function=launch_setup),
    ])
