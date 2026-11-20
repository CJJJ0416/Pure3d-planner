# 保留的现代 Gazebo 控制器示例，需配套模型包。
# Usage（先加载对应 ROS 工作区）：ros2 launch unitree_guide_controller gazebo.launch.py pkg_description:=a1_description
# 配置位置：所选描述包的 config/robot_control.yaml。
# 用途：控制器包保留的现代 Gazebo（ros_gz）示例，默认使用 go2_description。
# 它与 gazebo_classic.launch.py 使用不同的 Gazebo 接口；当前 A1 主入口在 unitree_gazebo 包。
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


# 参数解析后从所选描述包加载模型；height 是实体的初始世界 z 坐标（m）。
def launch_setup(context, *args, **kwargs):
    package_description = context.launch_configurations['pkg_description']
    init_height = context.launch_configurations['height']
    pkg_path = os.path.join(get_package_share_directory(package_description))

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    # 仅打开 GAZEBO 开关，具体仿真插件由所选描述包的 xacro 决定。
    robot_description = xacro.process_file(xacro_file, mappings={'GAZEBO': 'true'}).toxml()

    rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    # 使用描述包自带配置显示机器人模型与 TF。
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_ocs2',
        output='screen',
        arguments=["-d", rviz_config_file]
    )

    # ros_gz_sim/create 从 robot_description 生成实体，允许名称冲突时自动重命名。
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name',
                   'robot', '-allow_renaming', 'true', '-z', init_height],
    )

    # 发布机器人描述与关节 TF；关节数据由 joint_state_broadcaster 输出。
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

    # 变量名沿用原示例，实际加载的是 joint_state_broadcaster。
    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    # 加载四足控制所需的 IMU 状态广播器。
    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    # 加载四足引导控制器，控制器管理器由模型中的仿真控制插件提供。
    unitree_guide_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["unitree_guide_controller", "--controller-manager", "/controller_manager"],
    )

    return [
        rviz,
        Node(
            # 单向将 Gazebo /clock 桥接到 ROS；方括号表示 Gazebo → ROS 的桥接方向。
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            output='screen'
        ),
        # 启动 empty.sdf 空场景：-r 立即运行，-v 4 设置 Gazebo 日志详细程度。
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                       'launch',
                                       'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', [' -r -v 4 empty.sdf'])]),
        robot_state_publisher,
        gz_spawn_entity,
        # 实体生成进程退出后启动 IMU 与关节状态 spawner；此原示例未检查退出码。
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[imu_sensor_broadcaster, joint_state_publisher],
            )
        ),
        # 关节状态 spawner 退出后启动引导控制器，未额外等待 IMU spawner 完成。
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[ unitree_guide_controller],
            )
        ),
    ]


def generate_launch_description():
    # 目标描述包需包含机器人 xacro、控制插件配置与 RViz 配置，默认 go2_description。
    pkg_description = DeclareLaunchArgument(
        'pkg_description',
        default_value='go2_description',
        description='package for robot description'
    )

    # 初始高度（m），默认 0.5。
    height = DeclareLaunchArgument(
        'height',
        default_value='0.5',
        description='Init height in simulation'
    )

    return LaunchDescription([
        pkg_description,
        height,
        # 使用最终参数生成模型和仿真启动动作。
        OpaqueFunction(function=launch_setup),
    ])
