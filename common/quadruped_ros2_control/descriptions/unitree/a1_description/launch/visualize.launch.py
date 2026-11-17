# 用途：A1 URDF/TF 的静态可视化入口，通过 GUI 调整关节并在 RViz 中查看模型。
# 此入口不启动物理仿真、四足控制器或规划器，适合检查模型与关节连接。
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

import xacro

package_description = "a1_description"


# 参数解析后读取 robot_type，展开安装包中的 robot.xacro，生成 robot_description。
def process_xacro(context):
    robot_type_value = context.launch_configurations['robot_type']
    pkg_path = os.path.join(get_package_share_directory(package_description))
    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    robot_description_config = xacro.process_file(xacro_file, mappings={'robot_type': robot_type_value})
    return robot_description_config.toxml()


# 运行时生成模型并创建状态发布节点；OpaqueFunction 保证拿到命令行参数的实际值。
def launch_setup(context, *args, **kwargs):
    robot_description = process_xacro(context)
    return [
        Node(
            # 根据 joint_states 发布机器人各连杆 TF；固定关节使用静态 TF。
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'publish_frequency': 100.0,
                    'use_tf_static': True,
                    'robot_description': robot_description
                }
            ],
        ),
        Node(
            # GUI 滑块发布关节位置供模型展示，不向真实机器人或仿真控制器发送运动命令。
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher',
            output='screen',
        )
    ]


def generate_launch_description():
    # 模型类型参数传入 xacro；默认使用 a1。
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='a1',
        description='Type of the robot'
    )

    # RViz 使用该描述包随附的 URDF 可视化配置。
    rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    return LaunchDescription([
        robot_type_arg,
        # 先声明参数，再生成模型及状态节点；RViz 与这些节点一起运行。
        OpaqueFunction(function=launch_setup),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz_ocs2',
            output='screen',
            arguments=["-d", rviz_config_file]
        )
    ])
