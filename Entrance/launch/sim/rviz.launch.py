"""单独查看仿真建图，加载现有 SCAN / Faster-LIO RViz 布局。

Usage（先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/rviz.launch.py
  ros2 launch Entrance/launch/sim/rviz.launch.py rviz_config:=/绝对路径/布局.rviz
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库；sim 环境提供 RViz 所需的 ROS overlay。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

# 显示配置：此 RViz 文件；LIO 话题与 TF 配置见同目录 faster_lio.launch.py 中的 YAML 路径。
RVIZ_CONFIG = ROOT / 'Entrance/config/scan.rviz'


def setup(context):
    # 直接启动 rviz2：-d 指定布局；重命名节点避免与建图会话内的 RViz 重名；
    # use_sim_time:=true 使显示与 Gazebo /clock 同步；布局可由 rviz_config:= 覆盖。
    return [runtime.process(['/opt/ros/humble/bin/ros2', 'run', 'rviz2', 'rviz2',
        '-d', context.launch_configurations['rviz_config'], '--ros-args',
        '-r', '__node:=faster_lio_rviz', '-p', 'use_sim_time:=true'], 'sim')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rviz_config', default_value=str(RVIZ_CONFIG), description='RViz 显示布局'),
        OpaqueFunction(function=setup),
    ])
