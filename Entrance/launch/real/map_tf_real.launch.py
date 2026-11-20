"""单独调试真机 LIO 到公共地图的 TF 转换。

Usage（先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/real/map_tf_real.launch.py
  ros2 launch Entrance/launch/real/map_tf_real.launch.py frames_config:=/绝对路径/frames.yaml

正常建图方案已包含转换节点；调试时关闭原建图方案中的同一转换节点。
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库，获得工作区根目录与 process 辅助函数。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

# 修改机身/雷达/IMU 外参与地图锚定：此 YAML。
FRAMES_CONFIG = ROOT / 'Entrance/config/real/frames.yaml'


def setup(context):
    # 单独运行 map_anchor.py，按 frames YAML 发布 world -> map 锚定与公共 TF。
    # 环境变量让脚本与正常建图流程读取同一份安装配置和输出目录；可用环境变量覆盖默认值。
    return [runtime.process(['/usr/bin/python3', str(ROOT / 'Simulator/unitree_gazebo/scripts/map_anchor.py'),
        '--config', context.launch_configurations['frames_config']], 'real', env_overrides={
        'PURE3D_LIO_FRAMES_CONFIG': context.launch_configurations['frames_config'],
        'PURE3D_LIO_OUTPUT': os.environ.get('PURE3D_LIO_OUTPUT', str(ROOT / 'outputs/unitree_real/lio')),
    })]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('frames_config', default_value=os.environ.get('PURE3D_LIO_FRAMES_CONFIG', str(FRAMES_CONFIG)),
                              description='安装外参与地图 TF 配置 YAML'),
        OpaqueFunction(function=setup),
    ])
