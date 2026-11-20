"""键盘控制单项方案：保留模式键，是否发布速度由参数决定。

Usage（在交互终端执行，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/keyboard.launch.py publish_cmd_vel:=false
  ros2 launch Entrance/launch/sim/keyboard.launch.py publish_cmd_vel:=true
  ros2 launch Entrance/launch/sim/keyboard.launch.py --show-args
按键：2 蹲下/站立，7 进入 RL，P 缓停/恢复，1 被动。
"""
import importlib.util
import os
import sys
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

# 控制器 YAML：Simulator/quadruped_ros2_control/descriptions/unitree/a1_description/config/gazebo.yaml。
# SCAN 控制时设置 publish_cmd_vel:=false，避免两个速度发布源。
# 按路径加载公共运行库；process 会在 sim 环境中启动子进程。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def setup(context):
    # 先校验布尔参数，避免把拼写错误静默当成 false。
    value = context.launch_configurations['publish_cmd_vel'].lower()
    if value not in ('true', 'false'):
        raise ValueError('publish_cmd_vel 必须为 true 或 false')
    # ROS launch 默认给子进程管道输入；显式连接当前终端，保留键盘的真实交互输入。
    if not sys.stdin.isatty():
        raise RuntimeError('键盘方案需要交互终端，请在独立终端运行此 launch')
    terminal = os.ttyname(sys.stdin.fileno())
    # bash 包装器先把 tty 保存到变量，再 exec ros2 run，并把 stdin 重定向回该 tty。
    return [runtime.process(['bash', '-c', 'terminal="$1"; shift; exec "$@" <"$terminal"',
                             'pure3d-keyboard', terminal,
                             '/opt/ros/humble/bin/ros2', 'run', 'keyboard_input', 'keyboard_input',
                             '--ros-args', '-p', 'publish_cmd_vel:=' + value], 'sim',
                            env_overrides={'PYTHONUNBUFFERED': '1'})]


def generate_launch_description():
    # publish_cmd_vel 默认 true；由 SCAN 控制速度时传入 false，键盘只保留模式切换。
    return LaunchDescription([
        DeclareLaunchArgument('publish_cmd_vel', default_value='true', description='键盘是否发布 /cmd_vel；SCAN 模式请设 false'),
        OpaqueFunction(function=setup),
    ])
