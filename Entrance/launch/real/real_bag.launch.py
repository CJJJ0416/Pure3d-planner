"""真机 bag 回放与离线显示单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/real/real_bag.launch.py action:=run args:="--profile fastlio_official --model tiny --rviz"
  ros2 launch Entrance/launch/real/real_bag.launch.py --show-args
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库；bag 回放以 common.bag_tools.real_bag 模块运行。
# 本文件位于 Entrance/launch/real，上溯两级才是 Entrance/launch，公共库在 ../_internal。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

# 处理配置：Entrance/config/real_bag.yaml。
# LIO 与外参配置：Entrance/config/real/ 下按 profile 选择的 YAML。


def generate_launch_description():
    # 本方案没有固定参数；--profile、--model、--rviz 等通过 args:="..." 传给 real_bag。
    # 独立 ROS 域（默认 186）并限制 localhost，避免影响正在运行的仿真或真机会话。
    return runtime.module_launch('common.bag_tools.real_bag', 'real', [

    ], action='run', env_overrides={'ROS_DOMAIN_ID': __import__('os').environ.get('ROS_DOMAIN_ID', '186'), 'ROS_LOCALHOST_ONLY': '1'})
