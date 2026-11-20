"""历史 bag 采集、回放与处理单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/mission.launch.py action:=plan args:="--output /tmp/pure3d_mission"
  ros2 launch Entrance/launch/sim/mission.launch.py --show-args
"""
import importlib.util
from pathlib import Path

# 以文件路径动态加载公共运行库，保持本文件可独立被 ros2 launch 执行。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
# ROOT 为 Pure3D_ws 工作区根目录；任务目录与输出目录由 args:=... 传入。
ROOT = runtime.ROOT

# cost 子命令的 YAML：Entrance/config/whole_map.yaml（用 args:="--config 路径" 覆盖）。
# 场景与出生点：Entrance/config/scenes/；任务数据由 --mission 对应目录管理。


def generate_launch_description():
    # 本方案没有固定的公开参数；任务目录、--rviz 等选项全部通过 args:="..." 原样传给 bag_mission。
    # action 默认 plan，可换成 bag_mission 支持的其它子命令（如 collect/view）。
    return runtime.module_launch('common.bag_tools.bag_mission', 'sim', [

    ], action='plan')
