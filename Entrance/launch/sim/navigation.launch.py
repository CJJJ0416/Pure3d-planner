"""人工目标与整图 A* 导航单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/navigation.launch.py rviz:=true
  ros2 launch Entrance/launch/sim/navigation.launch.py --show-args
  ros2 launch Entrance/launch/sim/navigation.launch.py config:=/绝对路径/配置.yaml
"""
import importlib.util
from pathlib import Path

# 以文件路径动态加载公共运行库，避免把 Entrance/launch 变成会遮蔽 ROS launch 包的 Python 包。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
# ROOT 为 Pure3D_ws 工作区根目录，用于拼出配置与默认输出路径。
ROOT = runtime.ROOT

# 修改网络、话题、地图盒子、代价触发和推理参数：此 YAML 是该方案的配置入口。
CONFIG = ROOT / 'Entrance/config/navigation.yaml'


def generate_launch_description():
    # module_launch 把下面 (名称, 默认值, 说明) 逐项转换为 common.planning.navigation 的命令行参数：
    # 非空值传 --名称=值，flags 中的参数仅在 true 时附加同名开关，其余参数由 args:=... 原样透传。
    return runtime.module_launch('common.planning.navigation', 'sim', [
        ('config', str(CONFIG), '业务配置 YAML 的完整路径'),
        ('planner_backend', '', '规划后端覆盖值：surface 或 dc'),
        ('rviz', 'false', '是否启动 RViz'),
        ('output', '', '输出目录'),
    ], flags=('rviz',))
