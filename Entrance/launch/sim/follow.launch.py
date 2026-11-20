"""独立局部 A* 跟踪单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/follow.launch.py output:=/tmp/pure3d_follow
  ros2 launch Entrance/launch/sim/follow.launch.py --show-args
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库；跟踪节点以 common.planning.local_astar_follow 模块运行。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

# 规划器与跟踪参数：Entrance/config/scan.yaml。
# 本方案仅启动跟踪模块，output 参数必须指定。其余参数可通过 args 传入。


def generate_launch_description():
    # 参数映射为跟踪模块的 --output/--cost-limit/--backend；output 必须显式给出，
    # 其它跟踪参数（如 --bridge-body-height）可通过 args:="..." 追加。
    return runtime.module_launch('common.planning.local_astar_follow', 'sim', [
        ('output', '', '必填：路径跟踪输出目录'),
        ('cost_limit', '0.5', 'A* 代价上限'),
        ('backend', 'astar', '跟踪后端：astar 或 light'),
    ])
