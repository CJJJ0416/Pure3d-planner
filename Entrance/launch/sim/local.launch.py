"""局部点云接收与 cost 节点单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/local.launch.py model:=tiny rviz:=true
  ros2 launch Entrance/launch/sim/local.launch.py --show-args
  ros2 launch Entrance/launch/sim/local.launch.py config:=/绝对路径/配置.yaml
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库，获得工作区根目录与 module_launch 参数转换。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

# 修改网络、话题、盒子和 KNN 过滤参数：此 YAML 是该方案的配置入口。
CONFIG = ROOT / 'Entrance/config/local_cost.yaml'


def generate_launch_description():
    # 参数映射为 common.cost.local 的 --config/--model/--mode/--rviz/--output；
    # model 为空时沿用 YAML 中的默认模型，rviz 为 true 时才追加 --rviz。
    return runtime.module_launch('common.cost.local', 'sim', [
        ('config', str(CONFIG), '业务配置 YAML 的完整路径'),
        ('model', '', '空值沿用 YAML 中的模型'),
        ('mode', 'cost', '运行模式：cost 或 cloud'),
        ('rviz', 'false', '是否显示局部 cost RViz'),
        ('output', '', '输出目录'),
    ], flags=('rviz',))
