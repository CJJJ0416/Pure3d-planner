"""整图分块与代价预测单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/whole_map.launch.py mode:=blocks rviz:=true
  ros2 launch Entrance/launch/sim/whole_map.launch.py --show-args
  ros2 launch Entrance/launch/sim/whole_map.launch.py config:=/绝对路径/配置.yaml
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库；本文件不依赖 Entrance 成为 Python 包。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

# 修改整图话题、分块尺寸、到达判定和推理参数：此 YAML 是该方案的配置入口。
CONFIG = ROOT / 'Entrance/config/whole_map.yaml'


def generate_launch_description():
    # 参数映射为 common.cost.whole_map 的 --config/--mode/--rviz/--output；
    # mode 取 cost 或 blocks，rviz 为 true 时才追加 --rviz。
    return runtime.module_launch('common.cost.whole_map', 'sim', [
        ('config', str(CONFIG), '业务配置 YAML 的完整路径'),
        ('mode', 'cost', '运行模式：cost 或 blocks'),
        ('rviz', 'false', '是否启动 RViz'),
        ('output', '', '输出目录'),
    ], flags=('rviz',))
