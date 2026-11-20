"""局部 cost 与耗时监测单项方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/cost.launch.py model:=tiny rviz:=true
  ros2 launch Entrance/launch/sim/cost.launch.py --show-args
  ros2 launch Entrance/launch/sim/cost.launch.py config:=/绝对路径/配置.yaml
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库，获得工作区根目录与 module_launch/process 等辅助函数。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

# 修改网络、话题、盒子、KNN 过滤和推理参数：此 YAML 是该方案的配置入口。
CONFIG = ROOT / 'Entrance/config/local_cost.yaml'


def generate_launch_description():
    # 参数元组 (名称, 默认值, 说明) 会映射为 cost_session 的 --config/--model/--rviz/--output/--no-monitor；
    # flags 中的参数按布尔开关处理，只有值为 true 时才追加对应命令行开关。
    return runtime.module_launch('common.cost.session', 'sim', [
        ('config', str(CONFIG), '业务配置 YAML 的完整路径'),
        ('model', 'tiny', '网络选择：tiny 或 small'),
        ('rviz', 'false', '是否显示局部 cost RViz'),
        ('output', '', '新会话输出目录'),
        ('no_monitor', 'false', '是否关闭耗时监测'),
    ], flags=('rviz', 'no_monitor',))
