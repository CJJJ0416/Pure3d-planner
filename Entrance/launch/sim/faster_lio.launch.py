"""仿真 Faster-LIO 独立建图方案。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/faster_lio.launch.py
  ros2 launch Entrance/launch/sim/faster_lio.launch.py check_config:=true
  ros2 launch Entrance/launch/sim/faster_lio.launch.py lio_config:=/绝对路径/lio.yaml frames_config:=/绝对路径/frames.yaml
  ros2 launch Entrance/launch/sim/faster_lio.launch.py --show-args
"""
import importlib.util
from pathlib import Path

# 按路径加载公共运行库；sim 环境决定 ROS overlay、默认输出目录与建图显示入口。
spec = importlib.util.spec_from_file_location('pure3d_runtime', Path(__file__).resolve().parents[1] / '_internal/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ROOT = runtime.ROOT

# 修改话题、预处理和估计参数：此 Faster-LIO YAML。
LIO_CONFIG = ROOT / 'Entrance/config/sim/faster_lio.yaml'
# 修改机身/雷达/IMU 外参、地图锚定和 TF：此安装配置 YAML。
FRAMES_CONFIG = ROOT / 'Entrance/config/sim/frames.yaml'


def generate_launch_description():
    # lio_launch(profile, frames, config)：profile 选择 shell 环境与输出根目录，
    # frames 提供安装外参/地图锚定，config 提供 Faster-LIO 算法参数。
    return runtime.lio_launch('sim', FRAMES_CONFIG, LIO_CONFIG)
