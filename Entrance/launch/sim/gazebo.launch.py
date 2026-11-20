"""仿真单项入口：选择场景，启动 A1、MID360、控制器与所选规划器。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building
  ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=terrain-prepared gui:=false rviz:=false
  ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building x:=-5.5 y:=6.0 height:=0.5 yaw:=0.0
  ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=terrain-prepared dry_run:=true
  ros2 launch Entrance/launch/sim/gazebo.launch.py list_scenes:=true
  ros2 launch Entrance/launch/sim/gazebo.launch.py --show-args
"""
import importlib.util
import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource

# 修改默认场景：Entrance/config/startup.yaml；修改 world、模型、出生点：Entrance/config/scenes/<场景>.yaml。
# SCAN 规划参数：Entrance/config/scan.yaml。
# 话题配置：Entrance/config/local_planners.yaml。
LAUNCH_ROOT = Path(__file__).resolve().parent      # Entrance/launch/sim
COMMON_ROOT = LAUNCH_ROOT.parent                   # Entrance/launch，存放 sim/real 共用 runtime.py


def load_helper(name, root=LAUNCH_ROOT):
    # 按文件路径加载 _internal 下的模块，避免依赖 sys.path 或把 launch 变成 Python 包。
    path = root / '_internal' / (name + '.py')
    spec = importlib.util.spec_from_file_location('pure3d_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(context):
    scenes = load_helper('scenes')
    values = context.launch_configurations
    # list_scenes 只读取 config/scenes 下的文件名，不解析资源，也不启动任何进程。
    if values['list_scenes'].lower() == 'true':
        return [LogInfo(msg='可用场景：' + ', '.join(scenes.scene_names()))]
    # 两个控制参数必须是合法布尔值；dry_run=true 仅做资源解析与检查。
    for key in ('dry_run', 'list_scenes'):
        if values[key].lower() not in ('true', 'false'):
            raise ValueError(f'{key} 必须为 true 或 false')
    preview = values['dry_run'].lower() == 'true'
    # 收集命令行显式给出的场景参数；空字符串表示“沿用 YAML 默认值”。
    overrides = {key: values[key] for key in (*scenes.ARGUMENT_DEFAULTS, 'terrain_dir') if values.get(key, '') != ''}
    if preview:
        plan = scenes.scene_plan(values['scene'], overrides)
        return [LogInfo(msg=json.dumps(plan, ensure_ascii=False, indent=2))]
    runtime = load_helper('runtime', COMMON_ROOT)
    env = runtime.environment('sim')
    # ament 查询读取进程环境；子节点读取 launch 上下文环境，两者同步到同一 overlay。
    os.environ.update(env)
    context.environment.update(env)
    from ament_index_python.packages import get_package_share_directory
    # 实际启动时使用安装后的 unitree_gazebo 资源目录，并允许 terrain 场景现场导出。
    assets = Path(get_package_share_directory('unitree_gazebo'))
    plan = scenes.scene_plan(values['scene'], overrides, assets=assets, prepare=True)
    options = plan['launch_arguments']
    # 未显式指定策略权重时使用包内楼梯策略；输入输出均使用 launch 参数而非环境变量拼接。
    if not options['model_path']:
        options['model_path'] = str(Path(get_package_share_directory('unitree_guide_controller'),
                                         'policies/policy_act_inference_stair.pt'))
    # 将解析后的 world、模型目录、出生位姿与规划参数整体交给内部 A1 启动实现。
    return [
        LogInfo(msg=f'场景：{plan["scene"]}；配置：{plan["profile"]}；world：{options["world"]}'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(LAUNCH_ROOT / '_internal/a1.launch.py')),
                                 launch_arguments=options.items()),
    ]


def generate_launch_description():
    scenes = load_helper('scenes')
    # 场景默认值来自 startup.yaml（可被 PURE3D_SIM_SCENE 覆盖）；其余参数留空，
    # 由场景 YAML 与 startup.yaml 的 launch_arguments 逐层填充，命令行显式值优先级最高。
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value=scenes.default_scene(), choices=scenes.scene_names(),
                              description='场景名称，对应 Entrance/config/scenes/<名称>.yaml'),
        DeclareLaunchArgument('dry_run', default_value='false', description='只解析并检查资源，不启动仿真'),
        DeclareLaunchArgument('list_scenes', default_value='false', description='只显示可用场景名称'),
        DeclareLaunchArgument('terrain_dir', default_value='', description='临时覆盖 terrain 导出目录'),
        *[DeclareLaunchArgument(key, default_value='', description='空值继承场景 YAML 与公共默认值')
          for key in scenes.ARGUMENT_DEFAULTS],
        OpaqueFunction(function=setup),
    ])
