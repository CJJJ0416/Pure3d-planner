"""场景 YAML 的唯一读取位置，供 Gazebo launch 与整体方案共用。"""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

# 本目录不建立名为 launch 的 Python 包，避免遮蔽 ROS 自带的 launch 包。
# 本文件位于 Entrance/launch/sim/_internal，上溯三级得到 Entrance 目录。
CONFIG_ROOT = Path(__file__).resolve().parents[3] / 'config'
# 出生位姿只允许这四个键，单位分别为米、米、米、弧度。
POSE_KEYS = ('x', 'y', 'height', 'yaw')
# 所有可被命令行/场景 YAML 覆盖的启动参数及空值默认；实际默认值按 startup.yaml → 场景 YAML → 命令行逐层填充。
ARGUMENT_DEFAULTS = {
    'gui': 'true', 'rviz': 'true', 'rviz_config': '', 'debug': 'false',
    'mid360': 'true', 'mid360_samples': '10000', 'mid360_downsample': '1',
    'device': 'cpu', 'trace': 'false', 'clock_rate': '100.0', 'model_path': '',
    'world': '', 'scene_models_path': '', 'x': '', 'y': '', 'height': '', 'yaw': '',
    'odom_x': '0.0', 'odom_y': '0.0', 'odom_z': '0.0',
    'planner': 'scan', 'flight_type': '1', 'cost_astar_output': '',
    'cost_astar_limit': '0.8', 'cost_planner_backend': 'light_scan',
}


def settings():
    # 读取 startup.yaml：默认场景与所有场景共享的 launch_arguments。
    return yaml.safe_load((CONFIG_ROOT / 'startup.yaml').read_text())


def scene_names():
    # 场景名即 config/scenes 下的 YAML 文件名（不含扩展名），排序后用于枚举与校验。
    return tuple(path.stem for path in sorted((CONFIG_ROOT / 'scenes').glob('*.yaml')))


def default_scene():
    # 环境变量优先于 startup.yaml，兼容原有的场景选择习惯。
    return os.environ.get('PURE3D_SIM_SCENE') or settings()['default_scene']


def read_scene(name):
    # 先限制字符集，再确认是已登记的场景文件，避免路径穿越与拼写错误。
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or name not in scene_names():
        raise ValueError(f'未知场景 {name!r}，可选：{", ".join(scene_names())}')
    return yaml.safe_load((CONFIG_ROOT / 'scenes' / (name + '.yaml')).read_text())


def workspace():
    # 优先使用包含 Simulator/unitree_gazebo 的源码工作区；软链接/移动安装时回退到 PURE3D_WS。
    root = CONFIG_ROOT.parent.parent
    if (root / 'Simulator/unitree_gazebo').is_dir():
        return root
    if os.environ.get('PURE3D_WS'):
        return Path(os.environ['PURE3D_WS']).expanduser().resolve()
    raise ValueError('请设置 PURE3D_WS 为工作区，以解析数据集相对路径')


def resolve_path(value, assets):
    # @assets/ 前缀表示相对 unitree_gazebo 包资源目录；其余相对路径以工作区为基准。
    if value.startswith('@assets/'):
        return (assets / value[len('@assets/'):]).resolve()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace() / path).resolve()


def require_files(paths):
    # 场景资源必须存在且非空；空文件通常意味着导出中断或挂载失败。
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'场景资源缺失：{path}')


def validate_arguments(arguments):
    # 只允许 ARGUMENT_DEFAULTS 中的键；未知参数直接报错，避免拼写错误被静默忽略。
    unknown = set(arguments) - ARGUMENT_DEFAULTS.keys()
    if unknown:
        raise ValueError('未知启动参数：' + ', '.join(sorted(unknown)))
    # 枚举型参数：规划器、推理设备、飞行模式与 cost 后端。
    choices = {'planner': ('scan', 'ego', 'None'), 'device': ('cpu', 'cuda'),
               'flight_type': ('1', '2', '3', '4'),
               'cost_planner_backend': ('light', 'light_scan', 'astar')}
    for key, values in choices.items():
        if key in arguments and str(arguments[key]) not in values:
            raise ValueError(f'{key} 可选值：{values}')
    # 布尔型参数只接受 true/false（大小写不敏感）。
    for key in ('gui', 'rviz', 'debug', 'mid360', 'trace'):
        if key in arguments and str(arguments[key]).lower() not in ('true', 'false'):
            raise ValueError(f'{key} 必须为 true 或 false')
    # 出生位姿必须是有限数值；空字符串表示沿用上游默认值。
    for key in POSE_KEYS:
        if arguments.get(key, '') != '' and not math.isfinite(float(arguments[key])):
            raise ValueError(f'{key} 必须为有限数值')
    # 模式 4 依赖 SCAN 的 cost 参考，必须与 scan 规划器组合。
    if str(arguments.get('flight_type')) == '4' and arguments.get('planner') != 'scan':
        raise ValueError('flight_type:=4 要求 planner:=scan')


def scene_plan(name, overrides=None, *, assets=None, prepare=False):
    """先解析与检查资源；预览不会导出地形、启动节点或清理现有会话。"""
    cfg = read_scene(name)
    assets = Path(assets) if assets else workspace() / 'Simulator/unitree_gazebo'
    overrides = dict(overrides or {})
    # terrain_dir 单独取出：它只影响 terrain 场景的导出/复用目录，不参与 A1 参数。
    terrain_dir = overrides.pop('terrain_dir', '')
    # 先检查未知参数，组合默认值后再检查相互约束。
    unknown = set(overrides) - ARGUMENT_DEFAULTS.keys()
    if unknown:
        raise ValueError('未知启动参数：' + ', '.join(sorted(unknown)))
    # 优先级：内置默认 → startup.yaml 公共默认 → 场景 YAML → 命令行覆盖。
    arguments = dict(ARGUMENT_DEFAULTS)
    arguments.update(settings().get('launch_arguments', {}))
    arguments.update(cfg.get('launch_arguments', {}))
    report = dict(scene=name, description=cfg['description'], preparation=None)
    kind = cfg['kind']
    if kind == 'model':
        # model 场景：候选模型目录中选第一个含 model.sdf 的；可用 model_env 指向自定义副本。
        selected = os.environ.get(cfg.get('model_env', ''))
        candidates = [selected] if selected else cfg['model_candidates']
        model = next((resolve_path(value, assets) for value in candidates
                      if (resolve_path(value, assets) / 'model.sdf').is_file()), None)
        if model is None:
            raise FileNotFoundError(f'下列目录均无 model.sdf：{candidates}')
        require_files(model / name for name in cfg.get('required_model_files', ['model.sdf']))
        # world 与模型资源保持同一目录来源，避免 world 引用了另一份模型。
        arguments.update(world=str((model / cfg['world_from_model']).resolve()),
                         scene_models_path=str(model.parent))
        arguments.update({key: str(cfg['spawn'][key]) for key in POSE_KEYS})
    elif kind == 'world':
        # world 场景：world 与模型目录均可独立指定，出生点直接来自 YAML。
        models = resolve_path(cfg['models'], assets)
        require_files(models / name for name in cfg.get('required_model_files', []))
        arguments.update(world=str(resolve_path(cfg['world'], assets)), scene_models_path=str(models))
        arguments.update({key: str(cfg['spawn'][key]) for key in POSE_KEYS})
    elif kind == 'terrain':
        # terrain 场景：可由源 OBJ 现场导出（prepare=true），也可直接复用导出目录。
        output = resolve_path(terrain_dir or os.environ.get('PURE3D_TERRAIN_OUTPUT') or cfg['output'], assets)
        if cfg.get('prepare'):
            # 导出前检查源场景完整性；dry_run 时只把命令写入报告，不执行导出。
            source = resolve_path(os.environ.get('PURE3D_TERRAIN_SCENE') or cfg['source'], assets)
            require_files(source / name for name in ('terrain.obj', 'scene.json'))
            command = [sys.executable, str(assets / 'sim/prepare_world.py'), '--scene', str(source),
                       '--output', str(output), '--floor', str(int(os.environ.get('PURE3D_TERRAIN_FLOOR', cfg['floor'])))]
            report['preparation'] = command
            if prepare:
                subprocess.run(command, check=True)
        arguments['world'] = str(output / 'terrain.world')
        # 预览新导出场景时不读取可能过期的 spawn；实际导出后才读取。
        if prepare or not cfg.get('prepare'):
            require_files(output / name for name in ('terrain.world', 'terrain.stl', 'spawn.json'))
            spawn = json.loads((output / 'spawn.json').read_text())
            arguments.update({key: str(spawn[key]) for key in POSE_KEYS})
        report['terrain_dir'] = str(output)
    else:
        raise ValueError(f'未知场景类型：{kind}')
    # terrain_dir 只对 terrain 场景有意义，提前拦截误用。
    if terrain_dir and kind != 'terrain':
        raise ValueError('terrain_dir 只适用于 terrain 场景')
    # 命令行覆盖最后应用，并统一转成字符串以适配 launch 参数接口。
    arguments.update({key: str(value) for key, value in overrides.items()})
    arguments = {key: str(value) for key, value in arguments.items()}
    validate_arguments(arguments)
    # 对需要立即启动的场景再次确认 world 文件存在（导出场景此时已生成）。
    if kind != 'terrain' or prepare or not cfg.get('prepare'):
        require_files([Path(arguments['world'])])
    report['launch_arguments'] = arguments
    report['profile'] = str(CONFIG_ROOT / 'scenes' / (name + '.yaml'))
    return report
