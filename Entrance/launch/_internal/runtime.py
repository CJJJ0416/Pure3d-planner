"""启动文件共用的环境加载与参数转换；业务 YAML 路径仍由各 launch 明确给出。"""
import os
from pathlib import Path
import shlex
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction

# 源码/软链接安装优先使用文件所在工作区，避免其他工程遗留的 PURE3D_WS 串入。
# 若当前目录不是完整工作区，则回退到显式 PURE3D_WS 环境变量。
ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / 'Simulator/unitree_gazebo').is_dir():
    ROOT = Path(os.environ['PURE3D_WS']).expanduser().resolve()


def environment(profile):
    """只加载所选 sim/real 环境；参数通过 argv 传递，不拼接进 shell 代码。"""
    path = ROOT / 'Entrance/shell/_internal' / (profile + '_env.bash')
    # 在独立 bash 中 source 环境库，再以 NUL 分隔导出完整环境，避免解析换行或空格路径。
    result = subprocess.run(['bash', '-c', 'source "$1" >/dev/null && env -0', 'pure3d', str(path)],
                            capture_output=True, check=True)
    env = dict(row.split('=', 1) for row in result.stdout.decode().split('\0') if '=' in row)
    # 强制把工作区根目录放到 PYTHONPATH 首位，从本工作区导入 common、planner 与核心组件。
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PURE3D_WS'] = str(ROOT)
    return env


def process(command, profile, *, env_overrides=None):
    # 所有子进程固定在工作区根目录运行，并继承所选 sim/real 环境；覆盖项最后应用。
    env = environment(profile)
    env.update(env_overrides or {})
    return ExecuteProcess(cmd=command, cwd=str(ROOT), additional_env=env, output='screen')


def module_launch(module, profile, parameters, *, flags=(), action=None, env_overrides=None):
    """将公开 launch 参数转换为现有 Python 入口参数，保留原模块实现。"""
    def setup(context):
        # 参数解析在 OpaqueFunction 中完成：DeclareLaunchArgument 已全部注册，可直接读取最终值。
        values = context.launch_configurations
        command = ['/usr/bin/python3', '-u', '-m', module]
        # action 作为位置参数放在最前，对应原 Python 工具的子命令。
        if action is not None:
            command.append(values['action'])
        # 参数名中的下划线转换为命令行常用的连字符；空字符串表示“不传，沿用模块默认值”。
        for name, _default, _description in parameters:
            value = values[name]
            if name in flags:
                # 布尔开关只接受 true/false；true 时追加开关本身，false 时完全省略。
                if value.lower() not in ('true', 'false'):
                    raise ValueError(f'{name} 必须为 true 或 false')
                if value.lower() == 'true':
                    command.append('--' + name.replace('_', '-'))
            elif value:
                command.extend(['--' + name.replace('_', '-'), value])
        # args 用于原样透传未在本 launch 中声明的参数，按 shell 规则拆分成多个 argv。
        command.extend(shlex.split(values['args']))
        return [LogInfo(msg=f'入口：{module}；配置：{values.get("config", "见本 launch 中的中文注释")}'),
                process(command, profile, env_overrides=env_overrides)]
    # 所有参数先声明，再交给 OpaqueFunction 统一装配，保证 --show-args 能看到完整参数表。
    return LaunchDescription([
        *([DeclareLaunchArgument('action', default_value=action, description='Python 工作流子命令')] if action is not None else []),
        *[DeclareLaunchArgument(name, default_value=str(default), description=description)
          for name, default, description in parameters],
        DeclareLaunchArgument('args', default_value='', description='其余 Python 命令行参数，含空格时整体加引号'),
        OpaqueFunction(function=setup),
    ])


def lio_launch(profile, frames, config):
    """独立建图方案；参数文件、安装外参、输出目录均在 launch 参数中可见。"""
    def setup(context):
        values = context.launch_configurations
        arguments = []
        # --bag 仅用于仿真离线流程：把 bag 目录交给 lio_runtime.bash 做契约校验。
        if values['bag']:
            arguments += ['--bag', values['bag']]
        if values['check_config'].lower() not in ('true', 'false'):
            raise ValueError('check_config 必须为 true 或 false')
        if values['check_config'].lower() == 'true':
            arguments += ['--check-config']
        # 其余 ROS 参数（如 -p pcd_save.pcd_save_en:=false）原样透传。
        arguments += shlex.split(values['args'])
        # 输出目录优先级：launch 参数 → PURE3D_LIO_OUTPUT → sim/real 默认目录。
        output = values['output'] or os.environ.get('PURE3D_LIO_OUTPUT') or str(ROOT / (
            'outputs/gazebo_lio' if profile == 'sim' else 'outputs/unitree_real/lio'))
        return [
            LogInfo(msg=f'LIO 参数：{values["lio_config"]}；安装外参：{values["frames_config"]}'),
            # lio_runtime.bash 负责校验配置、等待 IMU、启动 map_anchor 与 Faster-LIO。
            process(['bash', str(ROOT / 'Entrance/shell/_internal/lio_runtime.bash'), *arguments], profile,
                    env_overrides={'PURE3D_LIO_CONFIG': values['lio_config'],
                                   'PURE3D_LIO_FRAMES_CONFIG': values['frames_config'],
                                   'PURE3D_LIO_OUTPUT': output}),
        ]
    return LaunchDescription([
        # frames_config/lio_config 支持同名环境变量作为默认值，便于复用外部标定文件。
        DeclareLaunchArgument('frames_config', default_value=os.environ.get('PURE3D_LIO_FRAMES_CONFIG', str(frames)),
                              description='机身/雷达/IMU 安装外参与地图 TF 配置 YAML'),
        DeclareLaunchArgument('lio_config', default_value=os.environ.get('PURE3D_' + profile.upper() + '_LIO_CONFIG', str(config)),
                              description='Faster-LIO 话题、预处理、估计参数 YAML'),
        DeclareLaunchArgument('output', default_value='', description='运行输出目录，空值使用对应 sim/real 默认目录'),
        DeclareLaunchArgument('bag', default_value='', description='仅仿真离线流程使用，需 ROS_DOMAIN_ID=184'),
        DeclareLaunchArgument('check_config', default_value='false', description='只检查配置，不订阅传感器或启动建图'),
        DeclareLaunchArgument('args', default_value='', description='传给 Faster-LIO 的其余 ROS 参数，如 -p pcd_save.pcd_save_en:=false'),
        OpaqueFunction(function=setup),
    ])
