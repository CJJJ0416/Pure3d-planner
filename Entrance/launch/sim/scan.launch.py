"""SCAN 单项方案：接入已有 A1 仿真、cost 参考路径和样条控制。

Usage（工作区根目录，先 source /opt/ros/humble/setup.bash）：
  ros2 launch Entrance/launch/sim/scan.launch.py flight_type:=4
  ros2 launch Entrance/launch/sim/scan.launch.py --show-args
通常由 Entrance/launch/sim/gazebo.launch.py 自动组合，避免重复启动。
配置 YAML：Entrance/config/scan.yaml；话题映射：Entrance/config/local_planners.yaml。
"""
from pathlib import Path
from datetime import datetime
import os
import math
import importlib.util
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node
import yaml

# 本文件位于 Entrance/launch/sim，上溯两级才是 Entrance。
_entrance = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('pure3d_runtime', _entrance / 'launch/_internal/runtime.py')
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)


def setup(context):
    # 单独运行也加载本工作区环境，YAML 始终从 Entrance 的唯一源文件读取。
    env = _runtime.environment('sim')
    os.environ.update(env)
    context.environment.update(env)
    # flight_type 覆盖 SCAN 的 FSM 模式（1 人工、2 预设航点、3 参考路径、4 cost 引导）；backend 决定模式 4 的跟踪实现。
    use_sim_time = context.launch_configurations['use_sim_time'].lower() == 'true'
    mode = int(context.launch_configurations['flight_type'])
    backend = context.launch_configurations['cost_planner_backend']
    direct_spline = mode == 4 and backend == 'light'
    # 模式 4 的 A* 代价上限必须位于 (0, 1]，否则规划器无法给出有效参考。
    cost_limit = float(context.launch_configurations['cost_astar_limit'])
    if not math.isfinite(cost_limit) or not 0 < cost_limit <= 1:
        raise ValueError('cost_astar_limit must lie in (0, 1]')
    # auto：模式 1/4 保留人工 marker；也可用 true/false 强制开关。
    marker_option = context.launch_configurations['manual_markers'].lower()
    manual_markers = mode in (1, 4) if marker_option == 'auto' else marker_option == 'true'
    # auto：模式 4 用 0.1 m 的密集采样，其它模式用 0.8 m 的稀疏参考；也可显式给数值。
    spacing_option = context.launch_configurations['path_sample_distance']
    path_spacing = (.1 if mode == 4 else .8) if spacing_option == 'auto' else float(spacing_option)
    if path_spacing < 0:
        raise ValueError('path_sample_distance must be nonnegative')
    # 话题映射统一来自 local_planners.yaml；模式 4 的参考路径固定为 /pure_path。
    topics = yaml.safe_load((_entrance / 'config/local_planners.yaml').read_text())['topics']
    if mode == 4:
        topics['path'] = '/pure_path'
    if context.launch_configurations['path_topic']:
        topics['path'] = context.launch_configurations['path_topic']
    if mode == 4 and topics['path'] != '/pure_path':
        raise ValueError('Mode 4 requires path_topic:=/pure_path')
    # scan.yaml 是基础参数；params_file 可追加一份覆盖文件，后加载者优先。
    scan_config = str(_entrance / 'config/scan.yaml')
    extra = context.launch_configurations['params_file']
    parameters = [scan_config] + ([extra] if extra else [])
    # 所有 SCAN 节点把内部话题收拢到 /scan 命名空间，避免与外部规划器/工具重名。
    internal = [('body_pose', '/scan/body_pose'), ('sensor_pose', '/scan/sensor_pose'),
                ('fallback_goal', '/scan/fallback_goal'),
                ('planning/reference_accepted', '/scan/planning/reference_accepted'),
                ('planning/reference_rejected', '/scan/planning/reference_rejected'),
                ('planning/fallback_accepted', '/scan/planning/fallback_accepted'),
                ('cloud', '/scan/cloud'), ('move_base_simple/goal', '/scan/goal'),
                ('initial_path', '/scan/initial_path'), ('planning/bspline', '/scan/planning/bspline'),
                ('planning/scheduled_spline', '/scan/planning/scheduled_spline'),
                ('planning/spline_progress', '/scan/planning/spline_progress'),
                ('planning/data_display', '/scan/planning/data_display'),
                ('planning/go2_execution_frozen', '/scan/planning/go2_execution_frozen')]
    for name in ('goal_point', 'global_list', 'init_list', 'optimal_list', 'a_star_list', 'self_inflation',
                 'grid_map/occupancy', 'grid_map/occupancy_inflate', 'grid_map/sliding_map_bbox',
                 'grid_map/unknown', 'grid_map/depth_cloud', 'grid_map/sensor_pose_extrinsic'):
        internal.append((name, '/scan/' + name))
    map_parameters = {}
    tracking_parameters = {}
    # 直接样条跟踪需要把 grid_map 与跟踪器参数展开成节点参数；其它后端由 closed_loop_controller 自行读取。
    if direct_spline:
        for path in parameters:
            values = yaml.safe_load(Path(path).read_text()) or {}
            scan_values = values.get('scan_planner_node', {}).get('ros__parameters', {})
            map_parameters.update({key:value for key,value in scan_values.items() if key.startswith('grid_map.')})
            tracking_parameters.update(values.get('closed_loop_controller', {}).get('ros__parameters', {}))
            tracking_parameters.update(values.get('ego_local_trajectory_server', {}).get('ros__parameters', {}))
    # 模式 4 + light 使用样条服务器，其余使用闭环控制器；两者订阅同一组内部话题。
    trajectory_server = 'ego_local_trajectory_server' if direct_spline else 'closed_loop_controller'
    # 三个节点分工：bridge 把外部里程计/点云/路径接入 /scan，planner 做局部搜索，
    # 跟踪节点把参考路径转成 cmd_vel；manual_markers 与采样间距由上面的模式规则决定。
    scan_nodes = [
        Node(package='scan_planner', executable='scan_workspace_bridge', name='scan_workspace_bridge',
             output='screen', parameters=parameters + [{'use_sim_time': use_sim_time, 'manual_markers': manual_markers,
                                                       'path_sample_distance': path_spacing,
                                                       'bridge_reference_path': not direct_spline,
                                                       'commit_goal_on_release': mode == 4}],
             remappings=[('input_odom', topics['odom']), ('input_cloud', topics['cloud']),
                         ('input_path', topics['path']),
                         ('body_pose', '/scan/body_pose'), ('sensor_pose', '/scan/sensor_pose'),
                         ('cloud', '/scan/cloud'), ('goal', '/scan/goal'), ('initial_path', '/scan/initial_path')]),
        Node(package='scan_planner', executable='scan_planner_node', name='scan_planner_node',
             output='screen', parameters=parameters + [{'use_sim_time': use_sim_time, 'fsm.navi_mode': mode,
                 'interface.rolling_reference':mode==4 and backend=='light_scan',
                 'interface.allow_goal_fallback': mode != 4 and context.launch_configurations['goal_fallback'] == 'true'}],
             remappings=internal),
        Node(package='scan_planner', executable=trajectory_server, name=trajectory_server,
             output='screen', parameters=parameters + [map_parameters, tracking_parameters, {'use_sim_time': use_sim_time,
                 'adaptive_reference_speed': mode==4 and backend=='light_scan'}],
             remappings=internal + [('cmd_vel', topics['cmd_vel'])]),
    ]
    # 模式 4 额外启动 A* 跟踪：先确认工作区包含对应 Python 入口，再选择后端。
    if mode == 4:
        root = _runtime.ROOT
        if not (root / 'common/planning/local_astar_follow.py').is_file():
            raise ValueError('Mode 4 requires PURE3D_WS pointing to the Pure3D workspace')
        # 未指定输出目录时按时间戳在 outputs/entry_astar 下新建一次运行目录。
        output = context.launch_configurations['cost_astar_output'] or str(
            root / 'outputs/entry_astar' / datetime.now().strftime('mode4_%Y%m%d_%H%M%S_%f'))
        if direct_spline:
            # light 后端：移除 closed_loop_controller，改由 local_spline_follow 直接消费样条；
            # 地图分辨率、机身偏移、限速/限加速度均从同级 YAML 参数派生。
            scan_nodes.pop(1)
            planner_cmd=['common.planning.local_spline_follow','--use-sim-time',str(use_sim_time).lower(),
                         '--occupancy-resolution',str(map_parameters.get('grid_map.resolution', .05)),
                         '--body-offset',str(map_parameters.get('grid_map.double_cylinder_offset', .18)),
                         '--max-xy',str(tracking_parameters.get('max_vx', .75)),
                         '--body-reference-height',str(map_parameters.get('grid_map.body_height', .4)),
                         '--min-body-clearance',str(map_parameters.get('grid_map.obstacles_inflation_z_up', .1)),
                         '--max-acc',str(tracking_parameters.get('interface.max_tracking_acc', .5))]
        else:
            # light_scan/astar 后端：保留闭环控制器，从参数文件中取路径机身高度，
            # 并把 cost 上限作为 A* 跟踪的硬约束传入。
            body_reference_height=.4
            for path in parameters:
                values=yaml.safe_load(Path(path).read_text()) or {}
                bridge_values=values.get('scan_workspace_bridge',{}).get('ros__parameters',{})
                body_reference_height=bridge_values.get('path_body_height',body_reference_height)
            planner_cmd=['common.planning.local_astar_follow','--no-goal-fallback',
                         '--bridge-body-height',str(body_reference_height),
                         '--backend','light' if backend=='light_scan' else backend]
        # 跟踪进程作为独立模块运行；限制 BLAS 线程数，避免与 Gazebo/推理进程争抢 CPU。
        follow = ExecuteProcess(cmd=['/usr/bin/python3', '-m', *planner_cmd,
            '--output', output, '--cost-limit', str(cost_limit)], cwd=str(root), output='screen',
            additional_env={'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        return [follow, *scan_nodes]
    return scan_nodes


def generate_launch_description():
    # 参数分组：时钟与模式、人工 marker/路径采样、话题覆盖、参数覆盖、模式 4 的代价限制与后端。
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('flight_type', default_value='3', choices=['1', '2', '3', '4']),
        DeclareLaunchArgument('manual_markers', default_value='auto', choices=['auto', 'true', 'false']),
        DeclareLaunchArgument('path_sample_distance', default_value='auto'),
        DeclareLaunchArgument('path_topic', default_value=''),
        DeclareLaunchArgument('params_file', default_value=''),
        DeclareLaunchArgument('goal_fallback', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('cost_astar_output', default_value=''),
        DeclareLaunchArgument('cost_astar_limit', default_value='0.8'),
        DeclareLaunchArgument('cost_planner_backend', default_value='light_scan',choices=['light','light_scan','astar']),
        OpaqueFunction(function=setup),
    ])
