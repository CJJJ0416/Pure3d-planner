"""SCAN adapter for the existing glocal-nav2 A1 simulator; no second simulator."""
from pathlib import Path
from datetime import datetime
import os
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node
import yaml


def setup(context):
    use_sim_time = context.launch_configurations['use_sim_time'].lower() == 'true'
    mode = int(context.launch_configurations['flight_type'])
    cost_limit = float(context.launch_configurations['cost_astar_limit'])
    if not math.isfinite(cost_limit) or not 0 < cost_limit <= 1:
        raise ValueError('cost_astar_limit must lie in (0, 1]')
    marker_option = context.launch_configurations['manual_markers'].lower()
    manual_markers = mode in (1, 4) if marker_option == 'auto' else marker_option == 'true'
    spacing_option = context.launch_configurations['path_sample_distance']
    path_spacing = (.1 if mode == 4 else .8) if spacing_option == 'auto' else float(spacing_option)
    if path_spacing < 0:
        raise ValueError('path_sample_distance must be nonnegative')
    topics = yaml.safe_load(Path(get_package_share_directory('scan_planner'), 'config/workspace_topics.yaml').read_text())['topics']
    if mode == 4:
        topics['path'] = '/pure_path'
    if context.launch_configurations['path_topic']:
        topics['path'] = context.launch_configurations['path_topic']
    if mode == 4 and topics['path'] != '/pure_path':
        raise ValueError('Mode 4 requires path_topic:=/pure_path')
    scan_share = get_package_share_directory('scan_planner')
    scan_config = str(Path(scan_share, 'config/a1.yaml'))
    extra = context.launch_configurations['params_file']
    parameters = [scan_config] + ([extra] if extra else [])
    internal = [('body_pose', '/scan/body_pose'), ('sensor_pose', '/scan/sensor_pose'),
                ('fallback_goal', '/scan/fallback_goal'),
                ('planning/reference_accepted', '/scan/planning/reference_accepted'),
                ('planning/fallback_accepted', '/scan/planning/fallback_accepted'),
                ('cloud', '/scan/cloud'), ('move_base_simple/goal', '/scan/goal'),
                ('initial_path', '/scan/initial_path'), ('planning/bspline', '/scan/planning/bspline'),
                ('planning/data_display', '/scan/planning/data_display'),
                ('planning/go2_execution_frozen', '/scan/planning/go2_execution_frozen')]
    for name in ('goal_point', 'global_list', 'init_list', 'optimal_list', 'a_star_list', 'self_inflation',
                 'grid_map/occupancy', 'grid_map/occupancy_inflate', 'grid_map/sliding_map_bbox',
                 'grid_map/unknown', 'grid_map/depth_cloud', 'grid_map/sensor_pose_extrinsic'):
        internal.append((name, '/scan/' + name))
    scan_nodes = [
        Node(package='scan_planner', executable='scan_workspace_bridge', name='scan_workspace_bridge',
             output='screen', parameters=parameters + [{'use_sim_time': use_sim_time, 'manual_markers': manual_markers,
                                                       'path_sample_distance': path_spacing}],
             remappings=[('input_odom', topics['odom']), ('input_cloud', topics['cloud']),
                         ('input_path', topics['path']),
                         ('body_pose', '/scan/body_pose'), ('sensor_pose', '/scan/sensor_pose'),
                         ('cloud', '/scan/cloud'), ('goal', '/scan/goal'), ('initial_path', '/scan/initial_path')]),
        Node(package='scan_planner', executable='scan_planner_node', name='scan_planner_node',
             output='screen', parameters=parameters + [{'use_sim_time': use_sim_time, 'fsm.navi_mode': mode,
                 'interface.allow_goal_fallback': mode != 4 and context.launch_configurations['goal_fallback'] == 'true'}],
             remappings=internal),
        Node(package='scan_planner', executable='closed_loop_controller', name='closed_loop_controller',
             output='screen', parameters=parameters + [{'use_sim_time': use_sim_time}],
             remappings=internal + [('cmd_vel', topics['cmd_vel'])]),
    ]
    if mode == 4:
        root = Path(os.environ.get('PURE3D_WS', Path(__file__).resolve().parents[6])).resolve()
        if not (root / 'Entry/planning/local_astar_follow.py').is_file():
            raise ValueError('Mode 4 requires PURE3D_WS pointing to the Pure3D workspace')
        output = context.launch_configurations['cost_astar_output'] or str(
            root / 'outputs/entry_astar' / datetime.now().strftime('mode4_%Y%m%d_%H%M%S_%f'))
        follow = ExecuteProcess(cmd=['/usr/bin/python3', '-m', 'Entry.planning.local_astar_follow',
            '--output', output, '--cost-limit', str(cost_limit), '--no-goal-fallback'], cwd=str(root), output='screen',
            additional_env={'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')})
        return [follow, *scan_nodes]
    return scan_nodes


def generate_launch_description():
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
        OpaqueFunction(function=setup),
    ])
