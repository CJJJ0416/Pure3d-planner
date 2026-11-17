"""Pure3D terrain-generator scene with the existing A1/MID360/SCAN stack."""
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def setup(context):
    args = context.launch_configurations
    terrain = Path(args['terrain_dir']).expanduser().resolve()
    world = terrain / 'terrain.world'
    spawn = json.loads((terrain / 'spawn.json').read_text())
    if not world.is_file():
        raise FileNotFoundError(f'Run common/unitree_gazebo/sim/prepare_world.py first: {world}')
    options = {key: args[key] for key in ('gui', 'rviz', 'planner', 'flight_type', 'device',
               'mid360_samples', 'mid360_downsample', 'trace', 'debug', 'cost_astar_output', 'cost_astar_limit')}
    options.update(world=str(world), mid360='true', odom_x='0.0', odom_y='0.0', odom_z='0.0')
    for key in ('x', 'y', 'height', 'yaw'):
        options[key] = args[key] or str(spawn[key])
    return [
        LogInfo(msg=f'Terrain: {world}; landing: {spawn["landing"]}; Gazebo truth=world; map=camera_init'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(
            get_package_share_directory('unitree_gazebo'), 'launch/gazebo.launch.py'))),
            launch_arguments=options.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('terrain_dir', description='Prepared directory containing terrain.world and spawn.json'),
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('planner', default_value='scan', choices=['scan', 'None']),
        DeclareLaunchArgument('flight_type', default_value='1', choices=['1', '2', '3', '4']),
        DeclareLaunchArgument('cost_astar_output', default_value=''),
        DeclareLaunchArgument('cost_astar_limit', default_value='0.8'),
        DeclareLaunchArgument('device', default_value='cpu', choices=['cpu', 'cuda']),
        DeclareLaunchArgument('mid360_samples', default_value='10000'),
        DeclareLaunchArgument('mid360_downsample', default_value='1'),
        DeclareLaunchArgument('trace', default_value='false'),
        DeclareLaunchArgument('debug', default_value='false'),
        *[DeclareLaunchArgument(key, default_value='', description='Empty uses the checked landing pose')
          for key in ('x', 'y', 'height', 'yaw')],
        OpaqueFunction(function=setup),
    ])
