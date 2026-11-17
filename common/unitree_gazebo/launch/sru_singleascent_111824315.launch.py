"""Generated building with the existing A1 controller and Gazebo launch."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    package = Path(get_package_share_directory('unitree_gazebo'))
    defaults = {'x': '-1.0215162940345284', 'y': '-7.237344959137292', 'height': '0.5', 'yaw': '-1.4871114903730263', 'gui': 'true', 'rviz': 'false', 'planner': 'None', 'device': 'cuda', 'mid360': 'true'}
    args = {key: LaunchConfiguration(key) for key in defaults}
    args.update(world=str(package/'worlds'/'sru_singleascent_111824315.world'), odom_x='0', odom_y='0', odom_z='0')
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k,v in defaults.items()],
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(package/'launch'/'gazebo.launch.py')),
                                 launch_arguments=args.items())])
