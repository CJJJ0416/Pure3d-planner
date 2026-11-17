Source: https://github.com/legubiao/quadruped_ros2_control/tree/humble
Base commit: a808acbd0bdb9d298d07de3c954010e3d57a15f6
Selected six A1/control packages, with EOLP RL migration and Classic launch fixes.
No SDK2, OCS2, MuJoCo or modern Gazebo packages are imported.
Destination integration adds configurable spawn pose and explicit dependency declarations.
Original Building scene and ROS2 odometry bridge are retained in sibling unitree_gazebo, with original licenses.
