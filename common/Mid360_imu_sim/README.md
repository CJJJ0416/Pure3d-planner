# MID360 + IMU simulation for ROS 2 Humble / Gazebo Classic 11

Active A1 integration is in a1_description/xacro/mid360.xacro, called from robot.xacro. The plugin has been ported from ROS 1 to gazebo_ros::Node and native ROS 2 publishers. It retains the 800000-row CSV angular pattern and uses isolated ODE ray collision spaces.

## Run with A1

```bash
cd /home/cjh/EOLP_ws
./build_a1_sru.sh
./run_a1_sru.sh gui:=false rviz:=false
```

The EOLP A1 launch enables MID360 by default and retains the SRU front depth camera. Disable LiDAR and its IMU with mid360:=false. Default 24000 rays/frame at 10 Hz; mid360_downsample:=4 uses 6000 rays/frame while advancing through the same pattern. mid360_samples controls the pattern stride per scan. The A1 model and simulator implementation are reused from glocal-nav2 without changing their geometry or sensor code.

| Topic | Type | Frame |
| --- | --- | --- |
| /scan | sensor_msgs/msg/PointCloud2 | laser_livox |
| /livox/lidar | sensor_msgs/msg/PointCloud2, same frame/data as /scan | laser_livox |
| /livox/lidar2 | livox_ros_driver2/msg/CustomMsg | laser_livox |
| /livox/imu | sensor_msgs/msg/Imu, configured 1000 Hz | livox_imu_link |

/scan deliberately changes from the old ROS1 sensor_msgs/PointCloud to ROS2 PointCloud2. Invalid/no-return rays are omitted rather than emitted as false points at the sensor origin. PointCloud2 includes x/y/z/intensity, time (seconds), offset_time (ns), line/tag/reflectivity. The native C++ plugin publishes CustomMsg directly, so no converter is required in the A1 launch.

## Geometry and timing

Mounts exactly match the old glocal-nav A1: base→laser_livox xyz=(0.2,0,0.08), rpy=(0,0.785,0); laser_livox→livox_imu_link xyz=(-0.011,-0.02329,0.04412), rpy=0. Mesh copied from the original A1. Laser payload mass is modeled as 0.265 kg with a simple box collision. These inertia/collision parameters are simulation approximations, not a calibrated hardware model.

Each cloud is an instantaneous ODE snapshot. Physics is locked only for ray measurement; ROS packing/publication happens after releasing it. All per-point time/offset_time values are zero, and CustomMsg.timebase equals the simulation snapshot timestamp. We do not fabricate wall-clock point times. The CSV first column is a sequence index despite its Time/s heading. Laser channel metadata is not available from this CSV; line is 0. Reflectivity is Gazebo collision laser_retro, not a physical optical model. Noise and rolling-scan motion distortion are not reproduced.

The MID360 IMU uses the native ROS2 gazebo_plugins plugin with a unique sensor/node name, explicit frame_name and ~/out remapping. Existing body IMU/ros2_control interfaces remain separate. Fixed joints are retained in SDF, and robot_state_publisher owns their TF.

## Optional ROS2 converter

pointcloud2livox.py is ported to rclpy and reads PointCloud2. Source frame, timestamp and available offset/channel fields are preserved. To avoid a second publisher on the native CustomMsg topic, its default output is /livox/lidar2_converted:

```bash
ros2 run livox_laser_simulation pointcloud2livox.py
```

Override input_topic/output_topic with ROS2 parameters as needed. Do not run it on /livox/lidar2 simultaneously with the native publisher.

## Message interfaces

Sibling src/livox_ros_driver2 supplies only the official CustomMsg/CustomPoint ROS2 interfaces, not the SDK/hardware driver. See its README before installing the real driver: do not keep two packages with the same name. Definitions: https://github.com/Livox-SDK/livox_ros_driver2/tree/master/msg .

Historical ROS1 standalone launch files and other Livox-model xacros are retained for reference; the verified ROS2 entry for this migration is the A1 launch above. FAST-LIO/Faster-LIO end-to-end SLAM is a separate integration step.

Validation and backup location: /home/cjh/glocal-nav2/docs/mid360_validation.md .
