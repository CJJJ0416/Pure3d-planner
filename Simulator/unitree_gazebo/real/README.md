# Real：MID360 / Faster-LIO

真机入口与仿真分开。`env.sh` 只加载 ROS、Faster-LIO underlay 和可选的硬件消息 overlay，使用真实时钟；不会启动 Gazebo、硬件驱动或机器人控制。先启动自己的 MID360 驱动，再手动运行建图。

完整安装标定、分终端命令、重力对齐及 在线 cost/SCAN 接入项见 [真机安装与启动文档](../../../docs/REAL_WORLD_SETUP.md)。

## 配置

- `config/faster_lio.yaml`：`use_sim_time: false`，输入话题默认 `/livox/lidar`（Livox CustomMsg）和 `/livox/imu`。按实际驱动调整 `common` 话题、预处理和估计参数。
- `config/frames.yaml`：安装外参、坐标系与初始地图位姿。现有数字来自仿真样例，需填写实测标定值。

安装外参的平移单位为米，角度单位为弧度。`rpy` 是固定轴 roll/pitch/yaw，旋转为 `Rz(yaw) Ry(pitch) Rx(roll)`；每段表示 `p_parent = R × p_child + translation`。

```yaml
mount:
  source: parameters
  body_to_lidar:
    translation: [0.2, 0.0, 0.08]   # 替换为实际安装平移
    rpy: [0.0, 0.785, 0.0]         # 替换为实际 roll/pitch/yaw
  lidar_to_imu:
    translation: [-0.011, -0.02329, 0.04412]
    rpy: [0.0, 0.0, 0.0]
```

机身到 IMU 的外参为 `T_body_imu = T_body_lidar × T_lidar_imu`；发布机身位姿时采用 `T_camera_body = T_camera_imu × inverse(T_body_imu)`。MID360 雷达和内置 IMU 一起倾斜，因此安装角只转换一次。启动时从 `lidar_to_imu` 的逆变换生成 Faster-LIO 所需的 IMU←LiDAR 外参。

`anchor.source: initial_body` 将配置中的 `anchor.translation/rpy` 作为初始机身在 `map` 中的位姿，自动生成静态 `map → camera_init`。若已知该静态变换，可改为 `fixed` 直接提供。两种方式均无需 Gazebo 真值；真机入口拒绝 `gazebo` 锚定。外参也可来自真实机器人的 URDF/TF，此时需确保 `body_reference → imu` 由实际机器人描述发布；当前真机配置默认采用参数。

## 手动运行

在 `/home/cjh/Pure3D_ws` 下执行：

```bash
source /opt/ros/humble/setup.bash
# 仅检查配置
ros2 launch Entrance/launch/real/faster_lio_real.launch.py check_config:=true

# 驱动已启动且机器人静止后，手动建图
ros2 launch Entrance/launch/real/faster_lio_real.launch.py

# 另一个终端查看
ros2 launch Entrance/launch/real/rviz_real.launch.py
```

该入口加载现有 `Entrance/config/scan.rviz`，其中已直接加入 Faster-LIO 上游 ROS 2 的累计扫描（120 秒）与当前帧 PointCloud2 显示项；使用真实时钟，沿用已运行的建图进程。

若使用另一份标定文件：

```bash
source /opt/ros/humble/setup.bash
PURE3D_LIO_FRAMES_CONFIG=/absolute/path/frames.yaml  ros2 launch Entrance/launch/real/faster_lio_real.launch.py
```

`PURE3D_REAL_LIO_CONFIG` 可覆盖真机 LIO 参数文件；`PURE3D_REAL_UNDERLAY` 可指定驱动 ROS 工作区以加载其消息环境。确认 IMU 加速度单位：静止模长为 9.81 时使用默认 `PURE3D_IMU_GRAVITY_NORM=9.81`；驱动以 g 为单位时设为 `1`。就绪检查使用实际配置的 IMU 话题，不假设仿真时间戳。

默认结果位于 `outputs/unitree_real/lio/`，ROS 日志位于 `outputs/unitree_real/ros_log/`；可用 `PURE3D_LIO_OUTPUT` 覆盖运行目录。公共输出为 `/faster_lio/Odometry`、`/faster_lio/cloud_registered` 和 `/faster_lio/path`，RViz Fixed Frame 为 `map`。

坐标转换接口已有独立 ROS 测试，真机采集和建图效果尚未验证。
