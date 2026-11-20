# Real-world MID360/Avia bag replay

当前数据为 FAST-LIO2 官方 README 链接的 Avia 示例 `HKU_MB_2020-09-20-13-34-51.bag`，下载目录为 `datasets/FAST_LIO2/`。原 ROS1 文件、官方参数快照和下载来源均保留，ROS2 转换包位于 `ros2/HKU_MB_2020-09-20-13-34-51/`。

```bash
cd /home/cjh/Pure3D_ws
ros2 launch Entrance/launch/real/real_bag.launch.py action:=run 'args:=--profile fastlio_official --model tiny --rate 1 --rviz'
```

默认 ROS 域为 `186`、回放速度为 `1` 倍速。管线为现有 Faster-LIO → 局部稳定体素 → KNN → Tiny cost。官方 bag 的来源与运行的 LIO 算法分别记录；此入口运行当前工作区的 Faster-LIO。

## ROS1 到 ROS2

```bash
source /opt/ros/humble/setup.bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python common/bag_tools/convert_fastlio_official.py  --source datasets/FAST_LIO2/ros1/HKU_MB_2020-09-20-13-34-51.bag  --destination datasets/FAST_LIO2/ros2/HKU_MB_2020-09-20-13-34-51
```

转换器使用工作区 `.cache/bag_tools/` 的 rosbags；输出由系统 ROS2 Humble 的原生 SQLite3 writer 写入。数据话题保留为 `/livox/lidar` 和 `/livox/imu`，消息包名从 `livox_ros_driver` 转为 `livox_ros_driver2`，并按 ROS2 规范转换 Header。

官方原包的录制时间使用 Unix epoch，传感器 Header 使用启动后的秒数。转换只对所有录制时间减去同一个常数：完整 IMU 录制时间减 Header 时间的中位数。原始事件顺序、相对录制间隔、传感器时间戳、雷达 timebase、逐点 offset_time 和数值字段保持一致。`conversion.json` 记录平移常数、消息数量和数值抽查。

## 参数、外参与时钟

- `Entrance/config/real_bag.yaml`：局部体素、KNN 和 Tiny 权重，模型条件沿用 `[0.5,0.4,0.2,0]`。
- `Entrance/config/real/faster_lio.yaml`：真实传感器基础参数。
- `Entrance/config/real/fastlio_official_lio_overrides.yaml`：Avia 6 线、官方 IMU 噪声和量程参数；盲区按用户要求设为 `0.2 m`。
- `Entrance/config/real/fastlio_official_frames.yaml`：官方 `IMU ← LiDAR` 外参，平移 `[0.04165,0.02326,-0.0284] m`、单位旋转。

数据没有机器人安装标定，因此公共 `body` 表示 IMU。`world ← camera_init` 根据初始三秒加速度确定竖直方向，以初始 IMU 为原点、初始 IMU x 轴在水平面的投影为朝向。公共 TF 为 `world → map → camera_init → body`，其中 `map → camera_init` 为单位变换。

real-world 基础配置的 `use_sim_time=false` 用于在线硬件。bag 回放生成的实际参数设为 `true`，由唯一播放器的 `--clock 100` 驱动。Avia IMU 的原始加速度单位为 g，现有 Faster-LIO 在初始化后按重力常数归一化。

Faster-LIO 在雷达坐标系剔除 `x²+y²+z² ≤ 0.2²` 的点，再做配准。后续局部点云、KNN 和 cost 均使用经过盲区过滤的注册点云。

## 查看与产物

RViz 默认显示 cost、KNN 删除点和 `/faster_lio/path`，视角跟随 `body`；勾选 `Box LIO voxels` 查看过滤前局部点云。关闭 RViz 不会停止计算。`--rate` 是回放倍速，与 cost 刷新 Hz 分别统计。

输出位于 `outputs/real_world/fastlio_official/HKU_MB_2020-09-20-13-34-51/<时间>-tiny/`：

- `profile.json`、`preflight.json`、实际 YAML：来源、时钟、外参与运行配置。
- `local/`：现有 Entry 模型记录、推理历史、最终点云和 KNN 掩码。
- `snapshots/`：按时间戳严格配对的原始局部点云与实际 cost 输出。
- `audit.json`、`status_history.jsonl`、`cost_publications.jsonl`、`odometry.jsonl`：实际发布及运行状态。空点云表示清空，播放器预滚阶段的首次时钟复位单独记录。
- `run.json`：是否回放完成并排空最后扫描；动态改速另存 `rate_changes.json`。

保存的快照可独立查看：

```bash
ros2 launch Entrance/launch/real/real_bag.launch.py action:=view 'args:=--run '"'"'<输出目录>'"'"' --rviz --cycle-seconds 5'
ros2 launch Entrance/launch/real/real_bag.launch.py action:=view 'args:=--run '"'"'<输出目录>'"'"' --snapshot 2 --plot'
```

三栏依次为原始局部体素、KNN 保留点（紫色为删除点）、模型 cost。三栏仅平移展示坐标，保存坐标保持不变。

2026-11-19 用户要求删除原先下载的三个 JEPLO bag，现已删除；原实采实验记录保留于 `outputs/real_world/jeplo/`，其 LIO 诊断不作为当前官方 Avia bag 的验证结果。
