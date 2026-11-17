# 真机 MID360、Faster-LIO 与局部 cost 接入

核对日期：2026-11-17。适用环境为 ROS 2 Humble、MID360 和当前 Pure3D 工作区。

现有真机入口可以独立启动 Faster-LIO、安装外参转换和 RViz。Entry 局部 cost 与 SCAN 的在线入口仍绑定仿真时钟、`world` 和部分 Gazebo 接口，迁移项见下文。已有验证覆盖配置生成、坐标转换与仿真运行，实际安装标定、真机建图和运动控制需在设备上验收。

## 1. 安装位置与坐标含义

采用 `T_A_B` 表示将 B 中的点转换到 A：

```text
p_A = R_A_B p_B + t_A_B

B：物理机身参考点，例如底盘中心 base
L：雷达测量坐标系
I：雷达内置 IMU 坐标系
C：Faster-LIO 的 camera_init 初始化坐标系
```

平移单位为米，角度为弧度。配置中的固定轴 roll、pitch、yaw 使用 `R = Rz(yaw) Ry(pitch) Rx(roll)`。`camera_init` 是 LIO 初始化帧名。

需要确定两段外参：

| 配置 | 物理含义 | 获取方式 |
|---|---|---|
| `mount.body_to_lidar` | `T_B_L`，雷达在机身中的完整安装位姿 | 机械尺寸、实测或已校准 URDF |
| `mount.lidar_to_imu` | `T_L_I`，内置 IMU 在雷达中的位姿 | 对应型号的标定参数 |

当前 Faster-LIO 原始位姿是 `T_C_I`。机身位姿按以下公式计算：

```text
T_B_I = T_B_L T_L_I
T_C_B = T_C_I inverse(T_B_I)
inverse(R, t) = (Rᵀ, −Rᵀt)
```

因此要同时抵消安装平移和旋转。直接对欧拉角取负或只减去安装高度不等价于刚体逆变换。只有 `camera_init` 和原始 `body` TF，无法唯一反推出物理底盘的安装外参，仍需测量或独立标定。

当前共享桥已经完成上述转换，公共 `/faster_lio/Odometry` 的 `child_frame_id=body` 表示经过安装修正的机身估计。原始 IMU 位姿保留在 `/faster_lio/raw_odometry`。下游使用公共 `body` 时不再应用同一套安装修正。接入其他 LIO 时先查明其 `body` 究竟是 IMU、雷达还是已校准机身。

若物理 `base` 与公共 `body` 定义完全重合，可由唯一发布者提供单位 `body → base`。已有机器人定位或 URDF 链路时，应先检查父节点，避免给同一 child 再发布另一条边。

## 2. 重力与地图方向

默认真机帧链为：

```text
map → camera_init → body
```

`anchor.source: initial_body` 把首帧机身放到配置指定的 `map` 位姿，然后固定 `map → camera_init`。默认初始机身 RPY 为零，因此应在机身水平、静止时初始化。若在斜坡上初始化，强制机身 RPY 为零会把坡面倾斜吸收到地图方向中。

地形预测要求地图 Z 轴与重力反向一致。在非水平起步条件下，需要利用 IMU/LIO 重力估计确定地图 roll、pitch，再选择初始 yaw。静止重力不能确定绝对 yaw。安装标定和重力对齐分别检查，TF 连通并不保证点云竖直方向正确。

配置中加入 `frames.world` 时，共享桥会使用 `world → map → camera_init → body`，把初始对齐放在第一条边，`map → camera_init` 为单位变换。真机不使用 Gazebo 真值锚定。

## 3. 软件与设备准备

| 组件 | 当前路径或要求 |
|---|---|
| ROS | `/opt/ros/humble` |
| Faster-LIO | 默认 `/home/cjh/livox_ws/install/faster_lio`，可用 `PURE3D_LIO_WS` 修改工作区 |
| 完整 Livox 驱动源码 | 本机 `/home/cjh/livox_ws/src/livox_ros_driver2` |
| 本仓库 Livox 包 | `common/livox_ros_driver2` 只提供消息接口，不能代替硬件驱动 |
| GPU 推理环境 | 见 [PTv3 环境](PTV3_CONDA_ENV.md)，ROS 与 GPU worker 分进程运行 |

安装 Livox-SDK2 和完整 ROS 2 驱动，具体依赖见本机驱动源码的 README。该版本的构建方式为 `./build.sh humble`。其脚本会清理所在工作区的 `build`、`install`、`devel`，应在独立驱动工作区构建，避免清理现有 Faster-LIO 安装。一个运行环境中只加载一套名为 `livox_ros_driver2` 的消息/驱动包。

MID360 网口 IP、主机 IP 和实际设备地址写入驱动使用的 `MID360_config.json`。样例 IP 需要按设备修改。驱动安装外参保持与下游约定一致，避免 SDK 已旋转点云后，又在 TF 中重复施加安装角。

本机 `launch_ROS2/msg_MID360_launch.py` 固定选择相邻 `config/MID360_config.json`，默认 `xfer_format=1`、10 Hz、`frame_id=livox_frame`。该文件没有声明 `user_config_path` 启动参数，修改实际安装配置后用 `ros2 param get /livox_lidar_publisher user_config_path` 核实选中的文件。`livox_frame` 与机器人 URDF 中的雷达帧应表示相同原点和轴向。

## 4. 配置实测安装外参

在项目根目录复制独立配置，再填入实测值：

```bash
cd /home/cjh/Pure3D_ws
cp common/unitree_gazebo/real/config/frames.yaml \
   common/unitree_gazebo/real/config/frames.robot.yaml
cp common/unitree_gazebo/real/config/faster_lio.yaml \
   common/unitree_gazebo/real/config/faster_lio.robot.yaml
```

`frames.robot.yaml` 的结构如下。数字为格式示例，使用前替换为实测参数：

```yaml
frames:
  map: map
  camera: camera_init
  body: body
  body_reference: base
  imu: livox_imu_link
mount:
  source: parameters
  body_to_lidar:
    translation: [0.2, 0.0, 0.08]
    rpy: [0.0, 0.785, 0.0]
  lidar_to_imu:
    translation: [-0.011, -0.02329, 0.04412]
    rpy: [0.0, 0.0, 0.0]
anchor:
  source: initial_body
  translation: [0.0, 0.0, 0.0]
  rpy: [0.0, 0.0, 0.0]
```

真机样例的雷达高度为 `0.08 m`，当前仿真为 `0.18 m`，两者均不能替代实测。采用 `mount.source: parameters` 时不依赖 Gazebo 的 `base → imu` TF。采用 `urdf` 时必须由真实机器人描述提供该静态关系。

`faster_lio.robot.yaml` 保持 `use_sim_time: false`，根据实际驱动确认以下输入：

| 参数 | 当前真机默认值 | 核对内容 |
|---|---|---|
| `common.lid_topic` | `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` |
| `common.imu_topic` | `/livox/imu` | `sensor_msgs/msg/Imu` |
| `preprocess.lidar_type` | `1` | 本版本 Livox CustomMsg 分支，日志历史名称为 AVIA |
| `preprocess.scan_line` | `4` | MID360 硬件 line 通常为 0–3 |
| `preprocess.blind` | `0.3 m` | 按真实盲区与近身点质量核对 |
| `preprocess.time_scale` | `0.001` | 与当前驱动 offset_time 单位、预处理实现配套核对 |
| `mapping.extrinsic_est_en` | `false` | 采用配置标定 |
| `pcd_save.pcd_save_en` | `false` | 当前关闭自动 PCD 保存 |

共享启动器从 `lidar_to_imu` 的逆变换生成 Faster-LIO 的 `mapping.extrinsic_T/R`。最终运行参数保存在 `faster_lio.actual.yaml`，以该文件为核对依据。

## 5. 分终端启动

以下命令仅启动驱动、建图与显示。机器人控制器使用对应硬件的已有入口。

每个终端使用相同 ROS 域和网络设置，例如 `ROS_DOMAIN_ID=42`。多机通信时使用 `ROS_LOCALHOST_ONLY=0` 并检查 DDS 网络。本例使用真实时钟，不启动 `/clock`。

终端 1，启动已经编译的真实驱动：

```bash
cd /home/cjh/Pure3D_ws
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
# 独立驱动工作区存在时，先按实际路径设置 PURE3D_REAL_UNDERLAY。
source common/unitree_gazebo/real/env.sh
ros2 pkg prefix livox_ros_driver2
ros2 pkg executables livox_ros_driver2
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

`pkg executables` 应能列出 `livox_ros_driver2_node`。如果只找到消息包，先完成完整驱动安装，并让所有终端加载同一个 `PURE3D_REAL_UNDERLAY`。

终端 2，检查配置，再在机器人静止时启动 LIO：

```bash
cd /home/cjh/Pure3D_ws
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
export PURE3D_LIO_FRAMES_CONFIG="$PWD/common/unitree_gazebo/real/config/frames.robot.yaml"
export PURE3D_REAL_LIO_CONFIG="$PWD/common/unitree_gazebo/real/config/faster_lio.robot.yaml"
export PURE3D_LIO_OUTPUT="$PWD/outputs/unitree_real/lio_$(date +%Y%m%d_%H%M%S)"
common/unitree_gazebo/real/run_faster_lio.sh --check-config
common/unitree_gazebo/real/run_faster_lio.sh
```

`--check-config` 需要已编译 Faster-LIO，只生成参数，不启动建图或 TF 桥。正常入口自动启动唯一的 TF 桥，并将原始里程计、原始 TF 分别重映射到 `/faster_lio/raw_odometry`、`/faster_lio/raw_tf`。Ctrl+C 同时结束该入口管理的建图和桥接进程。

真机默认等待 IMU 静止，最长 120 秒墙钟时间。先确认驱动加速度单位。`PURE3D_IMU_GRAVITY_NORM=1` 只改变启动前的静止判据，不会把网络上的 g 单位消息转换为 m/s²。LIO 输入仍须符合其 IMU 单位要求。可用 `PURE3D_LIO_WAIT_TIMEOUT` 调整等待时间，`0` 表示持续等待。

终端 3，显示点云和 TF：

```bash
cd /home/cjh/Pure3D_ws
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
common/unitree_gazebo/real/view_faster_lio.sh
```

该入口使用共享 SCAN RViz 配置。配置当前保存的 Fixed Frame 为 `world`，真机默认只有 `map` 链时，在 RViz 的 Global Options 中将 Fixed Frame 改为 `map`。启用 Faster-LIO 点云显示，确认 TF 状态正常。

终端 4，只读检查接口：

```bash
cd /home/cjh/Pure3D_ws
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
source common/unitree_gazebo/real/env.sh
ros2 topic type /livox/lidar
ros2 topic type /livox/imu
ros2 topic hz /faster_lio/cloud_registered
ros2 topic echo /faster_lio/Odometry --once
ros2 run tf2_ros tf2_echo map camera_init
# 上一条 Ctrl+C 后检查下一段。
ros2 run tf2_ros tf2_echo camera_init body
```

公共里程计和注册点云的父帧为 `camera_init`。RViz 通过 `map → camera_init` 显示点云。单独调试 `real/connect_map_tf.sh` 时，输入必须是原始 IMU 里程计 `/faster_lio/raw_odometry`，并先退出已有桥接进程。只有 TF、没有对应里程计消息的其他 LIO 接口，需要先适配桥接输入。

## 6. Entry 与 SCAN 的迁移项

当前 `total_terminal.sh` 包含 Gazebo、仿真 keyboard 和模式切换，适用于仿真实验。真机使用上面的独立入口，并完成以下接口修改后再接入在线导航：

| 模块 | 当前绑定 | 真机处理 |
|---|---|---|
| Entry 配置 | `Entry/configs/local.yaml` 使用仿真时钟 | 改为真实时钟，统一所有节点的时间源 |
| Entry 地图帧校验 | `Entry/utils/local_map.py` 限定 `world ← camera_init` | 参数化为重力对齐地图帧，或明确建立真实 world 链 |
| Entry 局部盒子 | LIO 公共机身位姿，世界轴向 `6×6×5 m` | 保留独立维护，确认公共 body 是物理机身参考 |
| SCAN odom | `workspace_topics.yaml` 的 `/Odometry_gazebo` | 接入经过安装修正的真实机身状态，统一 frame 与速度定义 |
| SCAN 点云 | 仿真 `/livox/lidar` 为 PointCloud2 | 真实驱动同名话题为 CustomMsg，需要 PointCloud2 输出或转换适配 |
| SCAN sensor_pose | 仿真 `laser_livox` TF | 使用真实测量原点与安装标定，不能用 base 冒充雷达原点 |
| A* 跟随与监测 | `local_astar_follow.py`、`monitor_local.py` 含 world、真值或仿真时间逻辑 | 移除 Gazebo 依赖，接入同一真实里程计与时钟 |
| 控制输入 | 仿真控制器模式与速度接口 | 接入真实 A1 控制接口，单独验收跟踪与停车 |

话题重映射只能改名字，不能把 CustomMsg 转为 PointCloud2。使用驱动 PointCloud2 输出或转换器时保留正确时间戳、原点和坐标轴。若使用 LIO 注册点云作为 SCAN 输入，也需保持实际传感器原点和 ray casting 的一致性。

本机 Faster-LIO `PublishOdometry` 当前填充 pose 和 covariance，没有填充有效 twist。共享桥修正 pose/covariance，但没有补齐机身速度。接 SCAN 前需由估计器或适配器提供速度。对有杆臂的 IMU 原点，机身线速度还涉及 `v_GB = v_GI − ω_G × (R_GB t_BI)`，不能直接把 IMU 原点速度当作底盘中心速度。

SCAN 感知盒沿世界轴，中心随雷达 sensor_pose 移动。Entry 盒子沿世界轴，中心随校准后的 LIO body 移动。两者中心相差安装杆臂和盒子量化。SCAN 占用显示还会按传感器高度裁剪，RViz 中未显示上方点不等价于占用图未保存。

## 7. 当前局部 cost 参数

`total_terminal.sh` 和 `run_cost.sh` 当前默认选择 Tiny best49。直接读取 `local.yaml` 而不指定模型时，其 `model` 仍为 Small best56。运行模型和校验值以会话内 `cost/local/model.json` 为准。

| 项目 | 当前值 |
|---|---|
| 条件顺序 | `[clearance_min, slope_max, step_max, stand_ratio]` |
| 条件 | `[0.5, 0.4, 0.2, 0.0]`，step_max 已改为 `0.2 m` |
| 局部盒子 | `6×6×5 m`，LIO 位姿独立维护 |
| 体素与容量 | `0.05 m`，最多 500,000 点 |
| 编码 | `4×4×3 m core + 1 m context`，保留完整已观测上下文 |
| KNN | 编码前启用，k=16、alpha=2、半径 0.15 m、至少 3 个其他邻居 |
| 删除规则 | 平均 KNN 距离超出全局 mean+2std，且半径内邻居少于 3，两项同时成立 |
| 调度 | 每 0.25 s 检查最新输入，合并积压，实际频率由处理速度决定 |
| 输入 / 结果过期 | 1.0 s / 2.0 s，位姿过期 0.5 s |
| RViz | 0.1 m 显示降采样，最多 30,000 点，最高 1 Hz |
| 模式 4 A* | 严格 `cost < 0.8`，生成 `/pure_path`，SCAN 保留原局部优化 |
| SCAN 重规划预算 | 0.5 s，共享墙钟预算、超时停车，采用循环与阶段边界协作检查 |

条件是模型输入，需与实机能力分别标定。当前 `stand_ratio=0` 低于 Tiny checkpoint 记录的训练范围下限 `0.02`。KNN 对稀疏楼梯边缘的保留率需随真实点密度重新检查。没有观测点的区域保持未知，不通过零 cost 补点。

## 8. 检查与结果记录

首次接入按顺序检查静止重力、安装轴向、点云地面方向、机身位姿、点云与机身对齐，然后再检查移动时的漂移、局部 cost 和控制跟踪。保存实测安装配置、驱动配置、`faster_lio.actual.yaml`、`map_anchor.json`、模型 SHA256 和阶段耗时。

| 现象 | 优先检查 |
|---|---|
| 雷达旋转或点云整体倾斜 | 外参方向、SDK 重复旋转、初始化重力方向 |
| 机身与雷达间隔错误 | IMU 原点与机身原点混淆、安装修正重复应用 |
| 一直等待静止 IMU | 实际话题、单位、时间戳推进、静止程度 |
| 无点云或消息类型不匹配 | 消息包 overlay、CustomMsg 与 PointCloud2 接口 |
| RViz 报 world 不存在 | 使用默认真机帧链时将 Fixed Frame 设为 map |
| 盒子或预测停止更新 | LIO odom 时间、TF、输入过期原因、真实时钟配置 |
| 楼梯点被删除 | KNN 邻居统计、距离与局部点密度，保存过滤前后点集对照 |

录包时保留原始 LiDAR、IMU、TF、实际机器人状态及标定配置。真实时钟运行不需要仿真 `/clock`。当前 `run_faster_lio.sh --bag` 是仿真话题契约，真机回放需按真实驱动类型和时间源另行适配。

## 9. 对应源码

- [真机入口与配置](../common/unitree_gazebo/real/README.md)
- [安装外参与变换公式实现](../common/unitree_gazebo/scripts/frame_config.py)
- [原始 IMU 位姿到公共机身的桥接](../common/unitree_gazebo/scripts/map_anchor.py)
- [共享 LIO 启动器与输入检查](../common/unitree_gazebo/scripts/run_faster_lio.sh)
- [Entry 局部 cost 配置](../Entry/configs/local.yaml)
- [Entry 局部地图与配置校验](../Entry/utils/local_map.py)
- [SCAN 接口配置](../common/SCAN-Planner/src/planner/plan_manage/config/workspace_topics.yaml)
- [仿真局部 cost 操作说明](../Entry/docs/LOCAL_COST.md)
