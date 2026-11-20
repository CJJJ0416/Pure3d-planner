# Sim：Building-no-rails / A1 / MID360 / SCAN / Faster-LIO

在 Pure3D 工作区使用 ROS 2 Humble、Gazebo Classic 11。构建使用本工作区的机器人、控制器、MID360 插件与 SCAN 源码；Faster-LIO 复用本机 `/home/cjh/livox_ws/install/faster_lio`。不需要启动真实 Livox 硬件驱动。

## 构建和运行

启动入口集中于 [Entrance](../../../Entrance/README.md)，单项使用 launch，整套方案使用 shell。Faster-LIO 仍可独立启动。

在 `/home/cjh/Pure3D_ws` 执行：

```bash
source /opt/ros/humble/setup.bash
./Entrance/shell/build.sh
ros2 launch Entrance/launch/sim/gazebo.launch.py
```

默认打开 Gazebo 和 SCAN RViz；无窗口运行用 `ros2 launch Entrance/launch/sim/gazebo.launch.py gui:=false rviz:=false`。`planner:=None` 只启动仿真。默认 CPU 推理，`device:=cuda` 可显式切换。

默认场景使用 `/home/cjh/EOLP_ws/src/sru_a1_bringup/models/Building-no-rails` 及其所在包的 `worlds/Building-no-rails.world`。模型路径优先于 Pure3D 包内的同名模型，不加载 terrain 场景。出生点沿用 EOLP 当前预设 `[-5.5, 6.0, 0.5] m`、yaw `0`，可用 `x:=... y:=... height:=... yaw:=...` 覆盖。`./Entrance/shell/total_terminal.sh` 同样使用此场景。

没有该 EOLP 目录的机器默认使用本包内的同源模型副本，DAE 校验值记录在 `models/Building-no-rails/README.md`。也可用 `PURE3D_BUILDING_MODEL` 显式指定模型目录。

单项 Gazebo launch 不启动 Faster-LIO。机器人控制和建图由下述独立 launch 分步启动。

另开终端控制机器人：

```bash
source /opt/ros/humble/setup.bash
cd /home/cjh/Pure3D_ws
ros2 launch Entrance/launch/sim/keyboard.launch.py
```

按 `2` 进入蹲姿，等待约 3 秒；再按 `2` 站立，再等待约 3 秒。按 `7` 进入现有 RL 步态。W/S 前后、C/D 横移、Q/E 转向、空格清零、P 缓停/恢复、1 被动。CPU/GPU、关节、策略和状态机沿用现有控制器。

使用 SCAN 的 RViz `Interact` 工具拖动 `/basic_controls` 的 `end_pos`。此时键盘入口用 `ros2 launch Entrance/launch/sim/keyboard.launch.py publish_cmd_vel:=false`，保留模式键，由 SCAN 发布速度。默认 `flight_type:=1` 是手动目标；`flight_type:=3` 等待外部 `/pct_path`，本入口不生成全局路线。

## Faster-LIO 在线建图

离线采集流程与独立终端命令见 [Entry bag 工作流](../../../common/bag_tools/docs/BAG_WORKFLOW.md)。`ROS_DOMAIN_ID=184 ros2 launch Entrance/launch/sim/faster_lio.launch.py 'bag:=<bag目录>'` 只启动离线建图和 TF 桥；另开播放器先保存 PCD，再单独按 goal 分 node、计算 cost、合并。

机器人站稳后，在另一个终端启动：

```bash
source /opt/ros/humble/setup.bash
cd /home/cjh/Pure3D_ws
ros2 launch Entrance/launch/sim/faster_lio.launch.py
```

启动脚本等待连续 1 秒仿真时间的静止 IMU：加速度模长 9.3–10.3 m/s²、角速度小于 0.08 rad/s、各加速度分量变化小于 0.4 m/s²。等待上限默认 120 秒墙钟时间。该检查避免在出生后的自由落体阶段用接近零的加速度初始化重力。启动后再移动机器人。

在另一终端查看建图：

```bash
source /opt/ros/humble/setup.bash
cd /home/cjh/Pure3D_ws
ros2 launch Entrance/launch/sim/rviz.launch.py
```

Faster-LIO 上游 ROS 2 的两个 PointCloud2 显示项直接加入 `Entrance/config/scan.rviz`：累计扫描（120 秒）与当前帧，均订阅 `/faster_lio/cloud_registered`。Gazebo 启动的现有 SCAN RViz 可直接使用，已打开的窗口重新加载该配置即可。来源见 [配置说明](../asset/FASTER_LIO_RVIZ.md)。`Entrance/launch/sim/rviz.launch.py` 也加载同一份 SCAN 配置。

该脚本单独启动 RViz，Fixed Frame 为 `world`，使用仿真时钟；原始 `/livox/lidar` 扫描默认可见，录包时不依赖 Faster-LIO 或 `map` TF。公共建图 TF 链为 `world → map → camera_init → body`：`world → map` 首帧自动对齐后固定，`map → camera_init` 为单位变换，后一段为 LIO 机身估计。Gazebo 真值为独立 `world → base` 支路，SCAN 使用 `world`。

`Entrance/launch/sim/faster_lio.launch.py` 管理坐标转换节点的生命周期，无需另开连接脚本。改用这些帧名和安装外参后，须先在原建图终端 Ctrl+C，再手动重新运行 `ros2 launch Entrance/launch/sim/faster_lio.launch.py`；Gazebo 可以继续运行。`Entrance/launch/sim/map_tf.launch.py` 仅供独立调试转换节点，它要求输入 `/faster_lio/raw_odometry`，不与正常启动脚本重复运行。

Faster-LIO 原始 IMU 位姿保留在 `/faster_lio/raw_odometry`，其原始 TF 重映射至 `/faster_lio/raw_tf`。转换节点发布 `/faster_lio/Odometry`、`/faster_lio/path` 和公共 `camera_init → body`。这些公共位姿代表机身；原始估计与建图算法不变。`world → map` 只计算一次，`map → camera_init` 始终为单位变换；两条静态边和安装参数记录在 `outputs/gazebo_lio/map_anchor.json`。

启动日志中的 `Using AVIA Lidar` 是此 Faster-LIO 版本给 `lidar_type=1` 的历史分支名称。该分支订阅 `livox_ros_driver2/CustomMsg`，读取每点 XYZ、tag、line 和 offset_time；本仿真的 MID360 使用相同消息接口。实际雷达仍由 Gazebo 的 `mid360_points_plugin` 和 MID360 CSV 方向表生成。保持 `lidar_type=1`；不要为更改日志文字而切换到 Velodyne/Ouster 的 PointCloud2 分支。

| 接口 | 话题 / 坐标 |
|---|---|
| 仿真 LiDAR | `/livox/lidar`，PointCloud2，`laser_livox` |
| Faster-LIO 输入 | `/livox/lidar2`，原生 CustomMsg，`laser_livox` |
| 仿真 IMU | `/livox/imu`，`livox_imu_link` |
| 仿真真值 | `/Odometry_gazebo`，`world` → `base` |
| LIO 机身里程计 | `/faster_lio/Odometry`，`camera_init` → `body` |
| 注册点云 | `/faster_lio/cloud_registered`，`camera_init` |
| LIO 轨迹 | `/faster_lio/path` |

`faster_lio.yaml` 使用仿真时钟和原生 CustomMsg。启动时按外参接口生成 `outputs/gazebo_lio/faster_lio.actual.yaml`：当前 `p_IMU = p_LIDAR + [0.011, 0.02329, -0.04412]`，旋转单位阵。Faster-LIO 启动脚本只加载该包与本地 glog/gflags 依赖，使用本工作区构建的 Livox 消息。

仿真导航继续使用 Gazebo 真值。注册点云保持 `camera_init` 坐标；与 `map` 数值一致，通过 `world → map` 静态连接与 Gazebo 场景对齐。RViz 通过 Decay Time 累积显示每帧注册扫描；它不是单条消息发布的完整体素地图。初始静态对齐不修正后续 LIO 漂移，也不将估计结果反馈给 SCAN。

## 仿真外参与真机配置

仿真配置在 `Entrance/config/sim/`。`frames.yaml` 默认采用 `mount.source: urdf`，读取 `base → livox_imu_link` 的实际平移和安装角；`anchor.source: gazebo` 用首帧机身估计与同时间的 `world → base` 真值计算 `world → map`，随后保持静态。当前 `base → laser_livox` 安装平移为 `[0.2, 0, 0.18] m`，高度从 `0.08 m` 抬高了 `0.1 m`，安装俯仰保持 `0.785 rad`。实际模型位置来自 A1 的 `xacro/mid360.xacro`，`frames.yaml` 中的参数安装值同步维护。安装位置变更后重新生成 Gazebo 机器人并重启 LIO，以重新读取机身安装关系和建立首帧对齐。

共享转换公式为 `T_camera_body = T_camera_imu × inverse(T_body_imu)`。雷达和内置 IMU 一起倾斜，安装角只在机身与传感器之间转换一次，不重复加入内部 LiDAR→IMU 外参。

真机使用独立的 [real 配置和入口](../real/README.md)，不加载仿真 overlay、不使用仿真时钟或 Gazebo 初始真值。共用转换代码放在 `../scripts/`。

## 场景和路径

总入口默认使用 EOLP 的 Building-no-rails。用已经导出的 seed `20261109` 大地形启动飞行模式 4、Light + SCAN、Faster-LIO 和 Tiny cost：

```bash
source /opt/ros/humble/setup.bash
./Entrance/shell/total_terminal.sh --scene terrain-prepared --flight-type 4 --planner-backend light_scan --model tiny
```

`terrain-prepared` 读取 `datasets/simulation/gazebo_terrain/` 的 world、STL 和出生位姿。可用 `PURE3D_TERRAIN_OUTPUT` 指向其他已导出的目录。需要从原始 terrain-generator 场景重新导出时，使用 `--scene terrain` 或在独立入口设置 `PURE3D_SIM_SCENE=terrain`。

历史参考场景 `datasets/wfc_capability_dr_v2/scene_20261109/terrain.obj` 为 3 层、173,340 个三角形。`prepare_world.py` 将全部源三角形导出为 STL，可视和碰撞共用同一网格；坐标保持米制、Z 向上、源场景朝向与孔洞。STL 使用 float32 坐标，不作简化或凸包。

导出 terrain 场景时，下方额外添加静态 `ground_plane`，防止离开场景边界后无限下落。地面放在 Mesh 最低点以下至少 5 cm，并向下取整到 5 cm；历史参考场景为 `z=-1.0 m`，低于原 Mesh 最低点 `-0.948 m`。地面可视范围 200 × 200 m，ODE 平面碰撞无限延伸。设置 terrain 模式后重新运行 `ros2 launch Entrance/launch/sim/gazebo.launch.py` 即可自动更新世界，无需重新编译。此前的 Faster-LIO 测试结果对应添加该地面之前的场景。

地面碰撞已在独立 Gazebo 实例验证：将半径 0.15 m 的球放在场景外 `x=30 m`，从 `z=1 m` 下落，最终球心稳定在 `z=-0.85 m`。报告位于 `docs/reports/gazebo_ground_validation/report.json`；该检查未启动 Faster-LIO。

terrain 模式自动选择指定楼层的楼梯平整入口，检查 0.8 × 0.6 m 范围的 9 个竖直支撑与净空探针，再以地面上方 0.4 m 的 base 初始高度生成模型。历史参考位姿约 `(-8.659, 1.700, 0.400)`。这是有限探针检查；运行验证进一步检查机器人实际站立高度。Gazebo 场景与真值使用 `world`，地图通过首帧静态锚定与场景对齐。

```bash
source /opt/ros/humble/setup.bash
# 使用已有的另一个 terrain-generator 场景或楼层
PURE3D_SIM_SCENE=terrain PURE3D_TERRAIN_SCENE=/absolute/path/scene_123 PURE3D_TERRAIN_FLOOR=1 ros2 launch Entrance/launch/sim/gazebo.launch.py
```

场景须包含生成器的 `terrain.obj`、`scene.json`、楼梯入口单元与楼层元数据。也可以用 `x:=... y:=... height:=... yaw:=...` 覆盖出生位姿。覆盖位姿由调用者选择，不经过自动探针检查。

| 路径或环境变量 | 用途 |
|---|---|
| `.cache/gazebo_ws/{build,install,log}` | 独立 colcon 产物，15 个包 |
| `datasets/simulation/gazebo_terrain/` | 转换场景、出生位姿与源 SHA256 报告 |
| `outputs/gazebo_validation/` | 在线测试报告、日志、真值/LIO 轨迹、累积地图 PCD |
| `PURE3D_SIM_ROOT` | 修改构建与安装根目录 |
| `LIBTORCH_ROOT` | 默认 `$HOME/third_party_lib/libtorch` |
| `PURE3D_SIM_SCENE` | `building` 默认，`terrain` 使用原生成场景 |
| `PURE3D_BUILDING_MODEL` | 默认 EOLP 的 Building-no-rails 模型目录，使用同包 world |
| `PURE3D_TERRAIN_SCENE` / `PURE3D_TERRAIN_OUTPUT` | 源场景 / Gazebo 场景导出目录 |
| `PURE3D_LIO_WS` / `PURE3D_LIO_OUTPUT` | Faster-LIO underlay / 运行目录 |
| `PURE3D_LIO_FRAMES_CONFIG` | 覆盖仿真安装外参与地图连接配置 |
| `PURE3D_SIM_LIO_CONFIG` | 覆盖仿真 Faster-LIO 参数文件 |
| `PURE3D_LIO_WAIT_TIMEOUT` | 等待 IMU 稳定的墙钟秒数 |

运行环境只加载系统 ROS 和本工作区仿真 overlay，不加载训练环境或其他导航工程的 overlay。构建只发现四足包、MID360、Livox 接口、unitree_gazebo 和 SCAN planner 子目录，不构建 SCAN 自带的另一套 Go2 simulator。

## 历史在线验证

清理前的在线验证使用独立 ROS 域和 Gazebo master，检查控制器、静止初始化、LIO 输出、TF 和站立高度。一次性验证脚本已在 2026-11-17 源码清理中移除，历史结果与输入保留在本机输出目录。正常启动使用本文开头的独立入口或`Entrance/shell/total_terminal.sh`。

MID360 插件当前为瞬时射线快照，逐点 offset_time 均为零；本测试覆盖仿真启动和静止建图，不覆盖真实滚动扫描去畸变、完整路线、楼梯通过或实机效果。

2026-11-14 本机验证：15 个包构建通过；四个控制器均为 active。站稳后采集 15 秒仿真时间，收到 154 帧 Faster-LIO 注册点云与 154 条里程计，输出约 10 Hz，静止位置相对首帧最大变化 0.0090 m，累计地图 45,240 点。真值和 LIO TF 均有输出。报告见 `docs/reports/gazebo_validation/report.json`，地图见同目录 `registered_map.pcd`；测试进程已退出。Gazebo/RViz 图形窗口未纳入这次无窗口验证。

## Entry 分块与 cost

建图就绪后可运行 `ros2 launch Entrance/launch/sim/whole_map.launch.py mode:=blocks rviz:=true` 验收分块和节点，或 `ros2 launch Entrance/launch/sim/whole_map.launch.py mode:=cost rviz:=true` 启用 best56。完整接口与测试见 [Entry 说明](../../../common/README.md)。应用本次坐标约定后，需要重新构建并手动重启 Gazebo 与 Faster-LIO，使旧 TF 发布者退出。
