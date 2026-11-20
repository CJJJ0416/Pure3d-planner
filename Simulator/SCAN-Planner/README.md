# SCAN-Planner：glocal-nav2 的 ROS2 接入

Pure3D 工作区的 terrain-generator 仿真使用根目录 `./Entrance/shell/build.sh` 和 `ros2 launch Entrance/launch/sim/gazebo.launch.py`。
机器人、MID360、SCAN 与手动 Faster-LIO 建图步骤见 [Gazebo 集成说明](../unitree_gazebo/README.md)。

本目录保留 SCAN 的规划、搜索、地图、轨迹消息和跟踪源码，接入现有 Unitree A1 / Gazebo Classic / RL 控制器。来源、版本和改动范围见 [UPSTREAM.md](UPSTREAM.md)，许可证与原作者署名见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

## 启动

从 glocal-nav2 工作区根目录运行：

```bash
./auto_ros2.sh planner:=scan flight_type:=1  # 拖动 InteractiveMarkers 的 end_pos
./auto_ros2.sh planner:=scan flight_type:=3  # 接收独立 PCT 进程的 /pct_path
./auto_ros2.sh planner:=ego                 # 原 EGO
./auto_ros2.sh planner:=None                # 仅仿真
```

路径模式另开终端运行 `./global_planner.sh start_frame_id:=base`。手动模式无需 PCT。

RViz 使用 `Interact` 工具，在 `/basic_controls` 拖动 `end_pos` 六自由度控件；`start_pos` 跟随机器人，仅作显示。显示 Z = 目标 Z +0.5m，拖拽时减回；SCAN 保留目标 XYZ。显示反馈最多 20Hz、起点最多 10Hz 且变化才更新，每 0.5s 按 1cm 位置变化提交目标。SCAN 手动模式不使用 2D Goal Pose。

## 配置

- 主仿真话题：[local_planners.yaml](../unitree_gazebo/config/local_planners.yaml)。
- SCAN 地图、规划视野和控制器：[a1.yaml](src/planner/plan_manage/config/a1.yaml)。
- 独立启动：`ros2 launch Entrance/launch/sim/scan.launch.py flight_type:=1`；默认话题来自 [local_planners.yaml](../../Entrance/config/local_planners.yaml)。
- ROS2 依赖包与头文件加 `scan_` 前缀，避免覆盖 EGO 的同名包。

主 launch 不再暴露局部话题、地图尺寸和 planning_horizon；修改 YAML 后使用 `auto_ros2.sh` 重编译并启动。

## 提交范围与验证

本工作区只纳入所需的 `src/planner`、配置和许可证。上游演示地图、assets、自带 Go2 simulator 及其运行依赖没有纳入源码提交；其独立 `run.launch.py/simulator.launch.py` 不属于当前工作区支持的入口。使用上面的 A1 集成入口。

构建与测试命令、已验证范围和限制见 [SCAN 接入说明](../../docs/scan_planner_integration.md)。测试脚本在工作区 `tests/`，结果生成到本地 `test_results/scan_planner/`，不随源码提交。已验证 TF、PCT 缓存路径、交互标记限频与运动学闭环；不代表楼梯通行或实机验收。
