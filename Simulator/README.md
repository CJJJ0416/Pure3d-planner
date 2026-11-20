# 仿真与采集组件

导航框架的启动入口与运行配置仍统一位于 [Entrance](../Entrance/README.md)。本目录保存仿真、传感器采集和场景生成的实现与资源。

| 组件 | 职责 |
|---|---|
| [unitree_gazebo](unitree_gazebo/README.md) | Gazebo 世界、模型、真值里程计与 LIO/TF 支撑代码 |
| [quadruped_ros2_control](quadruped_ros2_control) | A1 描述、控制器、现有步态策略与键盘输入 |
| [SCAN-Planner](SCAN-Planner/README.md) | 仿真导航使用的 SCAN 地图、局部规划与跟踪实现 |
| [Mid360_imu_sim](Mid360_imu_sim/README.md) | MID360 模型、扫描方向表与 Gazebo 插件 |
| [livox_ros_driver2](livox_ros_driver2/README.md) | 仿真所需的 Livox 原生消息接口 |
| [scanning](scanning/README.md) | Warp 滚动扫描、IMU、Faster-LIO 回放和数据检查 |
| [geneator](geneator/README.md) | terrain-generator 源码快照与 WFC 多层场景生成工具 |

Usage（工作区根目录）：

```bash
./Entrance/shell/build.sh
source /opt/ros/humble/setup.bash
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building
./Entrance/shell/total_terminal.sh --scene terrain-prepared --dry-run
```

离线扫描模块使用 `python -m Simulator.scanning.pipeline`，场景生成脚本位于 `Simulator/geneator/`。生成数据、训练权重与运行输出继续使用工作区既定的 `datasets/`、`weights/` 和 `outputs/`。
