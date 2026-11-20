# Unitree Gazebo 与 LIO 支撑代码

导航框架的启动文件与运行配置统一放在 [Entrance](../../Entrance/README.md)。本包保留 Gazebo 场景资源、仿真真值节点、地形导出工具和 LIO 外参/TF 转换代码。

```text
unitree_gazebo/
├── models/、worlds/     Gazebo 模型与世界
├── src/                Gazebo 真值里程计
├── scripts/            IMU 检查、外参和地图 TF 转换
├── sim/                地形导出工具与仿真说明
├── real/               真机接入说明
└── asset/              RViz 来源说明和许可证
```

在工作区根目录执行：

```bash
./Entrance/shell/build.sh
source /opt/ros/humble/setup.bash
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building
# 机器人站稳后，在另一个终端独立启动建图
ros2 launch Entrance/launch/sim/faster_lio.launch.py
# 只检查真机参数
ros2 launch Entrance/launch/real/faster_lio_real.launch.py check_config:=true
```

仿真 LIO 配置在 `Entrance/config/sim/`，真机配置在 `Entrance/config/real/`，显示共用 `Entrance/config/scan.rviz`。独立 Gazebo 方案保持建图手动启动，整套顺序方案为 `Entrance/shell/total_terminal.sh`。

公共建图 TF 在仿真中为 `world → map → camera_init → body`。真机不使用 Gazebo 真值锚定。具体接口、历史验证和硬件接入要求见 [仿真说明](sim/README.md)、[真机说明](real/README.md) 与 [真机安装文档](../../docs/REAL_WORLD_SETUP.md)。
