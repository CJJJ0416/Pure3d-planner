# Unitree MID360：Sim / Real

Gazebo 仿真和真机接入统一放在本包，配置与启动入口分开。仿真公共建图 TF 为 `world → map → camera_init → body`：首帧对齐放在 `world → map`，`map → camera_init` 为单位变换，最后一段为经过安装外参转换的机身估计。Gazebo 真值使用独立 `world → base` 支路。真机未配置 `frames.world` 时沿用 `map → camera_init → body`。

```text
unitree_gazebo/
├── sim/                 仿真环境、场景导出、手动启动和验证
│   ├── config/          仿真 Faster-LIO 与 URDF/Gazebo 外参配置
│   ├── run_gazebo.sh
│   └── run_faster_lio.sh
├── real/                真机环境、独立手动启动
│   ├── config/          真机话题、安装外参与初始地图位姿
│   └── run_faster_lio.sh
├── scripts/             共用 LIO 运行、IMU 就绪检查、外参和 TF 转换
├── asset/               RViz 配置
├── launch/              Gazebo ROS 启动文件
├── models/、worlds/     Gazebo 模型与世界
└── src/                 Gazebo 真值里程计
```

仿真在工作区根目录依次运行 `./build_gazebo.sh`、`./run_gazebo.sh`；机器人站稳后另开终端手动运行 `./run_faster_lio.sh`。这些根目录脚本转发到 `sim/`，具体步骤见 [仿真说明](sim/README.md)。仿真入口不会自动启动建图。

真机入口为 `common/unitree_gazebo/real/run_faster_lio.sh`，安装外参在 `real/config/frames.yaml`，驱动话题和估计参数在 `real/config/faster_lio.yaml`。使用前填入实际标定和驱动配置，详见 [真机说明](real/README.md)。该入口不启动 Gazebo、机器人运动控制或硬件驱动。

两套入口分别使用仿真时钟和真实时钟。真机配置不接受 Gazebo 真值锚定，安装角通过 YAML 参数提供。Faster-LIO 仍复用外部 underlay，默认 `$HOME/livox_ws`；更改 `PURE3D_LIO_WS` 可切换。

只检查配置，不启动传感器订阅或建图：

```bash
./run_faster_lio.sh --check-config
common/unitree_gazebo/real/run_faster_lio.sh --check-config
```

先前的实际仿真建图和独立 TF 转换测试记录保留在 `runs/`。真机配置与启动入口尚未做硬件验证。
