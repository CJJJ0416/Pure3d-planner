# RViz 配置来源

当前显示布局统一放在 `Entrance/config/scan.rviz` 和 `Entrance/config/ego.rviz`，由 [Entrance](../../../Entrance/README.md) 的 Gazebo 与 RViz launch 选择。

Usage（工作区根目录）：

```bash
source /opt/ros/humble/setup.bash
ros2 launch Entrance/launch/sim/rviz.launch.py
ros2 launch Entrance/launch/real/rviz_real.launch.py
```

`scan.rviz` 保留 SCAN 显示与原始 MID360 扫描，并直接包含 Faster-LIO 累计 120 秒与当前帧两项 PointCloud2。来源和许可证见 [Faster-LIO 布局说明](FASTER_LIO_RVIZ.md) 与本目录 LICENSE 文件。

`ego.rviz` 来自早期 EGO/PCT 显示迁移。当前主流程使用 SCAN 布局；各布局的历史验证不代表本次导航闭环验证。
