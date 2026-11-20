# 上游来源与接入范围

- 仓库：[wuyi2121/SCAN-Planner](https://github.com/wuyi2121/SCAN-Planner)。
- 分支：`ros2-community`。
- 检出提交：`d0b921c9b05a6d291d144d60882b2e0e88d2c0e0`。
- 原始研究作者：Han Zheng、Zhe Chen、Yiwen Fu、Ming Yang、Tong Qin；ROS2 移植署名见 NOTICE。
- 上游许可证：Apache-2.0，保留 LICENSE 和 NOTICE。

本工作区直接收录所需源码，不使用 Git 子模块。`plan_env/path_searching/bspline_opt/traj_utils` 的包名、库目标和头文件路径添加 `scan_` 前缀；主包为 `scan_planner`，消息为 `scan_planner_msgs`。

接入改动包括 PCL 依赖导出、map/传感器 TF 适配、机体系速度转换、PCT 路径高度和 transient-local 缓存、ROS 仿真时钟、交互标记目标与限频更新、A1 配置。SCAN 的搜索与优化主体保留，现有 EGO 独立保留。

模式 3 的全局路径跟随迁移自本工作区 `src/planner/plan_manage/src/ego_replan_fsm.cpp`：路径筛选、最近点投影与累计路长目标、保留全局参考的三次重规划尝试、实测起始状态、5Hz FSM 和 0.2m 到达判断；旧局部轨迹重用时的起点及重采样次数限制来自本地 `planner_manager.cpp`。SCAN 原生建图、搜索、优化和跟踪器继续使用，模式 1/2 保留原状态机。具体映射与验证范围见工作区 `docs/scan_planner_integration.md`。

未纳入上游演示 assets、PCD、tools 和自带 simulator。`src/simulator/COLCON_IGNORE` 防止本地保留的上游 simulator 被 colcon 扫描；当前使用 glocal-nav2 已有的 A1/Gazebo。
