# 整图 cost 与历史导航工具

新增独立的 [SCAN 盒内 LIO 体素与定时局部 cost](LOCAL_COST.md)：`ros2 launch Entrance/launch/sim/local.launch.py model:=tiny rviz:=true`。它维护有界体素输入，不依赖整图落盘，使用独立 `/entry/local/*` 话题；当前仅输出代价，尚未加入 SCAN 优化器。

当前流程：`/faster_lio/cloud_registered` → 累计完整 PCD → 初始/goal 到达时整图推理 → cost < 0.5 稀疏多层表面 A* → SCAN。目标由 RViz interactive marker 提供，起点取当前机器人 base。完整启动与操作见 [人工目标探索](../../planning/docs/EXPLORATION.md)。

当前 `scan_path_source: astar`，SCAN 直接跟随 A*，完全跳过 GPMP 初始化、优化和其代价检查。A* 的 cost、局部高度匹配及图连通检查保留；`gpmp` 仍可作为显式选项恢复。

初始静止 3 秒后计算一次；每次到达 goal 并停留 3 秒后，重新对当时完整 PCD 计算一次。运动中持续累积扫描。在线流程已移除时间 node、按 node 保存/推理和节点合并。goal 仅是整图更新触发事件。

`surface` 后端仅按预测 cost 判定通行：不计算法向、坡度、台阶高度、净空或机身碰撞。坐标仅用于稀疏体素、分层与相邻单元连接。C++ 一次扫描完整数组，同时聚合完整 cost 体素和严格 `0 <= trav_cost < 0.5` 的候选表面；水平 0.1 m、垂直 0.05 m。相邻 XY 柱先限制高度差不超过 `surface_slice_dh`（默认 0.5 m），再按 `(cost, 高度差)` 互选配对，柱内不连上下层，缺测柱断开，对角不跨缺测角点。所有 cost 合格点均保留到表面单元，不因几何特征删除。

显式启用 GPMP 时，复用同次聚合的完整 cost 体素与缓存 KD-tree，保留高 cost 惩罚；优化结果检查观测表面连通性和连续查询 cost（严格 < 0.5）。失败保持停车。`maps/<版本>/map.json` 记录 `cost_only` 策略、计数和分阶段耗时；无稠密 EDT/DC，也不经 PCD 写入后重新读取才能规划。完整模型输入、保存点云及 SCAN 原始雷达保持现有流程。

在线配置选择 `planner_backend: surface`；用 `ros2 launch Entrance/launch/sim/navigation.launch.py planner_backend:=dc rviz:=true` 对照旧 DC 后端。独立静态 `planner` 工具默认仍使用 DC，旧后端及其几何参数不变。

```bash
cd /home/cjh/Pure3D_ws
ros2 launch Entrance/launch/sim/whole_map.launch.py config:=Entrance/config/navigation.yaml mode:=cost
# 独立导航终端：
ros2 launch Entrance/launch/sim/navigation.launch.py rviz:=true
```

Gazebo、Faster-LIO、Entry 和导航保持独立入口与生命周期。运行一套同 ROS 域的实例；`Ctrl+C` 后 Entry 排空任务并保存最终完整几何和 cost。

## 坐标与模型

TF 为 `world → map → camera_init → body`，`map → camera_init` 为单位变换；Gazebo 真值为 `world → base`。模型在重力对齐 world 中编码，输出与存储保留原 map XYZ 和点序。

Small 解冻 best56 只加载一次，记录 checkpoint SHA256，条件为 `[0.5,0.4,0.4,0.0]`。每次完整地图请求使用训练时的 4×4×3 m 空间窗口、1 m 上下文和原有代表点预处理。`inference_recompute: full` 强制每个窗口重算，`input_scope: whole_pcd` 表示所有原始点一次覆盖。空间窗口不代表运动时间 node。

## 话题与显示

| 话题 | 内容 |
|---|---|
| `/entry/cloud` | 完整累计 XYZ，供需要完整几何的订阅者使用 |
| `/entry/cost_cloud` | 完整 XYZ + trav_cost，供 DC+A* 使用 |
| `/entry/rviz/cloud` | RViz 专用降采样几何 |
| `/entry/rviz/cost_cloud` | RViz 专用降采样 cost |
| `/entry/astar_path` | world 下的 A* 机身参考路径，RViz 蓝色 |
| `/entry/gpmp_path` | world 下的 GPMP 路径，RViz 粉色 |
| `/entry/surface_cloud` | world 下的可搜索表面单元，显示限额同 RViz cost |
| `/pct_path` | 当前选中的 GPMP 路径，供 SCAN 模式 3 跟踪 |
| `/entry/blocks` | 模型空间窗口 MarkerArray |
| `/entry/status` | 整图版本、请求、已到达目标数、耗时和状态 |
| `/entry/rviz/status` | 显示前后点数和降采样耗时 |

几何与 cost 均以 map 发布；导航 RViz Fixed Frame 为 world。显示默认 0.1 m 体素、最多 20 万点、最高 1 Hz，cost 每体素保留最高代价原始点。完整 PCD、推理及规划使用全量数据。显示采样在后台执行，忙时合并显示版本；未显示完整几何时避免重复传输其全量消息。

导航 RViz 同时显示青色 SCAN 局部占用点云 `/scan/grid_map/occupancy` 和滑动地图边框。该点云使用 Best Effort、Volatile QoS；局部轨迹 `/scan/optimal_list` 使用 Marker、Reliable、Transient Local。膨胀占用点云显示项默认关闭，需要时勾选 `SCAN inflated occupancy - optional`，避免持续传输更大的膨胀点云。

## 保存与验证

输出目录包含 `config.yaml`、`frames.json`、`model.json`、`goal_events.json`、`prediction_history.json`、`geometry.json`、`status.json`、`display.json`、`worker.log`，以及最新 `latest_geometry.pcd/.npy`、`latest_prediction.pcd/.json`、`latest_cost.npy`。每份预测记录源地图版本、点数、整图范围与耗时。新运行不创建 node 或 EROI 目录。

历史软件验收已完成，相关一次性测试脚本在 2026-11-17 源码清理中移除，已有报告保留在本机输出目录。

历史集成验证使用隔离 ROS 域与合成位姿，运行真实 GPU 模型、DC+A* 和 SCAN 接口；不代表实际 Gazebo 路线或地形准确率。独立编码器/decoder 耗时需另行细分测量，现有 encoding_s 包含整个 PTv3 特征提取。

[旧 bag/node/VST 实验说明](../../bag_tools/docs/BAG_WORKFLOW.md)及其结果作为历史对照保留，不属于当前整图在线入口。


显示更新使用原生稀疏哈希单次扫描，仍按 0.1 m 体素选最高 cost 的原始点、同代价保留最早原始点，并保持原先排序和 20 万点限额。大地图 core 分组同样使用稳定原生分组，保留完全一致的块 ID、原始点序与窗口调用顺序；不改变模型采样、权重或每 goal 整图推理策略。未显示几何且采用 goal 触发时，不再后台每秒复制两次整图快照。这些内置优化在新启动 Entry 时生效。

当前运行额外通过 `common.visualization.display --topic-prefix /entry/rviz_fast` 使用快速显示，并让本轮 RViz 订阅 `/entry/rviz_fast/cost_cloud`；已在运行的旧 Entry/GPU worker 不会因修改源文件而自动更新。分组索引与显示一致性测速见 `docs/reports/entry_navigation_validation/cost_update_speed_20261115/report.json`。`processing_s` 记录降采样和小消息构造时间，`receive_to_publish_s` 记录显示接收到发布的延迟；二者均不包含模型推理及完整 ROS 输入传输。
