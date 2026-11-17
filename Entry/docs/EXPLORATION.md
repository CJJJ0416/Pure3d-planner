# 人工目标探索：在线 LIO → goal 触发 cost → 表面 A* → SCAN

本文中的对照实验命令依赖本地保留的实验模块，该模块不随源码发布。正式在线入口见 [Entry](../README.md)。
你通过 RViz interactive marker 选目标，起点始终取当前 `world → base`。全局默认使用 `surface` 稀疏多层表面图和现有 C++ 单向 A*，当前 `scan_path_source: astar` 完全跳过 GPMP；局部 SCAN 模式 3 直接跟踪 `/pct_path` 的 A* 路径。按最新策略，通行性完全由 cost 决定，不附加坡度、阶高、法向、净空和机身碰撞过滤。条件 `[.5,.4,.4,0]` 仍输入模型，surface 不再将这些条件解释为几何硬约束。

导航入口将原 GPMP 的 `endpoint_offset_m` 设为现有机身高度 `0.4 m`，沿 world +Z 只加一次；SCAN 桥先减 0.4 m、FSM 再加 0.4 m，最终保持全局 GPMP 的机身参考高度。独立静态 planner 的默认 0.5 m 配置保持原值。

初始连续静止观测 3 秒后，对当时的**完整累计 PCD**计算第一份 cost。运动中持续累积 LIO 点云；每次到达目标（LIO 位姿距离 ≤0.3 m 持续 3 秒），冻结一份完整地图快照，重新对全部原始点计算 cost。每次请求均为 `input_scope=whole_pcd`、`force_full=true`，每个空间窗口都重新运行 PTv3 和 cost head。全流程按整图版本保存结果，已移除时间 node、EROI 分段和节点结果合并。导航按 Gazebo base 距离 ≤0.2 m 停车，等待连续 3 秒以及本次整图 cost 完成，再等待人工目标。

模型和规划全部在重力对齐 `world` 中计算；Entry 的点云存储和输出保留原始 `map` XYZ 与点序。TF 保持 `world → map → camera_init → body`，其中 `map → camera_init` 为单位变换；Gazebo 真值为 `world → base`。

RViz 使用 `/entry/rviz/cloud` 与 `/entry/rviz/cost_cloud`：发布前按 `0.1 m` 体素降采样，最多 `200000` 点、最高 `1 Hz`，只更新有新输入的快照。cost 每个体素选择最高代价的原始点，避免平均淡化高代价区域。通过 `display_voxel_m`、`display_max_points`、`display_hz` 调整；这些参数只影响显示。推理、PCD 存储及 DC+A* 使用完整数据，导航仍订阅 `/entry/cost_cloud`。默认不显示原始几何时，也不反复构造其完整 ROS 点云消息。

## 独立终端

以下入口独立运行，不会互相启动或关闭。使用相同 ROS_DOMAIN_ID，在线默认域 0；不要与离线回放/旧地图查看器放进同一个域。

首次构建当前 SCAN 的导航等待接口：

```bash
cd /home/cjh/Pure3D_ws
./build_gazebo.sh
```

终端 1，仿真与 SCAN。关闭其默认 RViz，统一由导航入口打开一个窗口：

```bash
cd /home/cjh/Pure3D_ws
./run_gazebo.sh planner:=scan flight_type:=3 rviz:=false
```

终端 2，键盘只负责机器人状态，不发布速度：

```bash
cd /home/cjh/Pure3D_ws
./keyboard_gazebo.sh -p publish_cmd_vel:=false
```

按 `2` 蹲姿，等待约 3 秒，再按 `2` 站立并站稳。之后进入 RL 用 `7`；P 键仍可人工缓停。导航不会切换机器人模式。

终端 3，机器人静止时启动在线 Faster-LIO：

```bash
cd /home/cjh/Pure3D_ws
./run_faster_lio.sh
```

仿真入口会持续等待连续 1 秒的有效静止 IMU，直到站稳或手动退出；不会因在键盘上准备超过 120 秒而提前退出。可用 `PURE3D_LIO_WAIT_TIMEOUT=<秒>` 设置限制，`0` 表示持续等待。真机默认仍为 120 秒。若 cost 没显示，先检查 LIO 日志与 `/faster_lio/cloud_registered` 是否实际有输出；模型已加载并不代表已有点云输入。

终端 4，goal 触发的 Entry：

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/run.sh --config Entry/configs/exploration.yaml --mode cost
```

终端 5，人工目标、表面 A* 与唯一导航 RViz：

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/navigate.sh --rviz
```

等待初始 cost 显示后，使用 RViz 顶部 `Interact` 拖动粉色 `GOAL` 的 XYZ 箭头，**松开鼠标才提交规划**。目标标记表示机身 XYZ（第 0 层约 z=0.4 m），无额外的 0.5 m 显示偏置；青色起点随当前 base 更新，不可拖动。也可向 `/entry/navigation/goal` 发布带有效 frame 的 PoseStamped，导航会经 TF 转到 world。

新的目标先停车并规划；成功后仅发布一次完整 world 路径到 `/pct_path`，同时将实际路径终点发到 `/scan/goal`，用于 Entry 的 goal 到达触发。等待 SCAN 桥确认 `/scan/initial_path` 后恢复局部跟踪。新 cost 对应新的地图版本，表面图在下一次人工目标请求时重建；同一预测版本的表面图和 GPMP 查询结构复用。不会因新扫描或拖动过程重复规划，不自动选择下一个目标。

当前默认 `scan_path_source: astar`：只计算和发布蓝色 A* 机身路径，`/pct_path` 与 `/entry/astar_path` 的 XYZ 完全一致；不初始化 GPMP 查询结构，不运行 GPMP 优化及检查，并清空旧 GPMP 显示。改成 `gpmp` 可显式恢复原优化流程；该模式仍要求 GPMP 收敛并通过 cost 与层匹配检查。

默认 `planner_backend: surface`。C++ 一次扫描输入数组，聚合完整 cost 和原始点严格 cost < 0.5 的候选；使用 0.1×0.1×0.05 m 稀疏单元，沿竖直连续观测分组，跨度超过 0.1 m 就拆成多个单元，不删除该组。相邻 XY 柱先限制高度差不超过 `surface_slice_dh`（默认 0.5 m），再按 `(cost, 高度差)` 互选配对，同柱不连上下层；缺测柱和缺测对角断开。显式启用 GPMP 时，直接复用完整 cost 体素和 KD-tree，不再重读原始 PCD。优化后仅检查观测单元连接和连续 cost < 0.5，不做几何通行性检查。起终点优先在各自 XY 单元内按同一高度窗口和 `(cost, 高度差)` 匹配；雷达盲区无候选时，分别在原有 1 m/0.5 m 吸附半径内查找最近的有效单元，始终保留高度窗口，不再用无限制最近点跨层吸附：起点最大吸附 1 m，目标最大 0.5 m；超过限制、无路、cost 超限时保持停车。

`./Entry/launch/navigate.sh --planner-backend dc --rviz` 可单次选择旧 DC 对照，几何参数仍仅在 DC 后端生效。`Surface A* cells - height` 显示可搜索单元，最多 20 万点；不默认绘制图边。完整原始 PCD、整图模型推理与保存、SCAN 原始雷达输入保持原流程。

## 等待、失败和保存

导航状态 `/entry/navigation/status` 区分 `waiting_initial_cost`、`awaiting_goal`、`planning`、`waiting_scan_path`、`following`、`waiting_goal_cost`、`failed`。Entry `/entry/status` 另行报告累计几何版本、最后触发的整图版本、请求编号与 cost 版本；几何版本在运动时继续增长，而 cost 版本保持不变，这是预期行为。

`/entry/navigation/hold` 每 0.2 秒发布导航等待状态。SCAN 仍为唯一 `/cmd_vel` 发布者；等待、目标替换、规划失败和到达后均输出零速度，导航心跳消失超过 2 秒也停车。未运行该导航接口时，原 SCAN 行为保持。不要同时运行 VST 路径发布器、自动 bag mission follow 或另一个 `/pct_path` 发布者。

Entry 正常退出时会等待在算任务结束，并对最终接收几何补一次完整推理。输出 `latest_geometry.pcd/.npy`、`latest_prediction.pcd`、`latest_cost.npy` 和对应地图版本。`goal_events.json` 只记录人工目标及到达触发，`prediction_history.json` 记录每次整图推理的版本、点数和分阶段耗时。新运行不会生成 `nodes/`、`observations/n*/` 或节点合并产物。已有历史实验数据保留。

导航在 `queries/<编号>/` 保存 `astar_world.npy`、`support_world.npy`、`gpmp_world.npy`、`gpmp_support_world.npy`、`gpmp_time.npy`、`gpmp_trajectory.csv`、选中的 `body_world.npy` 和 `query.json`。后者记录 A*、GPMP 收敛状态、优化耗时、完整代价场准备耗时和 SCAN 路径来源。另保存 maps、目标与到达事件。实际执行完成、模型准确率与优化器收敛是不同证据。

## 推理范围与耗时

模型保留 Small best56、条件 `[.5,.4,.4,0]`、4×4×3 m 空间 core 和 1 m 上下文。这些空间窗口用于匹配训练预处理；每次请求覆盖整个累计 PCD，所有原始点只输出一份 cost。`inference_recompute: full` 强制所有窗口重新计算，默认关闭跨 goal 的窗口结果复用。

旧运行最后一次 29,336,711 点的记录为：总推理 17.16 s，点对应 5.14 s，分组/上下文索引 5.02 s，PTv3 特征提取 5.10 s，cost head 0.42 s。该记录属于旧缓存实现，仅用于说明 head 开销；没有独立测量 decoder，不能直接视为新全量实现的性能。

DC 原始点筛选实测：`docs/reports/entry_navigation_validation/dc_filter_1933/report.json` 使用相同 8,609,464 点输入，保留 6,083,508 点；距离场从按原输入包围盒估计的 122,096,448 个格子减少到实测 36,115,002 个。筛选构图 11.59 s，未筛选的历史记录 52.02 s；两组原起终点均可搜索到路径。时间对比并非同时受控跑分，路径连通性不代表实际跟踪成功。

历史 node/VST 对照见 `BAG_WORKFLOW.md`，不再作为当前在线流程。新版回归记录在 `outputs/entry_navigation_validation/whole_pcd_goal/`，实际仿真运行使用独立新目录，旧运行保持停止。


## surface 与 DC 离线对照

Gazebo 保持停止时可复用保存的同一份 world cost 点云。两个后端顺序各测三次，分开记录地图准备、GPMP 查询结构和缓存查询；不重新运行模型，也不将离线结果计作机器人实际执行。

```bash
cd /home/cjh/Pure3D_ws
/usr/bin/python3 -m Entry.experiments.surface_benchmark \
  --reference datasets/benchmarks/surface_planning/reference \
  --output outputs/entry_navigation_validation/surface_cost_only_new \
  --repeats 3 --backends surface dc
```

查看已完成的对照：

```bash
cd /home/cjh/Pure3D_ws
source common/unitree_gazebo/sim/env.sh
ROS_DOMAIN_ID=198 /usr/bin/python3 -m Entry.rendering.surface_view \
  --run outputs/entry_navigation_validation/surface_cost_only_20261115 \
  --map 8209 --query 23 --rviz
```

该查看器仅发布保存的 cost、表面和路径，不发布 `/pct_path`、目标或控制输入；不需要 `/clock`，Fixed Frame 为 world。蓝色为 surface A*，粉色为通过检查的 surface GPMP，橙色为 DC GPMP 对照。surface 单元按高度着色，可通过显示项开关与 cost 点云比较。失败候选单独保存为 `rejected_gpmp_world.npy`，不作为有效 GPMP 路径发布。


局部层匹配采用与 VST `match_best_layer` 相同的顺序：高度差先限制在 `surface_slice_dh` 内，再优先选低 cost，代价相同时选择高度更近者；零 cost 有效。该参数仅定义局部邻接窗口，不重新计算坡度、阶高或净空代价。A* 起终点与优化后采样点均使用同一匹配规则；无合格候选就明确失败，不放宽到远处楼层。端点允许在既有吸附半径内查找附近单元以处理原地雷达盲区；此回退不添加任何图边，也不改变图和轨迹检查的匹配规则。cost 仍保持当前管线严格 `< 0.5`（示例函数中的 `<=` 未改变现有阈值语义）。图保持双向互选和对角检查，柱内不添加垂直边。高度窗口边界使用 1e-6 m 浮点容差。
