# 局部 cost 点云规划与跟踪

## flight type 4 默认 Light 路径与 SCAN 跟踪

默认后端为 `light_scan`。Light 根据局部 cost 分层图搜索候选、评分并发布 `/pure_path`。SCAN 接收参考路径，使用自身占用图生成、优化局部 B 样条，由 `closed_loop_controller` 发布 `/cmd_vel`。人工目标来自 RViz 的 `end_pos`。

人工 marker 默认与实际 `/scan/goal` 使用相同 XYZ，移除了旧的 0.5 m 显示高度偏置。`goal_marker_height_offset` 默认 0，可显式配置显示偏移且反馈对称抵消。模式 4 在松开 marker 时提交目标。局部候选搜索优先选择当前范围内已经可达的目标候选，避免代价评分继续选择越过目标的路径。

### 靠近目标后取消

Light 先按原有高度窗口和最大 0.5 m 吸附距离，将目标映射到已观测 cost 支撑层，加固定 0.4 m 机身偏移得到规划目标。局部评分和到达判定共用这个支撑层，状态同时记录原目标、`resolved_goal` 和吸附距离。没有可用支撑匹配时保留原目标高度，避免仅凭 XY 接近就判定另一楼层到达。

默认要求相对原目标的 XY 距离不超过 0.3 m，相对规划目标的 Z 误差不超过 0.2 m。进入范围即保持停车，实际平面速度低于 0.1 m/s 持续 0.5 s 后自动取消目标。Python 参数 `--goal-tolerance-xy` 和 `--goal-tolerance-z` 可调整距离阈值。

停车确认使用滞回：进入后，仅当 XY 超过 0.4 m 或 Z 误差超过 0.25 m 才退出到达状态。确认期间保留已匹配的目标支撑高度，避免四足站姿的厘米级摆动或查询停止后的缓存年龄变化反复重置计时。超过退出范围后重新从当前机身规划，不继续旧交接。

取消后 `goal=null`、`phase=arrived`、`hold=true`，清空 `/pure_path` 和候选显示，停止后续搜索及交接。`/scan/planning/cancel_goal` 使用带仿真时间戳的 Header 同步清除 SCAN 参考、待处理请求和控制器轨迹。SCAN 在 `/scan/planning/goal_cancelled` 返回确认，旧时间戳的路径、确认和样条不再恢复执行。交互 goal 变灰，可重新拖动并提交相同位置。`status` 记录 `cancel_acknowledged` 和 `last_goal_result`。

也可手动取消当前目标：

```bash
ROS_DOMAIN_ID=183 ros2 service call /entry/navigation/cancel_goal std_srvs/srv/Trigger '{}'
```

```bash
cd /home/cjh/Pure3D_ws
./Entrance/shell/total_terminal.sh --flight-type 4 --planner-backend light_scan
```

Light 路径更新沿用近端约 1/3 保留、远段最高 2 Hz 更新的策略。当前运行的轨迹与速度由 SCAN 后端生成。

### 分层邻接（无额外连阶）

在线 Light 已移除 DC 移植的额外跨阶连边与编码前支撑补点。仅对原始观测点经 KNN 和 PTv3 得到的 cost 建图。保留原生相邻 XY 列中 `(cost, 高度差)` 互选规则和 0.5 m 高度窗口，同柱不连上下层，对角连接仍要求有效公共角点。缺失 XY 单元保持断开，不通过半径搜索补边。

`LocalCandidateQueries` 使用 `SurfaceOptions(step_connections=False)`，不再执行跨阶法向、净空和代表连接计算。滚动保留段只在原生图中校验。历史离线 DC 工具与对照数据保留，用于复现实验。

`light_scan` 的支撑和机身转换使用桥配置 `path_body_height`，当前固定为 0.4 m。每轮请求记录 `body_reference_height_m`，滚动前缀与新路径使用同一偏移。

新 cost 版本可能改变同列层片及其互选连接。保留段验证失败时，Light 清空失效参考并停车，0.5 秒后从当前 base 重新搜索完整局部候选。新参考经 SCAN 确认后恢复执行。完整搜索仍失败时保持 `cost_blocked`，仅在更新的 cost 内容到来后重试，最高 2 Hz。近端有效时继续保留原 1/3。日志分别记录 `prefix_recovery_scheduled`、`cost_map_recovery_scheduled` 和恢复查询计数。

### 执行进度与参考拒绝

控制器通过 `/scan/planning/spline_progress` 回传正在执行的轨迹 ID 和实际样条进度。SCAN 重规划与碰撞预测在收到同一轨迹的有效反馈时使用该进度，适配减速和停转。反馈超过 0.3 秒或轨迹 ID 不匹配时沿用现有时钟与冻结处理。

时间重分配之后若仍未通过三维动态检查，在原 0.5 秒规划预算内最多再进行三次重分配与拟合。每次保留原起终点导数，并重新执行动态检查。

SCAN 通过 `/scan/planning/reference_rejected` 明确返回无效或占用终点的参考。Light 根据时间戳和端点匹配当前交接，再顺序尝试同一搜索中的其他候选。候选用尽时停车并报告原因。

### 按剩余参考路径自适应限速

SCAN 规划、优化与控制器前向上限统一为 1.0 m/s。横向上限为 0.25 m/s。模式 4 的 `light_scan` 控制器同时订阅 `/pure_path` 和 SCAN 已确认的 `reference_accepted`，在三维路径上投影机身，再累计终点前的剩余水平路程。采用两份参考的剩余路程较小值，避免把已走过的路径或楼梯高度计入前方可行距离。

速度上限采用 `min(1.0, D/H, sqrt((a·T)^2 + 2a·D) − a·T)`。其中 `D=max(0,L−d)`，`L` 是剩余水平路程，`H=3.0 s` 是预留的前方运动时间，`a=0.5 m/s²`，`T=0.5 s`，终点余量 `d=0.15 m`。

| 剩余水平路程 | 合速度上限 |
|---|---|
| 3.15 m 及以上 | 1.000 m/s |
| 3.0 m | 0.950 m/s |
| 2.0 m | 0.617 m/s |
| 1.0 m | 0.283 m/s |
| 0.5 m | 0.117 m/s |
| 0.15 m 及以内 | 0 m/s |

新 `/pure_path` 缩短时立即应用更低上限，无需等待 SCAN 确认。延长时必须等 SCAN 确认，随后按 0.5 m/s² 加速。样条推进率随限速调整，平移指令统一缩放，保持 `vx:vy` 方向比例，并同时满足合速度及两轴上限。原始或已确认路径为空、坐标系不匹配，或机身与任一参考投影点的高度差超过 0.5 m 时停止平移。水平起点吸附仍沿用已有盲区接入策略，不增加横向偏离停车门槛，也不将盲区接入距离计入已规划的前方路程。

参数位于 `a1.yaml` 的 `closed_loop_controller` 中，包括 `reference_speed_preview_time`、`reference_speed_acceleration`、`reference_speed_reaction_time`、`reference_speed_stop_margin`、`reference_height_tolerance` 和固定的 `reference_body_height`。`/scan/planning/reference_speed` 最高 10 Hz 发布 JSON，包含两份参考的剩余路程、有效路程、速度上限、实际 `vx/vy`、合速度和限速原因。控制器日志每秒记录剩余路程及当前限速。

路径长度仅表示当前可用参考的长度，不能单独证明观测充分或支撑可靠。隔离域原生控制器测试见 `tests/verify_reference_speed_ros.py`，覆盖待确认短路径、延长确认、斜向合速度、空路径和错误高度层。实机失足率需要另行验证。

## 显式选择独立轻量样条后端

模式 4 的 `light` 从当前机身位姿朝人工目标，在分层图上用一次 Dijkstra 搜索 3 m 局部候选。在重力水平面按 8 个方向保留可达代表路径，再按 8×Z 进展 + 2×XYZ 进展 − 路径 cost 排序。滚动时按正在执行的样条弧长锁定剩余近端约 1/3，在固定交接点向前最高 2 Hz 刷新远段。新旧 XYZ 五次 B 样条在交接点匹配位置、速度、加速度和 jerk。控制器走完近段后才接入新样条。

搜索屏蔽膨胀占用图上冲突的有向边。沿选中路径按不超过 3.5 cm 的水平弧长采样，保留支撑顶点及高度。根据水平弧长拟合 XY 样条，统一使用 0.75 m/s 的速度上限、0.70 m/s² 的加速度上限和 1.20 m/s³ 的水平 jerk 上限。

XY 时间确定后，在同一时间轴上独立求解机身高度。机身参考高度由支撑面加固定的 0.40 m 配置偏移得到，起始状态取机身位姿。高度解受支撑面、观测净空和 0.25 m/s³ 的 Z 向 jerk 上限约束。原支撑层用于连续性检查，完整三维样条用于机身占用检查。样条切角遇到占用时加强对应时段的 XY 参考权重，最多重拟合两次。高度不可行时尝试其他候选。

从静止起步、近身缺少支撑时允许同层接入。`--start-blind-radius` 默认 0.70 m，最大高差 0.07 m。接入段也采用整条轨迹的速度分配。模型输出保持原始 cost，占用检查覆盖整个起步段。

```bash
cd /home/cjh/Pure3D_ws
./Entrance/shell/total_terminal.sh --flight-type 4 --planner-backend light
```

显式指定 `--planner-backend light` 时，执行链使用独立的 `ego_local_trajectory_server`。它按 EGO 轨迹服务器的方式对已提交样条采样，再将世界系平面速度转换到机器人航向系。跟踪滞后或转向时连续降速，保留正的轨迹推进率。A1 配置下，常规水平指令按 0.70 m/s² 限制变化，并提前过渡到下一段较低的上限。三维占用检查覆盖入队、交接和执行，碰撞或输入失效立即停车。交互目标在释放鼠标时提交。

RViz 中蓝色显示候选，金色显示已激活样条。`/pure_path` 用于显示已选参考，轨迹服务器通过 `/scan/planning/scheduled_spline` 接收排队轨迹。`status.json` 显示 `trajectory_backend=support_spline`，`events.jsonl` 记录拟合、交接和拒绝。修改了 ROS 消息的工作区需先编译仿真 overlay。

```bash
./Entrance/shell/total_terminal.sh --flight-type 4 --planner-backend light_scan  # Light 候选 + 原 SCAN 后端
./Entrance/shell/total_terminal.sh --flight-type 4 --planner-backend astar       # 单目标 A* + 原 SCAN 后端
```

下文是使用原 SCAN 后端时的接口说明。样条算法与保存点云对照见 [轻量支撑面 B 样条](../../../planner/docs/SUPPORT_SPLINE.md)。

输入起点始终是规划快照时的当前 TF `world←base`，终点直接来自 `/scan/goal`，按消息 frame 变换至 world。不会使用 SCAN 轨迹的起终点、local target 或前视点作为 A* 查询端点。

```text
/entry/local/cost_cloud ─┐
world←base ─────────────┼→ planner.surface.SparseSurfacePlanner → /pure_path
/scan/goal ─────────────┘                                    ↓
                     SCAN bridge → /scan/initial_path → SCAN 模式 4（模式 3 可显式接入）
                                             → /scan/planning/bspline → /cmd_vel
```

不使用 `/pct_path`。`/entry/astar_path` 与 `/pure_path` 的实际 XYZ 和点序一致。青色 Marker 仅为查看增加 0.02 m 显示偏移，不改变路径消息。

代价输入为 `/entry/local/cost_cloud` 的 `trav_cost`。在线 KNN 在编码前去除孤立上下文点并重新预测，不对 cost 乘 keep_score。没有 GRU、历史 cost 融合或 GPMP。模式 4 采用严格 `0 <= trav_cost < 0.8`。`cost == 0.8` 也不进入搜索图，零 cost 合法。保留多层匹配与缺测断开规则。沿用 planner 默认吸附上限，起点 1.0 m、终点 0.5 m，吸附不代表接入段已通过预测 cost 检查。cost 缺失、过期、没有合格支撑或 A* 不连通时，模式 4 清空参考并停车，记录 `cost_blocked`，不再把 goal 直接交给 SCAN。没有补点或为 A* 跨未知连边。SCAN 局部优化沿用自身占用图，0.8 上限只约束 A*，不检查 SCAN 优化后轨迹。

## 与当前仿真独立启动

选择 `astar` 可按人工目标触发单次 A*，然后由原 SCAN 跟踪。该模式保留人工 interactive marker。

```bash
cd /home/cjh/Pure3D_ws
./Entrance/shell/total_terminal.sh --flight-type 4 --planner-backend astar --cost-limit 0.8
# 原始 SCAN 人工目标模式仍可选择：
./Entrance/shell/total_terminal.sh --flight-type 1
```

独立仿真可用 `ros2 launch Entrance/launch/sim/gazebo.launch.py planner:=scan flight_type:=4 cost_planner_backend:=astar cost_astar_limit:=0.8`。传感器、Faster-LIO、cost 仍按独立入口运行。`astar` 只在新目标位置变化时计算一次 A*。同一目标的重复消息、机器人移动和 cost 更新均不重复发布路径。

已有 Gazebo、Faster-LIO、局部 cost 后，使用模式 4 的独立 SCAN 入口：

```bash
source Entrance/shell/_internal/sim_env.bash
ros2 launch scan_planner workspace.launch.py flight_type:=4 cost_planner_backend:=astar cost_astar_limit:=0.8
```

只启动这一组桥、SCAN、控制器和协调器，不能与已有同名节点叠加。`cost_astar_output:=/absolute/new_directory` 可固定规划与执行日志位置。模式 4 自动启用 `/pure_path`、0.1 m 路径采样及人工 marker，并关闭直接 goal 回退。规划使用 KNN 编码前过滤后重新预测的 `/entry/local/cost_cloud`。每次查询记录实际 `cost_limit` 与 `cost_gate`，避免显示阈值和搜索阈值混淆。

模式 3 的原有显式接入方法保留，下面的独立协调器命令只适用于模式 3，模式 4 已自动启动协调器，不能重复执行：

```bash
cd /home/cjh/Pure3D_ws
source Entrance/shell/_internal/sim_env.bash
ros2 launch scan_planner workspace.launch.py  flight_type:=3 manual_markers:=true  path_topic:=/pure_path path_sample_distance:=0.1 goal_fallback:=true
```

若从运行中的 SCAN 切换，应先保存三个旧节点的实际参数，并仅替换 `scan_workspace_bridge`、`scan_planner_node`、`closed_loop_controller`。使用 `params_file:=/absolute/original_scan_parameters.yaml` 保留原感知盒、机身碰撞和控制配置。已有这三个节点时不能再叠加启动另一组。Gazebo、LIO、cost 生命周期独立。

`workspace.launch.py` 原有默认行为保持，以上三个参数仅显式改变本次入口。当前试验感知盒为 6×6×5 m。桥接采样间距使用 0.1 m，避免默认 0.8 m 采样丢掉 A* 的局部转弯。

另一个终端：

```bash
cd /home/cjh/Pure3D_ws
ros2 launch Entrance/launch/sim/follow.launch.py output:=/absolute/new_output_directory
```

默认沿用现有 ROS 域。总入口使用 `ROS_LOCALHOST_ONLY=0` 时，该终端也需要相同设置。拖动现有 `end_pos` interactive marker 后，桥发布 `/scan/goal`。协调器订阅 `/scan/goal`。A* 成功时发布 `/pure_path`，失败或 cost 暂时不可用时，向 `/scan/fallback_goal` 发布原始 goal，由 SCAN 自身占用图规划。同一 goal 不自动重试，移动 goal 后再计算新的 A*。此功能由 `goal_fallback:=true` 显式启用，原有入口默认不变。

## 交接与停车

- goal 位置相对上次已接收位置变化至少 1 cm 后，使用当前 TF `world←base` 和当前有效 cost 快照计算一次 A*，成功时只发布一次 `/pure_path`。方向单独变化、重复目标、base 移动及 cost 点云更新都不触发 A*。网络 cost 预测频率保持原设置。
- 并发最多一个任务，快速修改 goal 时合并尚未开始的目标，旧目标结果不发布。每个目标只尝试一次，模式 4 没有可用快照、支撑或路径时停车。仅模式 3 显式开启回退时会直接交给 SCAN。cost 恢复不自动再发 A*。
- SCAN 接受一次全局参考后，持续执行自己的局部重规划和跟随。参考初始化使用当前位置、当前里程计速度、零起始加速度和零终点速度。SCAN 确认接受后解除停车，新目标交接时短暂保持。`/scan/planning/reference_accepted` 与 `/scan/planning/fallback_accepted` 按时间戳确认，跨来源延迟消息不能覆盖较新命令。
- 模式 4 无 A* 时停车。模式 3 显式启用的直接 goal 回退仍经过 SCAN 原生局部搜索、占用避障和轨迹优化。TF 失效、SCAN 未就绪、交接超时或发布者冲突时停车。
- `/pure_path` 与 `/entry/astar_path` 保存相同机身坐标点。桥减去 0.4 m、SCAN 再加回 0.4 m，净偏移为零。无有效 A* 时清除青色显示。
- 到达人工 goal 0.2 m 内且速度小于 0.1 m/s，持续 0.5 s 后记录到达并停车。吸附路径结束不等于已到原始 goal。
- `/entry/navigation/hold` 持续发送心跳。节点退出或失联 2 s 后控制器停车。返回原手动 SCAN 模式时，重启控制器以清除导航心跳状态。

## 查看与保存

`/entry/local_astar_follow/status` 和 output/status.json 保存状态、hold、错误、goal 编号、base TF 时间与 SCAN 模式。output/execution.json 保存交接及执行事件，odometry.json 保存实际轨迹。planner/light_planner/queries 保存查询端点、点云指纹、A* 路径和分阶段耗时。在线输入点云沿用 cost 模块存档，不按周期重复存储。events.jsonl 逐条追加完整事件，execution.json 保留近期事件。status 保存 source、`refresh_mode: goal_position_change`、A* 与回退的发布次数及地图准备加搜索的 P50/P95。

青色是 A* 全局参考，原 SCAN 黄红色 Marker 是局部轨迹。局部优化依旧使用 SCAN 自身占用约束，当前没有将网络 cost 直接加入 B-spline 目标函数，因此不保证局部轨迹严格保持在低 cost 单元内。日志分别统计局部轨迹与 A* 参考的偏差和实际到达结果。
