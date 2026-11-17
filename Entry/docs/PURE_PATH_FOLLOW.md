# 局部 cost 点云上的 planner A* → SCAN

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

模式 4 自动启动 cost A* 协调器，保留人工 interactive marker。总入口默认使用模式 4：

```bash
cd /home/cjh/Pure3D_ws
./total_terminal.sh --cost-limit 0.8
# 原始 SCAN 人工目标模式仍可选择：
./total_terminal.sh --flight-type 1
```

独立仿真可用 `./run_gazebo.sh planner:=scan flight_type:=4 cost_astar_limit:=0.8`。传感器、Faster-LIO、cost 仍按独立入口运行。模式 4 只在新目标位置变化时计算一次 A*。同一目标的重复消息、机器人移动和 cost 更新均不重复发布路径。

已有 Gazebo、Faster-LIO、局部 cost 后，使用模式 4 的独立 SCAN 入口：

```bash
source common/unitree_gazebo/sim/env.sh
ros2 launch scan_planner workspace.launch.py flight_type:=4 cost_astar_limit:=0.8
```

只启动这一组桥、SCAN、控制器和协调器，不能与已有同名节点叠加。`cost_astar_output:=/absolute/new_directory` 可固定规划与执行日志位置。模式 4 自动启用 `/pure_path`、0.1 m 路径采样及人工 marker，并关闭直接 goal 回退。规划使用 KNN 编码前过滤后重新预测的 `/entry/local/cost_cloud`。每次查询记录实际 `cost_limit` 与 `cost_gate`，避免显示阈值和搜索阈值混淆。

模式 3 的原有显式接入方法保留，下面的独立协调器命令只适用于模式 3，模式 4 已自动启动协调器，不能重复执行：

```bash
cd /home/cjh/Pure3D_ws
source common/unitree_gazebo/sim/env.sh
ros2 launch scan_planner workspace.launch.py \
  flight_type:=3 manual_markers:=true \
  path_topic:=/pure_path path_sample_distance:=0.1 goal_fallback:=true
```

若从运行中的 SCAN 切换，应先保存三个旧节点的实际参数，并仅替换 `scan_workspace_bridge`、`scan_planner_node`、`closed_loop_controller`。使用 `params_file:=/absolute/original_scan_parameters.yaml` 保留原感知盒、机身碰撞和控制配置。已有这三个节点时不能再叠加启动另一组。Gazebo、LIO、cost 生命周期独立。

`workspace.launch.py` 原有默认行为保持，以上三个参数仅显式改变本次入口。当前试验感知盒为 6×6×5 m。桥接采样间距使用 0.1 m，避免默认 0.8 m 采样丢掉 A* 的局部转弯。

另一个终端：

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/follow_local_astar.sh --output /absolute/new_output_directory
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

`/entry/local_astar_follow/status` 和 output/status.json 保存状态、hold、错误、goal 编号、base TF 时间与 SCAN 模式。output/execution.json 保存交接及执行事件，odometry.json 保存实际轨迹。planner/queries 保存查询端点、点云指纹、A* 路径和分阶段耗时。在线输入点云沿用 cost 模块存档，不按周期重复存储。events.jsonl 逐条追加完整事件，execution.json 保留近期事件。status 保存 source、`refresh_mode: goal_position_change`、A* 与回退的发布次数及地图准备加搜索的 P50/P95。

青色是 A* 全局参考，原 SCAN 黄红色 Marker 是局部轨迹。局部优化依旧使用 SCAN 自身占用约束，当前没有将网络 cost 直接加入 B-spline 目标函数，因此不保证局部轨迹严格保持在低 cost 单元内。日志分别统计局部轨迹与 A* 参考的偏差和实际到达结果。
