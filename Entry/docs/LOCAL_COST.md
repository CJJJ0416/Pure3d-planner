# LIO 独立感知盒与定时局部 cost

本文中的对照实验命令依赖本地保留的实验模块，该模块不随源码发布。正式在线入口见 [Entry](../README.md)。
局部输入由 `/faster_lio/cloud_registered` 逐帧增量维护，不读取累计 PCD。点云经完整 `world ← camera_init` TF 转为重力对齐 world。Entry 独立订阅已修正安装外参的 `/faster_lio/Odometry`，将 body 位置转换到 world，以该位置维护 **6×6×5 m**、world 轴向的盒子。位置按 0.05 m 网格定位，任一轴变化达到 0.2 m 后滑动。边界由 `/entry/local/box` 发布，不再订阅或依赖 SCAN 的可视化感知盒。

尺寸和有效期配置位于 `Entry/configs/local.yaml`：`box_size_m`、`box_shift_m`、`max_pose_age_s`。SCAN 自身感知尺寸仍独立位于 `a1.yaml`，规划视野保持 **1.5 m**，网络 core 保持 **4×4×3 m** 和 **1 m 上下文**。

C++ 稀疏哈希按 world 原点、0.05 m 体素保存首个真实 XYZ。默认 `context_mode: stable_core` 将 6×6×5 m 输出 ROI 与编码缓存分开：按 ROI 涉及的全部世界 core 向外扩展 1 m halo 保存已观测点。ROI 移动时保留共同 core 的上下文，删除缓存边界之外的点。移回已被淘汰的区域需要重新观测。最多 500000 体素，超限后 `capacity_rejections` 和 error 报告错误并清空输出，需提高预算后重启，避免用截断输入继续发布。不附加几何通行性判断、不膨胀、不填充未知。当前静态场景未实现射线空闲清理。

**本模块已实现局部 cost 生成、发布和显示；SCAN 优化器尚未订阅此代价。** `flight_type=1` 当前依靠 SCAN 原有占用/轨迹优化；`flight_type=3` 跟随 `/pct_path`。总入口模式 4 由 KNN cost A* 发布 `/pure_path`，失败停车，SCAN 保持自身局部优化。独立局部入口不恢复时间 node，也不改变原全局导航接口。

## 独立启动

仿真、键盘、LIO、局部预测分别启动，使用同一 ROS_DOMAIN_ID。先让机器人站稳，再启动 Faster-LIO。

各仿真入口默认统一为 `ROS_DOMAIN_ID=0`、`ROS_LOCALHOST_ONLY=1`，显式设置的环境变量保留。这样从不同新终端直接运行脚本也能互相发现。

根目录新增独立入口 `./run_cost.sh`，默认 Tiny best49，附带耗时监测，启动前只清理本工作区、同一 ROS 域的旧局部 cost／GPU worker／监测进程。Gazebo、MID360、Faster-LIO 和 keyboard 保持独立，原入口不改。当前 Gazebo 的 MID360 随 `run_gazebo.sh` 启动，第三个独立终端使用 `run_faster_lio.sh`。

```bash
cd /home/cjh/Pure3D_ws
./run_cost.sh                  # 只启动 cost 与耗时监测
./run_cost.sh --rviz           # 可选：另开局部 cost RViz 布局
./run_cost.sh --model small    # 可选：Small best56
```

在 cost 终端按 Ctrl+C 只退出本次 cost 与监测。当前 cost 目录记录在 `outputs/entry_local/current_cost_run.txt`，包含 `local/`、`monitor/` 和日志。新入口降采样显示最高 1 Hz，网络 core 与条件保持原值。原 `Entry/launch/local.sh` 继续可单独使用，保留其默认 Small 配置。

## 四终端顺序启动

```bash
cd /home/cjh/Pure3D_ws
./total_terminal.sh
```

总入口先清理本工作区、同一 ROS 域的旧仿真与配套进程，保留数据和日志，然后依次打开 **gazebo、keyboard、lio、cost** 四个独立 GNOME Terminal 窗口。没有单独的统一停止脚本，启动结束后总入口退出，各终端独立运行。关闭 cost 终端不退出其他模块。下次执行总入口会先清理旧会话。

默认顺序：run 的四个控制器及里程计就绪 → 等 3 秒 → keyboard → 蹲姿 3 秒仿真时间 → 站立 4 秒仿真时间 → RL 3 秒仿真时间 → Faster-LIO → 注册点云及 world TF 就绪后再等 3 秒 → trav_cost 与耗时监测、RViz。总入口只发布模式键 `2、2、7`，导航目标由你操作 `end_pos`。

```bash
./total_terminal.sh --run-delay 5 --stand-delay 5 --lio-delay 5
./total_terminal.sh --dry-run   # 只打印顺序与间隔，不清理、不启动
```

间隔和就绪检查同时生效。控制器、LIO 或 cost 未就绪时不会仅因 sleep 结束就进入下一步。PCD 自动保存显式为 false，当前感知盒 6×6×5 m、LiDAR 高度 0.18 m。若启动中途失败，只清理本次未完成启动产生的进程。数据、日志和启动状态位于 `outputs/entry_local/terminal_*`。

## 原有独立终端命令

```bash
cd /home/cjh/Pure3D_ws
./run_gazebo.sh planner:=scan flight_type:=1 rviz:=false
```

其他终端：

```bash
cd /home/cjh/Pure3D_ws
./keyboard_gazebo.sh -p publish_cmd_vel:=false
```

```bash
cd /home/cjh/Pure3D_ws
./run_faster_lio.sh
```

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/local.sh --model tiny --rviz
```

默认模型仍是原 Small best56；`--model tiny` 选择配套 Tiny 解冻 best49 和 head，条件均为 `[.5,.4,.4,0]`。可用 `--mode cloud` 只维护点云、不加载 GPU。局部入口不启动仿真、不切换机器人状态；退出仅等待当前局部任务并保存有限点云，不触发整图补算。

## 定时与接口

### 在线 KNN 编码前过滤

`local.yaml` 默认启用已保留的保守 KNN：16 近邻均距超过全体均值+2倍标准差，并且 0.15 m 内其他邻居少于 3 个时删除。使用已验证等价的最小统计实现，不计算未用于决策的64近邻和0.5 m计数。参数在 `knn_filter`，设 `enabled: false` 后重启 cost 可恢复原输入方式。

过滤在接收端的后台推理线程执行，不阻塞 ROS 回调。原始累计地图保持完整，仅当前推理快照的编码上下文与ROI查询删除相应点。`/entry/local/cost_cloud` 是过滤后重新编码得到的真实 cost，不使用分数乘积。`/entry/local/cloud` 仍为原始ROI几何，两者点数可能不同。相同上下文内容复用过滤掩码，ROI移动时重新提取保留查询点。上下文变化时重新计算KNN。

`/entry/local/knn_removed` 发布当前预测版本的删除点，最多2000个显示代表点。RViz 的 KNN removed observations 默认关闭，可手动启用。过期清空与预测有效期沿用原逻辑，过滤后ROI为空时发布空cost并标记无效。SCAN占用图不因本功能而改变。

status/预测报告增加 `knn_filter`、`cost_points` 与 `filtered_query_revision`，阶段耗时增加 `knn_filter_s`、`knn_statistics_s`。`pipeline_s` 和完整回调到发布延迟包含KNN，worker内部编码计时不含KNN。退出另存原始预测XYZ/上下文与对应保留掩码，便于恢复逐点关系。此开关由局部节点执行，独立调用GPU worker的离线工具不会自动筛点。

离线审计已发现少量稀疏楼梯点可能被删，详见 [KNN_FILTER_COMPARE.md](KNN_FILTER_COMPARE.md)。当前按用户接受的参数启用，保留过滤前数据和删除点以便复查。

`Entry/configs/local.yaml` 的 `period_s: 0.25` 是最短请求间隔，不保证 4 Hz 推理。GPU 常驻独立进程，接收端继续接收；最多一项任务在算，中间地图版本不排队、不复制，完成后取最新内存快照。未变化的 core 上下文复用精确输入哈希缓存，ROI 边界本身不使输入相同的 core 失效。worker 保留仍属于查询 core 范围的缓存，淘汰移出范围的 core。

沿用训练用 4×4×3 m core、1 m context，保留原来的缩放、按观测包围盒居中和最低 Z 归一化。完整编码点集以世界体素顺序保存，ROI 查询行通过索引传递，只对包含查询点的 core 编码。halo 仅参与编码，不进入输出。这里的完整上下文指空间范围内已观测点，未知区域不补点。新增真实观测仍会改变上下文、触发重算，稳定版不强制固定 cost。`context_mode: roi_clipped` 可切回原先的 ROI 裁剪方式。

原有 `revision` 为完整请求版本，另有 `query_revision` 和 `context_revision` 区分查询与编码集合。status 记录 ROI/上下文点数、两种边界及 `phase_timing_ms` 的阶段 P50/P95。halo-only 扫描不刷新 ROI 有效观测时间。退出时 `latest_geometry.npy` 为 ROI，`latest_context.npy` 为完整缓存，`context_geometry.json` 保存版本和边界。预测仍使用自身快照 XYZ，不冒充最终几何。

## 原版与稳定上下文离线对照

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/compare_context.sh --output /tmp/entry_context_comparison --rviz
```

默认使用已有 26480 点世界坐标快照，先按原始 ROI 保存已观测上下文，再沿 X 移动输出盒 0.4 m。原版只使用新 ROI，稳定版保留共同 core + halo，两个结果使用完全相同的查询 XYZ。可指定 `--input /absolute/world_xyz.npy`，可用 `--center X Y Z` 指定初始盒中心。优先从输入目录 status.json 的边界推导中心，没有该记录时使用输入 XYZ 包围盒中心。

每路顺序预热 3 次并强制重算 10 次，另测 3 次缓存查询。还执行固定输入重算、仅删除外围实测点的干预，以及以 0.2 m 步长移动输出盒的保留上下文实验。高度裁剪单独标记为压力实验。只改变外围时选定核心查询 XYZ 不变，保留上下文实验则验证共同点与共同 core 输入签名。输出 comparison.json、REPORT.md、原版/稳定版 cost、查询与上下文 XYZ、共同显示行以及模型校验值。

```bash
./Entry/launch/compare_context.sh --input /absolute/world_xyz.npy --output /tmp/entry_context_large
./Entry/launch/compare_context.sh --view-only --output /tmp/entry_context_comparison
```

对照入口默认强制使用隔离域 186、localhost=1，可通过 `ENTRY_COMPARE_ROS_DOMAIN_ID` 单独覆盖。RViz Fixed Frame 为 world，两个图层 `/entry/context_compare/original` 和 `/entry/context_compare/stable` 可切换，默认只显示稳定版。两路共用相机、XYZ 行和 0–1 绿红色标。共同显示行按两路 cost 的较大值选取每个 0.1 m 体素的原始代表点，最多 50000 点，静态结果仅发布一次，避免重叠渲染和独立采样造成差异。离线不发布传感器、时钟、目标或控制指令。

离线基准记录上下文快照、查询提取、编码、head、拼接和 NPY 传输耗时，不包含 ROS 扫描接收等待、DDS 发布传输或 RViz 渲染。在线仍只运行稳定版，完整每帧回调至发布延迟使用原来的 latency 口径。扩大上下文可能增加耗时，稳定性与准确率分别评价。当前阶段不启用 GRU、历史 cost 融合或近身 cost 置零。

## 在线话题与过期处理

原始 `trav_cost`、`keep_score`、乘积与独立有效掩码的离线对照见 [SCORE_COMPARE.md](SCORE_COMPARE.md)。没有 Filter 权重时可显式模拟评分，模拟入口不改动本节在线接口。

| 话题 | 内容 |
|---|---|
| `/entry/local/box` | LIO 独立维护的 world 轴向边界 |
| `/entry/local/cloud` | 盒内 LIO 代表点，world |
| `/entry/local/cost_cloud` | XYZ + `trav_cost`，world |
| `/entry/local/status` | 接收、点数、版本、边界、缓存和分阶段耗时 |
| `/entry/local/rviz/cloud` | 降采样几何 |
| `/entry/local/rviz/cost_cloud` | 降采样 cost |

RViz Fixed Frame 为 world，默认 0.1 m、最多 30000 点、最高 1 Hz，以 Points 显示。cost 每显示体素保留最高代价原始点。布局同时保留 SCAN 青色占用和边界框。SCAN 占用可视化话题带高度裁剪，膨胀话题含人工扩展点，两者均不作为模型输入。

调度和过期判断使用单调时钟；`/clock` 停止也会继续执行。`max_input_age_s` / `box_timeout_s` 均为 1 s，`max_result_age_s` 为 2 s；过期时清空局部 cost，status 的 `valid=false`。保存的预测 XYZ/cost 保持其自身版本，不冒充新几何。

收到扫描但没有盒内有效观测（包括全在盒外、无效点或重复时间戳）不再刷新观测新鲜度。`scans_without_in_box_points` 记录这类没有新增有效观测事件的扫描，避免定位发散后旧盒内缓存长期被标记为新数据。

`prediction_history.jsonl` 的 `latency` 使用同一主机单调时钟，分别记录每帧回调至 cost 发布、最新一帧年龄、纳入时间、等待快照、快照创建、提交后计算和打包发布。`pipeline_s` 只表示提交快照至取回结果，不包含提交前等待。2026-11-15 两个 1.5 m 往返目标补测：161 帧、74 次预测，完整每帧中位数 **383.8 ms**、P95 **654.9 ms**；同一轮旧口径中位数 178.7 ms。起点是收到 LIO 注册点云，原始雷达/Faster-LIO 与 RViz 显示另计，详见 `docs/reports/entry_local/e2e_box4_20261115_190849/REPORT.md`。

## 人工目标与轨迹更新监测

保持 `flight_type:=1`，在 RViz 的 **Interact** 工具下拖动 `/basic_controls` 的 `end_pos`。目标由操作者发布，监测器不发布目标或控制指令。独立终端启动监测，本次 run 必须包含 `local/prediction_history.jsonl`：

```bash
cd /home/cjh/Pure3D_ws
source common/unitree_gazebo/sim/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
export PYTHONPATH="/home/cjh/Pure3D_ws:${PYTHONPATH:-}"
/usr/bin/python3 -m Entry.runtime.monitor_local --run "$(cat outputs/entry_local/current_run.txt)"
```

`monitor/summary.json` 每秒更新完整链路中位数、P95、实际 cost/轨迹发布率、几何与预测版本、缓存与重新编码耗时。监测从启动后的新预测开始统计，`monitor/events.jsonl` 记录人工目标、轨迹以及其后首个更新输入快照的 cost。轨迹后 1 秒无新快照时，分别记录“几何未变且缓存有效”和“未及时得到新 cost”，不会把两者混同。这里的轨迹间隔来自消息到达，不是 SCAN 优化器的计算时间。

RViz 的 **Local cost timing** 在机器人上方显示统计。完整每帧延迟包含等待快照，静止且几何不变时这项等待可能增长，因此另存 `geometry_changed_frame_to_cost_ms`，以及最新输入帧延迟和缓存复用情况。1 Hz 或 2 Hz 的降采样显示不代表后台 cost 发布频率，也不代表每次轨迹更新都必须重新编码未变化的几何。

## 实测与检查

2026-11-16 同一份 74948 点实际 LIO 快照上，Tiny、相同 core/context、每档 3 次预热和 10 次强制完整重算：10×10×5 m 为 21 次编码、中位 603.8 ms，6×6×5 m 为 11 次编码、321.2 ms，6×6×3 m 为 11 次编码、311.4 ms。本轮据此选取 6×6×5 m，保留竖直上下文。该计时包含预处理、PTv3/upcast、head 和拼接，不含 ROS 接收等待、IPC 与发布。固定窗口跨界和重叠 context 决定实际调用次数，不能按盒子体积或单 core 时间线性估算。完整八档、token 数和方法见 `docs/reports/entry_local/box_selection_20261116_153448/REPORT.md`。单快照性能不代表其他位置或楼梯预测质量。

RTX 4060 Laptop 8GB，同一真实 LIO 快照，3 次预热、10 次完整重算：单 core Small 46.7 ms / Tiny 27.3 ms；6×6×3 m 为 438.5 / 238.7 ms；10×10×5 m 为 1125.9 / 610.3 ms（中位数）。包含预处理、网络、head 和拼接，不含 ROS/RViz；该离线对照保留完整 core 外扩上下文，输入大于当前严格盒内方式。不是闭环速度或相同精度的证明。细节见 `docs/reports/local_cost_latency_20261115/REPORT.md`。

在线同目标链对照：4×4×3 m 盒子运动任务区间 81 次预测，中位数 177.5 ms、P95 319.2 ms；原 10×10×5 m 为 431.6 / 571.3 ms，中位耗时减少 58.9%。三个目标均实际到达。小盒仍横跨约 7 个需重算的固定 core，不能将单 core 的 27.3 ms 当作整个移动盒耗时。规划视野同步缩短到 1.5 m。记录与导出的真实点云结果图见 `docs/reports/entry_local/box4x4x3_20261115_162503/REPORT.md`。

历史软件验收已完成，相关一次性测试脚本在 2026-11-17 源码清理中移除，已有报告保留在本机输出目录。

合成 ROS 验证检查 45° 输入 TF、移动盒剔除与交集保留、单任务调度、暂停 `/clock` 后清除过期 cost，不启动仿真、不发布控制指令。在线实际跟踪另行记录。

## 保存的 LIO 轨迹：局部 cost 回放

`Entry.mission.scanning_replay` 按原始帧顺序接入 `ContextVoxelMap` 和 `WorkerClient`，
复用正式 `stable_core`、5 cm 首个真实点代表、上下文签名缓存和真实模型权重。
滑动盒尺寸、5 cm 网格及 0.2 m 移动阈值读取 SCAN 的 `a1.yaml`，
盒中心使用统一到 map 的 LIO 估计 IMU 位姿。点云已经完成首帧固定变换，不再重复应用安装角。
原始扫描没有额外降采样。盒内 cost 查询采用 Entry 本身的 5 cm 体素地图。

以下例子只计算 55–85 秒的上楼片段。前 55 秒只维护历史观测，之后只用截至当帧的数据。

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Entry.mission.scanning_replay \
  --root datasets/simulation/isaac_lio_20261117/captures_exact_imu/route_001 \
  --model tiny --start-s 55 --end-s 85 \
  --output outputs/scanning/cost_replay_tiny_segment_20261117
```

计算结束后使用系统 ROS Python 打开 RViz：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=196 ROS_LOCALHOST_ONLY=1
export PYTHONPATH=/home/cjh/Pure3D_ws:${PYTHONPATH:-}
/usr/bin/python3 -m Entry.rendering.scanning_rviz \
  --replay outputs/scanning/cost_replay_tiny_segment_20261117
```

RViz 跟随传感器移动，显示真实局部预测、橙色感知盒、青色轨迹和过去的扫描。
独立控制窗提供播放、暂停、进度条、循环和倍速。背景显示使用 10 cm 显示代表，
保存的原始扫描不变，局部 cost 显示全部输出点。
这是因果输入上的实际推理快照回放，网络耗时单独记录，不把显示帧率当作在线推理速度。


## RViz 显示与退出记录

局部 cost 默认以 Points 显示，保留当前色标，显示上限 30000 点、0.1 m 体素、1 Hz。原始 LiDAR 显示默认关闭，可勾选 `Raw LiDAR - optional`。这些设置仅减轻 RViz 渲染负担，完整 cost 话题和推理输入不降精度。

cost 启动器的终端状态输出遇到 `Broken pipe` 后转入空输出，不会因此停止网络或监测进程。各模块的日志和 `session_status.json` 继续保存真实状态。RViz 的 OGRE 顶点缓冲区分配错误属于渲染故障，需与模型异常、Gazebo 接收 SIGINT 和退出阶段的强制终止分别检查。


## 规划阻塞时保持感知更新

Entry 的输入有效期只取决于 LIO 位姿和盒内注册点云。位姿默认超过 0.5 s 没有新时间戳即失效，注册扫描默认超过 1 s 即失效。仿真时钟停止或重复发送同一位姿时间戳不会延长有效期。`status` 保存 `box_source=lio_odometry`、`pose_age_s`、`pose_received/accepted/dropped`、`dropped_without_pose` 与独立的源点云接收年龄。

静止且几何版本不变时，每秒最多 2 次发布有效缓存，不清空后重新编码。`cache_refresh_history.jsonl` 记录缓存发布及其完整链路延迟，`prediction_history.jsonl` 只记录真实推理任务，监测分别统计，缓存发布不增加编码次数。

SCAN 的 `manager.replan_timeout_s` 默认 0.5 s，使用 steady clock。采样、内部 A*、碰撞重弹和 LBFGS 共用一次重规划预算，超时会丢弃候选并发布测量位置上的停车轨迹，随后返回执行器。`/scan/planning/replan_status` 发布阶段、预算、实际耗时和累计超时数，监测摘要保存最新记录。预算是循环与阶段边界上的协作取消，不使用线程强制终止或并发读写占用图。


## 与 SCAN 盒内数据的区别

两者都使用 world 轴向，但 SCAN 滑动地图由 `/scan/sensor_pose` 的雷达位置驱动，`scan_sliding_map` TF 则跟随 base。Entry 的盒子独立使用 LIO 修正后的 body 位置，因此边界会存在安装平移、体素定位和估计误差带来的差异。

SCAN 的 `/scan/cloud` 来自原始 `/livox/lidar` 经 TF 转换。其占用地图使用命中/穿透射线更新概率，`/scan/grid_map/occupancy` 输出占用体素中心，`occupancy_inflate` 还包含人为膨胀。两种占用显示都裁去高于雷达位置加 `vis_height` 的部分，当前该值为 0.3 m，显示不等于完整地图高度范围。

Entry 从 `/faster_lio/cloud_registered` 累积每个 0.05 m 体素内首个真实 XYZ，保留稳定已观测上下文。KNN 在编码前删除保守判定的孤立点，最终 cost 点云只发布 ROI 内保留的查询点。当前没有基于射线空闲证据删除历史观测，也不加入碰撞膨胀点。模型会使用 ROI 涉及 core 的额外 1 m 上下文，输出盒与编码输入范围分别定义。
