> 历史流程：当前在线方案已改为每次 goal 对完整累计 PCD 推理，参见 [EXPLORATION.md](EXPLORATION.md)。以下 node、EROI 和合并命令仅供复现旧实验。

本文中的对照实验命令依赖本地保留的实验模块，该模块不随源码发布。正式在线入口见 [Entry](../README.md)。
# 连续 VST 目标采集 → bag 建图 → goal 分 node → cost → 合并

当前推荐使用**短目标步进模式**：第 0 层完成 8 段移动，每次到达后静止 3 秒，再从当前 `base` 选择下一个距离约 1.5–3 m 的可通行新点；单段原生 VST A*/GPMP 路径上限 4 m。**不提前生成长目标链，不输入固定起点。** 原生 A* 省略起点的行为保留。

每个 node 的观测 EROI 保存其运动、停留与等待下一段规划期间的全部注册点云，按 node 分色；不做机器人附近裁剪。物理上重复看到的表面保留在各自 node 的观测中。早期四段长路线与 VST 对照作为历史结果保留，见文末。

各阶段独立顺序运行：先完成录包，再回放 Faster-LIO 保存 PCD，然后按 goal 分 node，逐 node 计算 cost，最后合并。回放阶段不启动在线 Entry 或模型。以下命令均从 `/home/cjh/Pure3D_ws` 执行；采集阶段保持同一 ROS 域，回放阶段统一使用隔离域 `184`。

## 采集：手动启动与状态切换

终端 1：启动当前场景，SCAN 使用模式 3 跟踪 `/pct_path`。

```bash
cd /home/cjh/Pure3D_ws
./run_gazebo.sh flight_type:=3
```

终端 2：保留键盘模式切换，让 SCAN 提供速度。

```bash
cd /home/cjh/Pure3D_ws
./keyboard_gazebo.sh -p publish_cmd_vel:=false
```

按 `2` 蹲姿，等待约 3 秒，再按 `2` 站立，等待站稳。暂不按 `7`。采集阶段关闭 Faster-LIO 与其他 `/pct_path` 发布器。

终端 3：初始化短目标任务。命令读取 `world → base`，此时不预选后续目标。

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh plan --stepwise --segments 8 --distance-min 1.5 --distance-max 3 --hold-seconds 3 --max-path 4 --output /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

默认场景为 `datasets/wfc_capability_dr_v2/scene_20261109/vst_nominal.pickle`，种子 `821111`，规划阈值 `15`，每段最多尝试 100 次。短目标优先在当前朝向前方选择，并避开此前目标 1 m 范围；前方没有候选时允许转向寻找附近新点。A* 支撑高度、GPMP 高度和采样代价均检查；机身路径为支撑面加 `0.4 m`，不叠加 GPMP 高度偏置。空间切片按物理支撑面判断楼层。

终端 4：开始录包，站稳静止录制至少 **10 秒仿真时间**。

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh record --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

录包默认保存至该任务目录的 `bag/`。包含 `/livox/lidar2` 原生 CustomMsg、`/livox/lidar`、`/livox/imu`、`/clock`、真值、TF、机器人描述、`/pct_path`、`/scan/goal` 和 `/control_input`。录包时启动 Faster-LIO 会终止本次录制并报告冲突，避免保存估计 TF。退出使用本终端的 `Ctrl+C`，等待 rosbag 写完元数据与检查报告。

静止时间足够后，在键盘终端手动按 `7` 进入 RL 模式。终端 3 开始逐段跟踪：

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh follow --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

`follow` 等待录包、初始静止时间与路径订阅者就绪。实测位置距目标 ≤`0.2 m` 并持续 `0.5 s` 后，使用现有 P 事件软暂停，等待机器人稳定并静止 3 秒，随后才从最新 `world/base` 位姿选择下一个可通行新点。规划完成后只发布一次可靠、持久化的机身路径与 `/scan/goal`，再恢复移动。每段超时为 `max(90 s, 3×路径长度/0.35 m/s)`，使用仿真时间。

路径规划期间 base 移动超过 `0.3 m`、里程计停止、录包停止、出现其他路径发布者或目标超时都会停止任务并保留失败原因。短目标模式只调用现有 P 软暂停/恢复事件，不切换蹲姿、站立或 RL 状态。结束时保持暂停；中断后使用新任务目录重新采集。

8 次实测到达与停留结束后，继续静止录制至少 **5 秒仿真时间**，再在录包终端 `Ctrl+C`。检查录包：

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh validate --bag /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01/bag --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

检查包括话题/类型、传感器时间单调与中断、前 10 秒静止真值、初始静止 IMU 窗口、最终扫描的 IMU 覆盖、四段目标/到达记录、后 5 秒静止和 TF 支路。缺少 `/livox/lidar2` 的旧 bag 不符合本接口，不能仅用 `/livox/lidar` 的话题名替代。

## 第二步：回放 Faster-LIO，先保存完整 PCD

终端 A：只启动 Faster-LIO 与 TF 桥。bag 模式先做只读检查，然后直接等待回放，不运行在线 IMU 等待器，不启动 Entry 或播放器。

```bash
cd /home/cjh/Pure3D_ws
ROS_DOMAIN_ID=184 ./run_faster_lio.sh --bag /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01/bag
```

终端 B：确认 Faster-LIO 和 TF 桥就绪后，以默认 1 倍速回放，并存储注册扫描、位姿和 goal。此时不运行 `Entry/launch/run.sh`。

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh play --bag /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01/bag --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

`play` 默认域为 `184`，支持 `--rate`。只回放传感器、记录的 `/clock`、真值/TF、机器人描述和段目标；不回放 `/pct_path` 或 `/control_input`，不生成额外时钟。预检拒绝 bag 中的 `map/camera_init/body` 估计 TF，回放重建 `world → map → camera_init → body`，其中 `map → camera_init` 严格为单位变换。

bag EOF 后，`play` 按墙上时间继续等待 Faster-LIO 的点云和机身里程计处理到最后一帧扫描末时刻，再等待写盘完成。保存 `offline/reconstruction/map.pcd`、`map.npy`、逐帧扫描及时间/点索引、机身位姿、goal 和 TF 锚点。完成后在终端 A `Ctrl+C` 退出 Faster-LIO。此时尚未分 node 或加载 GPU 模型。

`reconstruction.json` 和 `offline/play.json` 均为 `complete` 才进入后续阶段。若排空超时或中断，已接收数据保留为 `incomplete`，不冒充完整建图。每次重跑使用新输出目录，可通过 `play --output <新目录>` 指定；后续 `nodes/cost/merge/status` 使用相同 `--output`。

## 第三步：按 goal 到达区间分 node

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh nodes --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

使用 `execution.json` 的路径发布、实测到达与停留完成记录分段。`n0` 保存初始静止期间的全部注册扫描；`n1…n8` 从各自路径发布开始，保存运动、到达后停留及等待下一段规划的全部扫描，到下一次路径发布时结束；最后静止尾段归入 `n8`。节点位姿取到达时刻附近的 LIO 机身位姿并表达为 `map`。每节点保存 PCD/NPY、父节点、goal、运动与停留时间、帧/点范围及 `eroi.json` 的空间块集合。

每个原始有效点只属于一个节点，所有节点按顺序拼接，点数、XYZ 与原始累计 PCD 一致。目标、节点位姿以及空间分块都转换或表达为 `map`；原始目标的 `world` 坐标同时保留。

## 第四步：逐 node 计算 cost

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh cost --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

此命令不依赖 ROS 时钟、Gazebo 或 Faster-LIO。Small best56 只加载一次，依次输出 `offline/node_cost/n*/prediction.pcd`。**离线 cost 默认 `--inference-frame world`**：先按已保存的完整 `world → map` 变换把 XYZ 变换到重力对齐的 `world`，再按固定 `4×4×3 m` core 与 `1 m` 上下文分块、采样和编码。输出 PCD 始终保留原始 `map` XYZ 和点序；`map → camera_init` 的单位 TF 保持不变。

node 负责时间区间与原始点归属；上下文来自已经完成的最终几何，每个 node 只查询自己拥有的点，避免时间分段截断空间邻域；相同空间块复用缓存。`--inference-frame map` 可复现早期倾斜输入作为对照。两者改变的还包括对应空间网格和上下文，不将差异仅归因于独立旋转操作。这项默认修改针对本页的离线 cost 命令；独立在线 Entry 的输入约定见其 README。

## 第五步：合并节点结果

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh merge --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

输出 `offline/merged/prediction.pcd`（XYZ + `trav_cost`）和 `cost.npy`。合并前验证节点范围无重叠/空洞、节点与地图版本一致、每个预测 PCD 的 XYZ 和点序正确、cost 数量及数值有效。上下文点不参与重复拼接；缺少任何节点结果时不标记合并成功。

保存结果可独立用 RViz 打开，不需要再次回放 bag：

```bash
cd /home/cjh/Pure3D_ws
ROS_DOMAIN_ID=184 ./Entry/launch/mission.sh view --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01 --rviz
```

显示 cost、空间块、goal 节点和 LIO 实际轨迹；重力对齐版本默认 Fixed Frame 为 `world`，数据仍通过 TF 由 `map` 表达。默认最多显示 100 万个原始点的均匀子集以保持交互速度；保存的 PCD 和 cost 仍保留全部点。可通过 `--max-points` 调整显示上限。

## 每个 node 仅用自身观测：cost head 与 VST 对照

独立 EROI 对照使用 `--context-source node`，不读取其他 node 的点作为编码上下文。此前的最终地图上下文版本保留在 `node_cost/` 和 `merged/`；新增结果使用 `eroi` 后缀。两种方法均先把同一份 LIO node 点云变换到重力对齐 `world`，不使用场景已有 nominal cost 代替 VST 重算。

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh cost --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_nodes_20261114/mission --context-source node --inference-frame world --variant eroi
./Entry/launch/mission.sh merge --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_nodes_20261114/mission --variant eroi
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Entry.experiments.node_compare run --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_nodes_20261114/mission
/usr/bin/python3 -m Entry.experiments.node_compare plot --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_nodes_20261114/mission
ROS_DOMAIN_ID=184 ./Entry/launch/mission.sh view --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_nodes_20261114/mission --compare-nodes --cycle-seconds 6 --rviz
```

本次这些计算已完成；前四条为复现流程，已有输出目录不会覆盖。再次计算使用新的任务/输出目录。查看保存结果只需最后一条，先关闭当前同域 Entry 查看器。

RViz 左侧为 cost head、右侧为原生 VST；每 6 秒轮播 `n0…n8` 与 `merged`。使用 `Interact` 点击节点球暂停轮播，点击 `MERGED` 查看拼接。`/entry/node_compare/select` 接收 Int32，`0…8` 选 node、`9` 选拼接、`-1` 恢复轮播。两侧使用相同 VST 支撑位置和 0–1 色标；`All raw node cost points` 可选层显示模型对原始全部表面的预测，默认层展示同位置对照。

每个 VST 支撑点仅在自身 node 的原始点中查最近邻，距离超过 `0.15 m` 不参与对照。模型条件 `[.5,.4,.4,0]` 对应 VST `[.5,.4,.4,.5,0]`，多出的参数是固定的 `interval_free=.5`。结果位于 `offline/vst_nodewise/`：逐 node 原生支撑 PCD、同位置 head/VST PCD、最近邻索引、逐 node CSV、报告与对照图。模型独立预测位于 `node_cost_eroi/`，原始点序合并位于 `merged_eroi/`。

VST 拼接按固定 `layer/x/y` 单元采用最新观测 node，保留各 node 独立输出；某 node 未看到一个单元，不会删除之前的观测。独立 node 的上下文与整张地图不同，不要求其预测与最终整图重算相等。此处指标表示与 VST 的一致性，不能解释为独立真值准确率。静态对照图为清晰展示，只显示实际轨迹周围 5 m、第 0 层支撑点；计算输入、保存 EROI 和 RViz 支撑对照未作这种局部裁剪。

本次 9 个 node 的 4,469,090 个原始点均完成独立预测。全部 node 共 310,383 个配对支撑点，归一化 MAE 为 `0.3721`，在 `15/50=0.3` 阈值下的一致率为 `60.53%`。模型减 VST 的平均代价为 `-0.3084`，存在较多模型低代价、VST 高代价的区域，详见 `vst_nodewise/REPORT.md`。

## 产物与完成状态

短节点 EROI 使用单独的观测颜色查看，区别于 cost 色标：

```bash
cd /home/cjh/Pure3D_ws
ROS_DOMAIN_ID=184 ./Entry/launch/mission.sh view --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_nodes_20261114/mission --nodes-eroi --cycle-seconds 4 --rviz
```

每 4 秒轮播一个 node。使用 RViz 的 `Interact` 点击 `n0…n8` 的球形节点，会停止轮播并显示该 node 的完整观测点云；左侧也可以独立勾选各 node 话题，重叠观测不会去重。`ACTIVE node EROI spatial blocks` 显示当前 node 涉及的所有 `4×4×3 m` 重力对齐空间块。这是按运动与停留区间组织的观测 EROI，不包含 EDEN 的未知增益计算或 frontier 提取。

`/entry/node_eroi/cloud/n*` 发布每个节点的全部注册点云；`/entry/node_eroi/select` 接收 Int32，`0…8` 选择节点，`-1` 恢复轮播。本轮每个节点均少于默认显示上限 100 万点，实际发布点数与保存点数完全一致，无局部裁剪。保存目录为 `offline/nodes/n*/cloud.pcd`、`cloud.npy`、`node.json` 和 `eroi.json`。

```bash
cd /home/cjh/Pure3D_ws
./Entry/launch/mission.sh status --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/short_mission_01
```

状态分别展示规划、实际执行、录制、bag 检查、回放建图、分 node、cost 与合并；规划成功不代表机器人执行成功。主要产物：

- `mission.json`、`planning_base.json`、原始 A*、GPMP、支撑面/机身参考 NPY、失败尝试、VST/场景/PCD SHA256。
- `executed_paths/`：每段实测 base、实际重新规划的原生轨迹；`execution.json`：发布、到达或失败事件。
- `capture.json`、`bag/`、`bag_check.json`：录包进程与消息完整性证据。
- `offline/lio/`：实际 LIO 参数、bag 检查和完整首帧 TF 锚点；`offline/play.json`：LIO 输出时间范围与 EOF 排空状态。
- `offline/reconstruction/`：累计地图 PCD/NPY、逐帧点索引、位姿、goal、TF 与地图版本。
- `offline/nodes/`：每个节点的几何和节点记录；`nodes.json` 为完整归属索引。
- `offline/node_cost/`：模型配置与 SHA256、每节点预测和块缓存使用记录；`cost.json` 表示逐节点计算状态。
- `offline/merged/`：最终完整 `prediction.pcd`、`cost.npy` 和 `merge.json`。`merge.json` 为 `complete` 才表示最终结果通过所有点归属与拼接检查。

VST nominal 教师参数单独保存在规划清单；Entry 使用 Small 解冻 best56 和 `[0.5,0.4,0.4,0.0]`。两套条件各自保留，不描述为相同模型条件。实际路线完成与地形 cost 准确率分别评估。

## 软件回归

历史软件验收已完成，相关一次性测试脚本在 2026-11-17 源码清理中移除，已有报告保留在本机输出目录。

集成使用隔离域 `184`、合成静止 bag、实际 Faster-LIO 和 best56 GPU；域被占用时拒绝启动。依次运行建图、停 LIO、分 node、逐节点 cost 与合并，并与最终地图全量重算比较，最大 cost 差要求 ≤`1e-3`。不启动 Gazebo、不发送机器人控制，不作为实际路线执行或预测准确率证据。

本次串行集成通过：107,118 个点分为 n0 + 4 个 goal 节点，XYZ/点序完整，逐节点合并与全量重算最大 cost 差为 0。报告：`outputs/entry_bag_mission/serial_pipeline_01/integration.json`。四段原生路径与逐段当前 base 重规划接口分别见 `native_final_validation/mission.json`、`follow_integration_01/report.json`；后者使用合成位姿，不代表机器人已经执行四段路线。

## 实际 Gazebo 四段执行（2026-11-14）

此节及其 VST 对照属于早期长路线记录，当前推荐的短节点执行见文末。

用户授权代为执行后，已在独立域 186 运行当前场景、站立与 RL/SCAN 跟踪，4/4 段均通过实测到达判定。原始 bag 为 881,380,026 字节、1,869 帧雷达，初始静止 12.744 秒、末尾静止 5.994 秒，完整性检查通过。随后关闭本次采集仿真，在域 184 独立回放 Faster-LIO，保存 1,867 帧注册扫描、8,062,879 个点，再依次执行分节点、cost 和合并。

产物位于 `outputs/entry_bag_mission/live_20261114/mission/`。5 个节点点数分别为 4,665、1,836,177、1,280,195、1,889,371、3,052,471；完整 XYZ/点序验证通过，真实地图逐节点合并与全图重算最大 cost 差为 0。best56 SHA256 为 `056a5e293b9cbf1ad3719c5f1a531605ff23e3c7728b21460b184ae31e453517`。节点处 LIO 与 Gazebo 真值位置差为 0.006–0.031 m。地形代价准确率未在本次运行中单独测量。

运行报告为 `docs/reports/entry_bag_mission/live_20261114/result.json`，最终 PCD 为该任务下的 `offline/merged/prediction.pcd`；RViz 原生截图为 `docs/reports/entry_bag_mission/live_20261114/rviz_cost.png`。可重新打开：

```bash
cd /home/cjh/Pure3D_ws
ROS_DOMAIN_ID=184 ./Entry/launch/mission.sh view --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission --rviz
```

## 重力方向修正与同输入 VST 对照

该次 `map` 继承 `camera_init`，与重力竖直轴相差 44.949°。早期 `merged/` 在倾斜坐标中编码，已保留为对照。修正版本将同一份 8,062,879 点变换到 `world` 后重新编码，另外保存为 `node_cost_gravity/` 和 `merged_gravity/`；XYZ/点序保持原始 `map` 版本，逐节点合并与重力对齐全图重算最大差为 0。

```bash
cd /home/cjh/Pure3D_ws
# 已有产物不覆盖；复现到新输出目录时可使用这些命令
./Entry/launch/mission.sh cost --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission --inference-frame world --variant gravity
./Entry/launch/mission.sh merge --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission --variant gravity
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Entry.experiments.compare vst --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Entry.experiments.compare measure --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission
```

VST 也直接使用这份 LIO 观测，不使用完整场景的既有 nominal 地图代替。依次对累计至 `n0…n4` 的观测运行原生 tomography 与 cost，使用固定公共网格；每个节点保存新增/更新单元与删除 ID，以新版本更新重叠单元、删除失效单元。最终拼接得到 202,604 个原生 VST 支撑点，与最终全部观测的原生 VST 快照一致。VST 采用 `[clearance=.5, slope=.4, step=.4, interval_free=.5, stand_ratio=0]`，与模型四维条件对应，区别于原路径规划 nominal 的 step=.3、stand_ratio=.1。

模型仍预测全部原始点；VST 输出分层支撑面。数值比较在 VST 支撑位置上查询距离不超过 0.15 m 的最近原始点 cost，不把墙面预测和缺失支撑单元硬凑成相同输出。

| 相同支撑位置的比较范围 | 原倾斜输入 MAE | 重力对齐 MAE |
|---|---:|---:|
| 场景 XY 范围内，151,219 点 | 0.2814 | 0.2320 |
| 全部观测范围，202,604 点 | 0.3440 | 0.3698 |

这些数值是相对 VST 的一致性；VST 与模型共享观测，不属于独立真值。场景范围内阈值 15/50 的通行判定一致率从 60.67% 升至 73.15%，全体观测范围并非所有指标均改善。

原始数值、节点增量与拼接 PCD 位于 `offline/vst_comparison/`；`comparison.json` 保存全部口径，`comparison_topdown.png` 显示第 0 层相同支撑位置的原输入、重力修正与 VST。RViz 使用水平 `world` 视图：两个面板默认显示重力对齐模型与 VST，相同支撑几何并排呈现；可选图层提供原预测、绝对差和全部原始点预测。

```bash
cd /home/cjh/Pure3D_ws
ROS_DOMAIN_ID=184 ./Entry/launch/mission.sh view --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission --compare --rviz
# 仅看重力修正后的全部原始点 cost：
ROS_DOMAIN_ID=184 ./Entry/launch/mission.sh view --mission /home/cjh/Pure3D_ws/outputs/entry_bag_mission/live_20261114/mission --variant gravity --rviz
```

## 实际短目标与 node EROI 执行（2026-11-14）

用户选定每个 node 保存运动和停留期间的全部点云后，已在隔离域 188 完成 8 段新采集。每段仅在前一目标实测到达并静止 3 秒后选取；单段长度为 `2.141、2.846、1.728、1.902、2.477、2.635、2.084、2.568 m`，总长 `18.380 m`，8 次停留均达到至少 3 秒，bag 检查通过。

域 184 独立回放生成 979 帧、4,469,090 个注册点，按上述规则分为 `n0…n8`。各节点点数依次为 `591093、446317、500729、379661、410808、507377、499532、405059、728514`；没有对节点点云做局部裁剪，物理表面的重复观测保留。重力对齐 cost 计算与拼接完成，全图重算最大差为 0。ROS 可视化验证已确认选取 n3 时，发布的全部 379,661 个点与该 node 保存的 XYZ/点序完全一致，并带独立 RGB 颜色。

产物位于 `outputs/entry_bag_mission/short_nodes_20261114/mission/`，报告为其上级 `result.json`，EROI 检查为 `eroi_validation.json`，分色概览为 `node_eroi_overview.png`；所有点云与 EROI 块集合均可分别读取。
