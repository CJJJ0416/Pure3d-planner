# NEUPAN 学习规划方法

三维目标和当前支撑节点进入邻居评分器。节点输入为相对 XYZ 与 `trav_cost`，共享 MLP 和两层邻域聚合产生 64 维特征。每次解码从最多 8 个有效邻居及 `STOP` 中选一个动作。输出路径的 Z 来自各支撑节点。默认局部上限为水平路程 3 m、64 步。

原生 VST A* 生成整图专家节点链。局部监督用低代价支撑截断专家链，并允许在同一支撑图上绕行后接回专家路线。楼梯样本保留原观测、遮挡和恢复三个版本。GT 切片使用 VST 原始 `cost <= 15`，特征除以 50。在线预测点云使用原始 `0 <= trav_cost < 0.8`。网络将两类代价按各自的通行阈值归一化。局部图复用 VST 高度兼容选层，要求实际相邻支撑和对角两侧支撑，单边最大高差 0.30 m，禁用跨空洞边。

## 提前停止与路径可达性

继续行走和 `STOP` 使用独立的二分类，移动方向单独排序。解码启用局部可达性检查。当目标在本轮路程及步数预算内有一条已观测低代价路径时，只在能保持目标可达的邻居中用网络打分。目标分支与当前节点断开时，网络只考虑缩短高度误差或沿当前支撑层向目标靠近的边。推进条件用尽时输出最后一段已验证前缀。用 `evaluate --unguarded` 单测纯网络的解码。可用 `calibrate` 在验证集上为二分类选择分数阈值，校准排除到达判定和无可用邻居的确定性动作。

## 生成标签和训练

在仓库根目录运行。训练使用 PyTorch、NumPy 和 SciPy，原生标签生成需要 g++、Eigen 及 VST 仓库内的 pybind11。ROS Humble 仅在预览发布时使用。已有 PTv3 或 cost head 通过保存的 XYZ 和 cost 提供输入。

```bash
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
DATA=outputs/planner/learned_path_gt
RUN=outputs/planner/learned_path_run

"$PY" -m planner.light_planner.learning_planner generate --output "$DATA" --count 10000
"$PY" -m planner.light_planner.learning_planner train --train "$DATA" --validation "$DATA" --output "$RUN"
"$PY" -m planner.light_planner.learning_planner evaluate --checkpoint "$RUN/best.pt" --dataset "$DATA" \
  --split test --output "$RUN/test.json"
"$PY" -m unittest planner.learning_planner.neupan.verification.component_tests
```

`generate` 默认查找 `datasets/wfc_capability_dr_v2/scene_20261109/vst_nominal.pickle`。它保存 `routes.npz`、`manifest.json` 和原生搜索日志，包括各类路径数量、失败原因和输入校验值。默认将 12 m XY 区块作为划分单元，每个区块保留 3.05 m 缓冲。全尺寸训练建议给出包含完整楼梯的显式区域。教师在各区内独立搜索。可用 `--regions` 提供 `{"regions":[{"id":"stairs_a","split":"train","lower":[x0,y0],"upper":[x1,y1]}]}`，为完整楼梯或独立设施指定边界。小样本回归可用两份不同地图并各指定 `--scene-split`。生成量不足时退出码为 2，同时保留已有数据。

## 预测代价配对与 RViz

为训练、验证和测试集分别准备配对清单。每项记录同一地图坐标系下的真实局部点云、对应专家路线 ID 及从零开始的节点偏移。NPY 接收 `N × 4` 的 XYZ/cost。NPZ 接收 `xyzc`，或 `xyz` 加 `trav_cost`、`prediction` 或 `cost`。相对路径基于清单位置。

```json
{"frames":[{"route":8,"offset":14,"frame":"world","cloud":"snapshot.npz"}]}
```

对已对齐的大地形 cost 回放，用 `python -m planner.light_planner.learning_planner.neupan.data.replay_pairs --help` 从保存的传感器位姿和原生 VST 重建监督。训练、验证的 6 m 观测区域之间保留空间间隔。若近身点云没有训练起点，可用 `--train-query-radius` 在训练快照中选取已有低代价点与 GT 均能匹配的位置。验证使用原始传感器参考点换算的支撑位置。两段来自同一次回放时，按空间留出结果解读。

微调通过 `train --resume <best.pt> --train-pairs <train.json> --validation-pairs <validation.json>` 依次传入 checkpoint 和对应数据。仅训练规划器。投影先在每个 XY 柱中按实测高度分层，并把实测点的 cost 写入对应支撑层。完整 GT 只提供目标和标签。

```bash
"$PY" -m planner.light_planner.learning_planner view --dataset "$DATA" --checkpoint "$RUN/best.pt" \
  --split validation --index 0 --domain 186 --rviz
```

可通过 `--pairs <frames.json>` 切换到预测快照。预览在本机隔离域发布分层 cost、金色 VST 专家前缀、蓝色学习前缀和 JSON 状态。状态带有 cost 版本、路径长度和结束原因。单点 `STOP` 的预览有起点记录，Path 显示需要两个点才能绘线。只有显式传入 `--publish-pure-path` 才在同一隔离域更新 SCAN 路径。空路径代表当前没有可执行前缀。启动另一个路径发布器前先关闭预览。

## 在同一 cost 快照对比候选路径

`view --beam` 在一次图编码后用 64 条束保留不同空间前缀，优先返回到达目标的路径。每对候选在物理 XYZ 上分别要有一段至少 0.30 m 的连续弧长，且该段距另一条整条路径至少 0.25 m。最多显示 3 条，窄通道允许只给一条。候选沿用贪心解码的邻接、可达性、长度与步数约束。

```bash
BASE=outputs/planner/support_policy_pilot_20261120
CKPT=outputs/planner/path_generator_full_20261120/train/best.pt

"$PY" -m planner.light_planner.learning_planner view --dataset "$BASE/revision_1994" --split train \
  --pairs "$BASE/recorded_cost_pairs.json" --index 26 --checkpoint "$CKPT" \
  --beam --domain 189 --rviz
```

`/entry/learning/candidates` 是候选的 MarkerArray。RViz 中的 `dZ` 为终点减起点高度，`span` 为整条路径的最高点减最低点。可打开已有的单路径显示进行对照。`--report result.json` 保存路径和成对分离距离，搭配 `--plan-only` 可离线生成报告。束搜索预览独立于 SCAN 的路径命令。

## 输出短程落脚目标

`--local-goals` 对已观测的支撑图做默认 1.2 m 的短程搜索，只输出最多三个相隔 0.30 m 以上的候选落脚点和结束原因。候选支持点发布到 `/entry/learning/support_goals`。单窗口首选点以机身参考高度发布到预览话题 `/entry/learning/local_goal`。SCAN 模式 1 的目标话题是 `/scan/goal`，楼梯导航使用 `interface.manual_goal_use_z: true` 和一致的坐标系。当前 SRU A1 导航桥将目标 Z 设为当前底座高度，同 XY 跨层目标需要保留高度信息的接入方式。

```bash
RUN=outputs/planner/path_generator_full_20261120
"$PY" -m planner.light_planner.learning_planner view --dataset "$RUN/gt" --split test --route-id 4808 \
  --local-goals --checkpoint "$RUN/train/best.pt" --domain 189 --rviz
```

`--route-id` 从 VST 留出测试路线的起点开始，随后以网络上一次选出的支撑点为中心裁剪局部代价图。RViz 用黄色线显示 VST 专家路线，用带编号的青色点显示网络选择的落脚点。该入口记录离线落脚点序列供对照，预览不会送入控制话题。

对实测代价快照，在带 `--pairs` 的 `--local-goals` 预览命令上加 `--rollout`。每轮以新选出的支撑点为中心从同一份预测 XYZ 和 cost 重建局部图。RViz 显示配对的原生 VST 参考，VST 只提供这次查询的参考起终点和对照线。

## 按支撑面积与可达性训练局部目标评分器

这条离线流程用 VST 的 GPMP 曲线选取平地、绕障、上楼和下楼的查询位置。曲线通过密采样的支撑与净空检查后，沿弧长按约 0.4 m 取检查点，并保留转弯及踏面变化处。检查点需在半径 0.15 m 的圆盘内获得至少 0.04 m² 的连续几何支撑。面积按真实高度匹配及四邻接计算，同一物理支撑面积只计一次。候选点的代价单独检查，GT 用未膨胀 VST cost ≤ 15，预测用 `trav_cost < 0.8`。GPMP 在参考点附近 0.20 m 的同一支撑面优先选择面积最大的合格点，面积相同时取贴近参考的位置。

对每个查询，训练样本包含当前已观测的候选点。面积头估计 GT 支撑面积。目标效用头学习哪些候选同时具备足量支撑、1.2 m 内的 GT 短程接近和通往最终 goal 的连续支撑路径，并且能缩短至少 0.08 m 的 GT 剩余距离。符合条件的候选均可获正标签。配对遮挡保留相同的 GT 监督，用于检查缺测前后的选点。离线图可达性是训练与评估的依据，实际落脚及控制效果需在仿真或实机中评估。

接近覆盖监督使用当前局部 Tomogram。从机器人所在的支撑格出发，在 `trav_cost < 0.8` 的分层支撑图上求 1.2 m 内的连续可达范围。缺测格和高代价格切断连接，有效的低代价绕行仍可获得正标签。覆盖头学习当前观测能否支持接近，目标效用正标签同时要求 GT 目标进展和当前接近覆盖。推理时，支持面积、当前覆盖和效用分数都通过才输出候选。没有符合条件的候选则返回空列表和停止原因。

```bash
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
NATIVE_PY=/home/cjh/glocal2-nav/.venv/bin/python
ROUTES=outputs/planner/path_generator_full_20261120
LAND=outputs/planner/landing_rebuild
RUN="$LAND/student"
GRID=datasets/wfc_capability_dr_v2/scene_20261109/vst_nominal.pickle

# CLEAN_PCD 指向生成 GRID 时使用的同一份地形点云
"$PY" -m planner.light_planner.learning_planner.neupan.data.landing_raw --pcd "$CLEAN_PCD" --grid "$GRID" --output "$LAND/raw_gt.npz"
"$NATIVE_PY" -m planner.light_planner.learning_planner.neupan.data.landing_gpmp --dataset "$ROUTES/gt" \
  --raw-map "$LAND/raw_gt.npz" --output "$LAND/gpmp_expanded" --count 600
"$PY" -m planner.light_planner.learning_planner.neupan.data.landing_dataset --dataset "$ROUTES/gt" --experts "$LAND/gpmp_expanded" \
  --raw-map "$LAND/raw_gt.npz" --regions "$ROUTES/regions.json" --output "$LAND/dataset_expanded"
"$PY" -m planner.light_planner.learning_planner.neupan.data.landing_approach --dataset "$LAND/dataset_expanded" --nominal "$GRID" \
  --output "$LAND/dataset_approach"
"$PY" -m planner.light_planner.learning_planner.neupan.training.landing_train --dataset "$LAND/dataset_approach" --output "$RUN" \
  --epochs 12 --preview-every 3
```

训练在首轮、每隔指定轮数和最后一轮保存图和 JSON。默认监视同一组留出验证窗口，包括同一下楼位置的完整、接近区缺测、落脚区缺测三个输入，以及上楼、绕障和平地。图中圆圈对应实际面积查询半径，标注 GT 面积、高度变化和可达性判断。`preview/latest.png` 始终指向最近一轮的图。进度图固定采用 `--preview-threshold`，最终 checkpoint 的阈值另在全量验证集上校准，并保存独立的最佳轮次预览。绘图需要 Matplotlib。最终再用测试集评估。

接口 `LandingScorer(checkpoint).candidates(tomogram, start_support_xyz, goal_support_xyz)` 接收重力对齐的局部 Tomogram，返回最多三个支撑 XYZ、观测面积、候选点 `trav_cost`、估计面积和目标效用分数，以及为空或有候选的原因。带覆盖头的评分器先按分层路线筛选，再在同一连通踏面、距离候选 0.20 m 内优先选面积更大的点。替代点也须通过接近和目标进展检查。新数据及权重记录面积定义，旧权重仍读取训练时使用的低代价面积特征。

带覆盖头的评分器以分层支撑图计算点间代价。每条边的权重为水平边长乘 `1 + 4 × 两端平均 trav_cost`。目标有连通观测时，从目标反向计算剩余代价。目标超出当前可达观测时，用带三维目标偏差的估计值给可观察的前缀排序。候选必须使剩余代价下降至少 0.08，并通过原有接近距离和落脚面积检查。排序奖励剩余代价的下降，扣除相对最优路线的绕行代价。返回结果含实际计价的支撑节点、接近代价、剩余代价和是否连通到观测目标。调用方可通过 `visited=` 传入已抵达的支撑点，在滚动快照之间抑制回访。

覆盖训练的进度图用红叉标出有落脚面积、但当前观测未证实能接近的点。JSON 记录每个候选的观测接近距离，停止窗口显示 `STOP`。用 `--baseline <旧评分器 checkpoint>` 可在同一测试集上对照覆盖通过数和停止情况。加载旧 checkpoint 可复现原来的选点接口。

查看 `revision_1994` 实测快照的独立诊断。青色 N 点是网络候选，G 点显示观测支撑面积，带问号的 P 点来自未通过检查的 GPMP 曲线。该预览只发布 `/entry/landing_preview/*` 和其静态 TF。

```bash
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=192 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 -m planner.light_planner.learning_planner.neupan.visualization.landing_preview \
  outputs/planner/landing_gpmp_20261120/revision/revision_preview.npz --rviz
```

用 `landing_revision audit --baseline <旧 checkpoint> --checkpoint <新 checkpoint> --reference <原审计目录> --output <对照目录>` 输出前后选点。对应的 RViz 预览用橙色显示旧候选，青色显示新候选，标注从同一起点测得的低代价支撑接近距离。

加上 `--rollout-steps 12` 可在同一快照中反复用首选点推进起点，对照原评分与点间代价评分。省略 `--baseline` 时两组共用同一 checkpoint。报告记录回访、到达和剩余代价，RViz 用橙色连接旧选点，用青色绘制新选点所依据的实测支撑路线。

## 组件调用

### 分层支撑点序列

`SequencePlanner(checkpoint).plan(tomogram, start_support_xyz, goal_support_xyz, visited=())`
返回最多三个有序支撑点、每段观测支撑链与结束原因。单段水平路径最长 1.2 m，总长最多 3 m。输入和输出使用同一个重力对齐坐标系。部署入口继续由调用方决定何时将支撑目标交给跟踪器。

面积辅助头回归半径 0.15 m 内的 GT 连通踏面覆盖。面积排序损失比较同踏面、代价和进展相近的合法点。推理候选来自当前观测图，支撑面积仅在选点后测量并写入诊断。裸几何占用和相对高度送入几何分支，点代价和分层邻接送入三层消息传递分支。

以下命令使用已对齐的大地形 GT raw map。路线生成以约 70/15/15 划分独立实体楼梯和支撑栅格。`sequence_bank --count` 指的是 A* 路线候选数，GPMP 清退数量和各折有效路线以专家清单为准。训练与无面积损失对照共用同一专家清单与种子。

```bash
CPU=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
NATIVE=/home/cjh/glocal2-nav/.venv/bin/python
GRID=datasets/wfc_capability_dr_v2/scene_20261109
RUN=outputs/planner/sequence_experiment
RAW=outputs/planner/landing_gpmp_20261120/raw_gt.npz

"$NATIVE" -m planner.light_planner.learning_planner.neupan.data.sequence_bank --nominal "$GRID/vst_nominal.pickle" \
  --scene "$GRID/scene.json" --output "$RUN/bank" --count 7800
"$NATIVE" -m planner.light_planner.learning_planner.neupan.data.landing_gpmp --dataset "$RUN/bank" --raw-map "$RAW" \
  --output "$RUN/expert" --count 0 --sequence
"$CPU" -m planner.light_planner.learning_planner.neupan.training.sequence_train --expert "$RUN/expert" --raw-map "$RAW" \
  --output "$RUN/student"
"$CPU" -m planner.light_planner.learning_planner.neupan.training.sequence_train --expert "$RUN/expert" --raw-map "$RAW" \
  --output "$RUN/no_area" --no-area-loss
"$CPU" -m planner.light_planner.learning_planner.neupan.evaluation.sequence_benchmark --expert "$RUN/expert" --raw-map "$RAW" \
  --checkpoint "$RUN/student/best.pt" --ablation "$RUN/no_area/best.pt" \
  --baseline outputs/planner/landing_gpmp_20261120/student_approach/best.pt \
  --replay outputs/planner/path_generator_full_20261120/small_pairs_supported \
  --output "$RUN/evaluation"
```

训练默认 12 轮。后半程会重标学生选到的状态，每 3 轮保存同一组验证图。小规模调试可指定 `--train-limit`、`--val-limit` 和 `--test-limit`，单位为成对遮挡之前的路线数。测试产物 `test_*.npz` 可用 `landing_preview <snapshot> --rviz` 在独立 ROS 域复现。RViz 的圆盘、面积与点 cost 分别标注。

### 原单步组件

```python
from planner.light_planner.learning_planner.neupan.inference.path_generator import PathGenerator

generator = PathGenerator("best.pt")
result = generator.plan(local_xyzc, start_support_xyz, goal_support_xyz, cost_version="1234")
body_path = result.body_path(height=0.4)
```

输入 XYZ 必须在同一重力对齐坐标系。调用方负责 TF 和从机身参考点换算到支撑高度。也可传入 `Tomogram`，或者以 `SupportGraph(tomogram)` 构图后调用 `plan_graph()`，同一 cost 版本可复用图。每次调用返回支撑层索引、支撑 XYZ、结束原因、几何版本及分阶段耗时。集成时按新的 cost 版本或 goal 重算，使用调用方的目标撤销、路径跟踪及速度接口。
