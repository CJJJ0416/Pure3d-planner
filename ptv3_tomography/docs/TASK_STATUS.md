# PTv3 / Tomography 训练任务确认表

更新日期：2026-11-05。

## 确认规则

- `[x]` 表示任务产物存在，且列出的技术验收已经通过。完成后记录结果、证据和适用范围，再勾选。
- `[ ]` 表示仍有工作或验收未完成；有部分产物时说明进度，不把整个阶段提前勾选。
- 后续变更使原验收失效时，撤回相应确认并记录原因。每次完成任务同步更新本表，回复中说明完成项与下一步。
- 本项目可视化优先使用 Isaac Sim 原生视口。额外网页或重绘场景在用户另行要求时使用。

完整实施说明见 [训练流程](TRAINING_WORKFLOW.md)，网络接口见 [A/B 设计](ARCHITECTURE_AB.md)。

## 已完成并确认

- [x] **S0：组织训练流程与 A/B 接口。**
  - 结果：已记录数据采集、标签掩码、共同感知预训练、冻结、BC/DAgger/PPO/PGCL 和独立评估的顺序。
  - 证据：[实施流程](TRAINING_WORKFLOW.md)、[A/B 设计](ARCHITECTURE_AB.md)。
  - 范围：方案与工作包已组织，不代表网络实现或训练已完成。

- [x] **S1：配置简化 PhysX MID-360。**
  - 结果：Isaac Sim 5.1 原生 PhysX Generic Sensor；默认 1000 射线/帧、10 Hz、0.1–10 m、前下倾 45°；250/500/1000/2000 射线预算通过离线配置检查。
  - 证据：[配置](../../../configs/lidar_mid360_physx_lite.json)、[采集入口](../../../isaac_training/scripts/perception/run_mid360_physx.py)、[传感器说明](../../../isaac_training/docs/sensors/mid360-physx-lite.md)。
  - 范围：单传感器 CPU PhysX、位姿驱动载体；不包含完整四足动力学、真实逐点时序或训练用 GPU 批量后端。

- [x] **S2：完成几何标定和扫描接口验收。**
  - 结果：1000 射线静态标定 12 帧、500 射线六自由度移动标定 24 帧通过；解析命中判断无不一致，覆盖角度正负、外参、位姿更新和缓冲续传。最终显式配置启动另通过 3 帧检查。
  - 证据：[静态标定](../../../outputs/isaac_mid360_lite_calibration_20261104/report.json)、[移动标定](../../../outputs/isaac_mid360_lite_moving500_20261104/report.json)、[最终启动检查](../../../outputs/isaac_mid360_lite_final_smoke_20261104/report.json)。
  - 范围：与解析几何的一致性，不是硬件传感器精度或真实 MID-360 仿真保真度验证。

- [x] **S3：加载现有坡道碰撞网格并导出原始点云序列。**
  - 结果：短片段 24 帧、上坡参考轨迹回放 130 帧，均实际输出 1000 射线/帧；保存命中/缺测状态、方向、传感器系/世界系点、外参、位姿与扫描起止时间。
  - 证据：[24 帧报告](../../../outputs/isaac_mid360_lite_ramp_20261104/report.json)、[130 帧报告](../../../outputs/isaac_mid360_native_viewer_20261104/report.json)。
  - 范围：单场景原始序列与参考轨迹采样；后续标签配对见 D1，不代表机器人自主上坡结果。

- [x] **S4：使用 Isaac Sim 原生窗口显示扫描。**
  - 结果：原生场景、原生传感器点云及位姿回放已显示；`--gui --keep-open` 提供持续查看、原生 Play/Pause 和关闭退出。
  - 证据：[实际原生视口截图](../../../outputs/isaac_mid360_native_viewer_20261104/isaac_viewport.png)、[窗口启动记录](../../../outputs/isaac_mid360_native_viewer_20261104/viewer_state.json)、[使用说明](../../../isaac_training/docs/sensors/mid360-physx-lite.md)。
  - 范围：确认原生可视化功能已运行；窗口启动记录不保证未来任何时刻进程仍存活。

本机 `outputs/` 下的证据不随源码包分发；换机器后需要重新运行相应验收。

- [x] **D1：点云与 Tomography 标签配对，首包连续样本可加载。**
  - 结果：复用 130 帧实际 PhysX 扫描；生成 130 份局部标签、K=8 的 123 个连续窗口，另生成 K=1 的 130 个窗口。局部重力/yaw 系、0.1 m / 64×64 / 6 个实际高度带，保存源层、独立掩码、时间和相对位姿；spec/manifest 锁定源场景、Tomography、扫描与配对代码指纹。
  - 有效监督：K=8 高度 72,119、代价 69,607 单元；代价≥10 为 34,458、代价60为187。未知及同带异高歧义不当安全，支撑全部 `unknown(-1)`、`mask_support=0`。
  - 验收：全部 130 帧与 123 个窗口通过；监督高度对原始三角网格最大误差 2.54 cm；源代价/高度一致、外参/旋转/因果窗口通过。起点 frame0、坡中 frame52、平台 frame129 已用 Isaac Sim 原生视口叠加检查。K=1 另通过数值核验；两包实际 PyTorch DataLoader 均可组成 batch。
  - 证据：[配对说明与重现入口](legacy-v1-data.md)、[K=8 数据契约](../../../outputs/mid360_tomography_pairs_20261104/dataset_spec.json)、[最终验收](../../../outputs/mid360_tomography_pairs_20261104/completion_report.json)、[几何核验](../../../outputs/mid360_tomography_pairs_20261104/validation/alignment_report.json)、[原生叠加](../../../outputs/mid360_tomography_pairs_20261104/validation/native/report.json)。
  - 范围：此项确认的是单场景配对/加载，当时尚未训练PTv3；后续训练见P1-T1。代价局部可观测性目前使用覆盖代理，未证明完整全局 VST 代价可从局部恢复；支撑和机器人可执行性仍未验证。K=1 掩码内没有代价60，先用于工程链路，不能替代完整危险区域训练集。

- [x] **D1-V1：交互展示配对标签与监督掩码。**
  - 结果：直接打开 D1 保存的 Isaac Sim 原生场景；增加帧/高度带选择、点云开关、教师代价/有效监督/代价掩码三种模式。frame52 的 4046 个教师单元与 331 个代价监督单元分开展示；绿0→黄30→红60，支撑仍 unknown。
  - 证据：[入口与坐标说明](legacy-v1-data.md)、[原生截图与回调核验](../../../outputs/mid360_tomography_label_viewer_20261104_v2/report.json)。三种模式截图已查看，面板回调通过，源数据未改动。窗口就绪记录不保证未来进程始终存活。

- [x] **S0-R1：确认单帧 PTv3 + SRU 主架构。**
  - 结果：当前帧雷达点云经预训练、冻结的 PTv3 输出空间特征；融合相对雷达 odom、时间差、目标和状态后，由 SRU 和 Actor 输出速度。SRU 保留时序状态；历史 Attention 为后续可选对照。
  - 证据：[当前主架构与特征接口](ARCHITECTURE_AB.md)。范围：此项为设计确认；后续微型感知训练见P1-T1，新导航网络仍未接入。

- [x] **S0-R2：绘制当前导航主架构。**
  - 结果：用独立框图展示单帧点云、冻结 PTv3、特征适配、相对雷达 odom/时间/目标/状态、SRU 状态接续与 Actor 速度输出。
  - 证据：[PNG 图](figures/ptv3_sru_navigation.png)、[SVG 矢量图](figures/ptv3_sru_navigation.svg)、[可编辑 DOT 源](figures/ptv3_sru_navigation.dot)。图形已渲染并检查；表示设计，不表示模型已实现或训练。

- [x] **S0-R3：按确认的单帧 PTv3 + SRU 主线重新绘制 Plan A/B。**
  - 结果：Plan A 通过预测地图及地图 CNN 接入 SRU；Plan B 通过空间特征适配接入 SRU，cost/高度预测头为预训练与可选诊断支路。两者均使用逐步输入、相对雷达 odom/时间及 SRU 状态。
  - 证据：[Plan A 图](figures/plan_a_single_frame_sru.png)、[Plan B 图](figures/plan_b_single_frame_sru.png)；均提供同名 SVG 和 DOT。两图已渲染并目视检查，属于网络设计图。

- [x] **R1：PTv3代码审查与目录集中。**
  - 结果：官方模型统一使用 `third_party/PointTransformerV3` 原有仓库；项目配对、loader、配置、验收及文档归入 `src/ptv3_tomography`。逐文件校验一致后删除重复官方副本，历史指纹引用的旧文件保持原字节。
  - 修复：相对路径数据迁移、跨脚本几何导入、启动器导入副作用、2步展开预算、极小射线外参保留、分块hash与原子报告。
  - 验收：官方46份文件与9份历史源文件/配置指纹通过；新目录导入、数据包改名后的3帧小检查及有限回归通过，峰值约557MiB，硬限制2GiB。本次没有运行模型或训练，也未启动Isaac。
  - 证据：[代码审查](CODE_REVIEW.md)、验收记录（历史验收入口已归档）、[项目入口](../README.md)。审查时完整训练器/解码/checkpoint/SRU接入仍待实现；后续P1-T1已完成感知部分，SRU仍待接入。

- [x] **P1-T1：真实扫描的单帧PTv3监督训练、保存与恢复。**
  - 结果：使用frame0/52/129实际PhysX扫描，最多256点、6×32×32标签、batch1、workers0。固定官方微型PTv3 + 3近邻空间插值/MLP解码，cost/重力高度分别用独立掩码Huber；5步Adam更新，骨干与解码头均有非零梯度和实际参数变化。
  - 恢复：第3步checkpoint在另一进程加载，重放第4/5步。模型及BN缓冲、Adam、Torch/Python/NumPy随机状态、样本游标、损失/梯度与eval输出完全一致，输出最大差0。5个训练步另重放2步，共执行7次更新。
  - 资源：按用户最新要求，训练和恢复全程6GiB cgroup硬限制、swap0，CPU单线程，峰值约536MiB；未启动Isaac/GPU/SRU。原始数据与官方源码指纹通过。
  - 证据：训练与重现说明（历史验收入口已归档）、验收摘要（历史验收入口已归档）、配置（历史验收入口已归档）。详细结果与checkpoint保存在`outputs/p1_cpu_smoke_6g_20261104/`。
  - 范围：随机初始化微型模型的工程链路确认，不是正式预训练或可冻结导航模型。支撑损失保持关闭，SRU状态接续仍待验收。

- [x] **PA-M：Plan A局部cost/高度图接口。** 当前帧点云经完整PTv3与解码头，输出`[1,2,6,64,64]`地图；另提供只依赖当前输入端点的观测标记。130帧已推理。cost MAE=7.5453/60，高度MAE=0.07807m，均为当前训练场景监督单元。通道/归一化/观测标记/预测3D重建及文件指纹核验通过。见[说明](PLAN_A_LOCAL_MAPS.md)、验收（历史验收入口已归档）。
- [x] **PA-V1：Plan A训练结果静态可视化。** 起点0、坡中52、平台129，每帧6带的标签/预测/绝对误差对比图，共18张，另有3张全部高度带稠密预测概览。图片已生成并查看，未知位置独立标识。
- [x] **PA-V3：同视角RGB与GT对照。** 第52帧使用同一场景网格的轻量EGL/OpenGL离屏渲染，生成RGB、GT cost投影、GT重力高度投影三联图；58个cost和63个高度监督单元均通过视野/遮挡检查。图像已查看。RGB为合成网格渲染，不是真实相机采集，未启动Isaac。产物`outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt.png`及同名JSON。
- [x] **PA-V2：Isaac原生预测可视化。** 用户随后明确授权无硬件配额启动，原生窗口已实际打开。复用原场景的网格/拓扑通过，GT/预测cost、高度及误差面板、帧/高度带/稠密显示回调通过，生成4张原生截图并查看实际窗口。证据`outputs/plan_a_maps_20261104/native/report.json`。当前viewer_state为closed，成功运行记录不代表窗口此刻仍打开。
- [x] **PA-V4：在RGB/GT原图中补充PTv3预测。** 保留第52帧RGB和原相机，在第二排加入真实保存的预测cost和高度。沿用相同GT参照点位及色标，58个cost单元、63个高度单元，MAE分别7.9042/60和0.06149m。新图已查看，原图未修改，产物`outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt_prediction.png`及同名JSON。

## 接口保留验收项：D1-R0 雷达参考系接口与标签统一

- [ ] **D1-R0：将逐帧雷达 odom、点云、时间与当前雷达参考的输出契约统一。**
  - 已明确：在线每步输入当前帧点云、雷达 odom、时间与有效点；雷达 odom 描述同一参考系下的雷达自身位姿。由相邻帧计算相对位姿与时间差；训练时按顺序展开多步，历史由 SRU 状态携带。
  - 已完成数值接口：新增独立 v2 包，130帧当前雷达点云、雷达自身 odom、相对位姿/时间与当前帧监督；输出采用雷达XY及重力高度带，保留完整雷达XYZ表面坐标。完整数值检查最大高度误差2.36cm，有序loader为17个不重叠块；旧v1保留。
  - [x] 小内存检查：按用户要求改为3个独立帧、每帧最多256点、32×32标签、batch=1、workers=0、CPU逐帧；减点/裁图后重新计算掩码。在2GiB系统硬上限下通过，实际进程峰值约557MiB。
  - [ ] 原生资源验收：6张截图已生成，但报告未完整写入，用户报告内存耗尽；本轮不标为完整通过。已关闭默认原生启动，后续仅允许显式启用且受4GiB硬限制的单帧320×240检查。本轮不自动重试Isaac。
  - 证据：[v2接口与受限检查说明](lidar-reference-v2.md)、[数值验收](../../../outputs/mid360_lidar_reference_v2_20261104/validation/alignment_report.json)、[2GiB硬限制实测](../../../outputs/mid360_lidar_reference_v2_20261104/validation/small_memory_guarded_check.json)、[资源中断记录](../../../outputs/mid360_lidar_reference_v2_20261104/validation/native/resource_interruption.json)。整体D1-R0继续保留未确认；用户随后授权先完成数值接口上的P1-T1训练，已通过，未重试原生渲染。
  - 依据：用户明确学生使用雷达坐标参考、逐帧 odom 基于雷达自身。原生资源验收独立保留，已核对的数值标签用于P1-T1；支撑仍unknown。

## 当前主线：单帧 PTv3 + 地图解码头预训练

用户2026-11-04明确：当前仅推进单帧点云的cost/地面高度感知训练，SRU和导航后端暂不推进。PTv3与地图解码头共同训练，感知训练完成后再讨论冻结。

- [x] **P1-T2：持续感知训练入口。** 支持完整捕获包、多包数据登记、epoch遍历、shuffle、单帧microbatch与梯度累积、独立cost/高度掩码、AdamW、原子checkpoint及中途恢复。CPU已遍历130帧一轮，33次更新；恢复第32次更新后的两个剩余样本，结果与连续训练完全一致。
- [x] **P2-C：完整PTv3配置。** 官方5级编码/4级解码骨干46,154,272参数，64通道F，分块解码6×64×64地图。提供A100持续训练与4060单轮配置。
- [x] **P2-G：本机RTX4060完整模型短epoch。** 完整PTv3骨干46,154,272参数，合计46,163,618参数，130帧完整遍历1轮，33次AdamW更新，末次2帧。训练阶段31.79秒。主机RSS峰值2151.74MiB，PyTorch显存分配峰值835.85MiB、保留峰值936MiB。未设置硬件配额，CPU使用默认24线程。
  - cost监督13,992单元，高度监督16,513单元，原始数据及46份官方源码指纹通过。骨干与解码头均有非零AdamW矩。
  - 同3帧训练诊断Loss由0.442921降到0.090921，cost MAE由21.84降到6.50，高度MAE由0.18074m降到0.05547m，属于训练集诊断。
  - 第32步在独立GPU进程恢复，再更新最后两个样本，模型、AdamW、CPU/CUDA/Python/NumPy随机状态、样本游标及整轮指标完全一致。
  - 已严格重载checkpoint，在第52帧导出6×64×64的cost/高度和184×64的空间F。证据：GPU验收结果（历史验收入口已归档）、`outputs/rtx4060_full_epoch_20261104/last.pt`。
- [ ] **P2-D：数据覆盖与独立验证。** 训练仍只有单建筑/单路线。另已完成一个新生成楼梯场景的40帧原权重诊断，见PA-X1。多路线与建筑谱系划分、危险区域覆盖，以及系统的独立dev指标仍待补充。
- [ ] **P2-S：H100 服务器正式训练。** 当前目标为 H100，采用独立 PTv3 环境和新 batch 配方。用户当前要求只完成本地，目标机吞吐尚未记录。

硬件限制更新：先前2/6GiB检查记录保留其历史含义。用户随后明确“不做任何硬件限制”，当前持续训练和推理不再强制这些配额。支撑loss保持关闭。

## 后续待办

- [x] **P1：当前单帧感知工程链路。** PTv3/解码、掩码损失、反向、保存恢复与完整轮次遍历均已确认。SRU接入转入未来导航阶段，不是当前感知训练的前置任务。
- [ ] **D2：支撑标签专项核验。** 平台边缘、孔洞、桥上/桥下及未知区域，补充可靠标注与独立诊断；通过后启用支撑头。
- [ ] **D3：多路线、多建筑数据与分组。** 增加反向、转身、偏离中心线等观测；建筑谱系隔离 train/dev/test。
- [ ] **P2：感知预训练。** 在未见建筑上验证代价和高度，锁定可复现 PTv3 checkpoint；支撑在 D2 核验后另行启用。
- [ ] **N0：在线传感器与冻结前端适配。** 对齐离线/在线传感器契约，实际学生位姿产生观测，核对完整吞吐。
- [ ] **N1：单个导航分支。** 保留既有任务与奖励契约，依次运行 BC、DAgger、PPO/PGCL 和仅终点独立执行。
- [ ] **N2：A/B 对照和未见建筑评估。** 使用共同感知 checkpoint，按相同条件比较地图接口与直接特征接口。

P1 是工程链路检查，D2/D3 是可靠性和覆盖扩展；它们的完成记录分别维护，不用一项成功替代另一项。

## 目录归属

当前维护入口为本项目 `scripts/`、`configs/` 与 `src/ptv3_tomography/`。旧 `isaac_training` 下已被数据指纹引用的文件保留为历史快照；新工作不在旧副本继续扩展。审查结论见 [CODE_REVIEW.md](CODE_REVIEW.md)。

## 新场景迁移检查与完整VST地图

- [x] **PA-X1：Terrain Generator新场景原权重推理。** ETH楼梯模块加本地平台/障碍物拼装，40帧实际PhysX扫描，独立标签与几何核验通过。原checkpoint未改，cost MAE=11.5192/60，高度MAE=27.957cm。RGB/GT/预测图及Isaac原生六模式截图已生成并查看。属于一个新场景的诊断，P2-D仍待多场景系统评估。
- [x] **PA-X2：整张地图VST原始costmap可视化。** 两个104×104层，0.1m分辨率，21632格全部显示。未应用局部裁图或监督掩码，原始cost=60保持红色，无地面观测加斜纹。另在Isaac原生场景叠加全部有地面高度的cost格，源地图和场景指纹未变。详见[新场景诊断](TERRAIN_GENERATOR_TRANSFER.md)与`outputs/terrain_generator_transfer_20261104/native_full_vst/report.json`。

- [x] **PA-X3：独立逐点恢复分支与完整点云可视化。** 新楼梯40帧11,234个有效点全部恢复，frame16为166×64→307×64。原地图预测差为0，原权重和官方源码未变。Isaac四模式与全景/近景共8张截图及界面回调通过，最后窗口状态为closed。PCA颜色表示特征，cost颜色来自既有地图采样。详见[逐点恢复](POINTWISE_RESTORATION.md)。
- [x] **PA-V5：按逐点恢复配置重新输出第52帧RGB/GT/预测。** 原坡道130帧37,164个有效点全部恢复。frame52为184×64→271×64，271点均保留在原相机中，267点着色、4点范围外显示灰色。GT仍为原58个cost格、63个高度格，误差保持7.9042/60和0.06149m。新图`outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt_prediction_point_restored.png`已查看，原图未覆盖。
- [x] **PA-V6：GT也映射到完整LiDAR点云。** 新图`outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt_prediction_all_points.png`中，全部271个点从原生VST世界XY及匹配高度带取到无歧义GT。预测267点有值、4点范围外灰色，双方共享原始XYZ/相机/色标。逐点VST索引和值核验通过，267个共同点的查表诊断cost MAE=10.1714/60、高度MAE=8.216cm。训练掩码和权重未改，旧图与旧指标保留其原统计范围。
- [x] **PA-V7：最后一帧平台边界GT/预测。** frame129的237个有效点全部恢复，GT均可取值。已生成平台近景与自动拟合完整237点的全景图，分别为`plots/frame_00129_rgb_gt_prediction_all_points.png`和`plots/frame_00129_rgb_gt_prediction_all_points_full_view.png`。全景预测202点着色、35点范围外灰色。共同202点cost MAE=9.5231/60、高度MAE=2.3968cm，平台高代价边缘明显低估。相机投影计数、GT索引/数值和两个视图均已核验，训练数据及权重未变。

## 服务器源码整理与提交准备

- [x] **R2：迁移前代码审查与测试清理。** 活动源码清理61个测试/旧入口/结果文件，历史证据归档保留。加入仅权重初始化、checkpoint保留、环境变量与配置校验修复，正式训练/配对/推理/可视化代码保留。
- [x] **R3：本地独立迁移包与真实恢复验收。** 本地忽略路径`outputs/server_transfer_20261104/ptv3_a100/`包含130帧train、40帧dev、源码及原权重。完整模型5次GPU更新和另进程恢复1次更新完全一致。迁移包、数据、权重和验证输出不进入Git。见[本轮审查](SERVER_CODE_REVIEW.md)和[A100入口](SERVER_TRAINING.md)。
- [ ] **R4：H100 服务器训练。** 保留独立 torch2.10 与 H100 batch 配置。用户当前要求只完成本地训练与峰值测量，服务器训练尚未执行。旧 preflight 不再作为启动流程。

## 2026-11-05：真实 batch 训练与 GPU 峰值

- [x] **P2-B：真实 batch 与异步数据加载。** 一次性 mmap 缓存、官方点云 collate、并行地图头、按帧 masked loss、BF16/Triton、AdamW/OneCycle 已实际训练。
- [x] **P2-P：RTX 4060 本地峰值与吞吐。** 相同 40 个实际扫描帧，batch=4/16/32 稳定吞吐分别为 62.5/115.6/118.4 帧次每秒。CUDA 分配峰值为 0.82/1.45/2.30 GiB，GPU 采样峰值为 92/94/100%。
- [x] **P2-R：分段接续。** batch=32 从 epoch10 / step20 恢复模型、优化器、OneCycle 与 RNG，实际完成 epoch30 / step60，共 1200 帧次。
- [x] **R5：精简正式流程。** 删除环境预检、RayCaster 自检、资源守护入口。新版训练热路径不做重复全量哈希、逐帧断言测试或 fsync，不施加硬件配额。

证据与复现命令见 [LOCAL_BATCH_TRAINING.md](LOCAL_BATCH_TRAINING.md)。这些记录来自当前两条路线的实际训练，未连接服务器。

- [x] **R6：官方训练代码比对与中文注释。** 官方固定版本 46 份上游文件逐文件相同。按官方 README 引用的 Pointcept 训练代码比对 DataLoader、前反向、AMP、优化器、OneCycle 和 checkpoint，并标注地图回归任务适配。仅注释和文档，可执行逻辑保持不变，见 [对照记录](OFFICIAL_TRAINING_COMPARISON.md)。

- [x] **R7：训练工程与上游核心分离。** 任务工程移至 `src/ptv3_tomography`，上游核心移至 `third_party/PointTransformerV3`，46 份固定上游文件保持原字节。安装、导入、脚本、迁移包和文档引用统一为新布局，见[目录说明](PROJECT_LAYOUT.md)。原本地产物在整理执行时已不存在，本次未重新生成。


## 2026-11-05：整理后全流程短训练与耗时清理

- [x] **P2-F：重新采集并贯通完整单帧流程。** 九区域场景两条路线共 40 帧、1000 射线/帧，实际完成原生 RayCaster 采集、标签、portable、mmap、真实 batch=16 的 3 epoch / 9 步，新进程恢复至第 4 epoch / 第 12 步，再导出三帧 cost/height 地图。
- [x] **R8：删除生产链路中的重复防御操作。** 清理重复全帧/序列回读、重复全量哈希、额外射线方向回传、CPU 回传后的 Warp 同步和逐文件 fsync。checkpoint 阶段/best 改为同文件系统原子硬链接。没有新增防御性测试、预检、守护或硬件配额。
- 结果：标签内部计时 3.131→2.337 s，恢复段 checkpoint 2.143→0.426 s。短窗口训练吞吐 120.49 帧次/秒，CUDA 分配峰值 1.45 GiB。监督格数量及三帧导出误差指标与首轮一致。证据、计时边界和本地产物见[短训练记录](SHORT_PIPELINE_PROFILE.md)。
- 范围：训练数据上的流程确认。独立随机 seed 评估和服务器正式训练仍待执行。


## 2026-11-05：服务器旧版单帧 BatchNorm 故障入口修复

- [x] **R9：调度器接入真实 batch v2。** 原默认 A100 v1 模板改为 H100 batch32/workers8/drop_last。正式协议为 10 段×10 epoch，OneCycle 固定 100 epoch，段间完整接续。
- [x] **R10：已有数据与旧权重复用。** `--reuse-campaign` 保留原 registry、DR 和评估 seed，跳过训练数据采集配对。首段支持 `--init-checkpoint`，新 v2 段间/同段重试使用 `--resume`。本地已有 40 帧数据实际完成 2×1 epoch、6 次更新，没有重采数据。
- [x] **R11：失败帧可定位。** v2 错误报告记录已有 batch 来源元数据，保留异常抛出。没有新增预检、深层点数测试、GPU 同步或官方 BatchNorm 修改。
- 证据：[切换说明](SERVER_TRAINING.md#81-旧版-batchnorm-单点故障与切换方法)、本地 `outputs/ptv3_tomography/campaign_v2_dispatch_20261105/report.json`。默认 H100 协议已实际生成，未在 H100 启动训练，也未读取服务器 v1 权重文件。
