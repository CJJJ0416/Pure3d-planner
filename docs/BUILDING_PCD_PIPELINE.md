# 建筑 PCD 清理、结构修复与机器人条件化代价地图

更新日期：2026-11-09。工作目录：`/home/cjh/Pure3D_ws`。

当前执行优先级：已有机器人条件化 cost 已完成训练与新 seed 测试，三维点云 A* 和 GPMP 已接入。现在准备局部部件的结构保留评分及过滤后修复，RECON 采用完整 Mesh 监督的几何自编码与条件 Flow Matching，见 [RECON 训练方案](../Reconstruction/docs/TRAINING_PLAN.md)。目录以 [工作空间文件架构](WORKSPACE_LAYOUT.md) 为准。

网络与规划结构以 [完整网络结构 v3](NETWORK_ARCHITECTURE.md) 为准：rough PCD → filter → recon → calc trav_cost → 归一化代价点云 → A* → GPMP。

## 目标与处理顺序

```text
大型建筑原始 PCD
    -> 带重叠上下文的空间分块
    -> 固定 E_shared：一次编码得到坐标关联特征 F
    -> FilterHead：保留评分，取得 XYZ_keep + F_keep
    -> ReconDecoder / GeometryHead：生成局部缺失表面及特征场 H
    -> 保留规划相关细节的简化静态建筑 PCD
    -> 查询 H + 适配器 / FiLM / CostHead
    -> 带 trav_cost 的三维地图
```

2026-11-12 更新：RECON 当前采用连续大地形的整图 Subtraction，再提取 4×4×3 m 核心和 1 m 上下文区块，加入缺损课程与终点几何 Loss。只删点，不扰动坐标，不加语义条件。完整 Mesh 提供监督，按完整 terrain seed 划分训练、验证和测试。见 [当前 RECON](../Reconstruction/README.md)。

当前采用共享编码器、分阶段训练并冻结。公共 E_shared 固定，先训练过滤头，再训练读取实际保留特征的重建器，最后训练读取重建特征 H 的条件化代价头。完整推理每窗口只编码一次。已有独立 CostModel 及其首轮权重保留为对照，新代价头需要适配 H。

## 当前可复用内容

| 内容 | 实际位置与状态 |
|---|---|
| 真实建筑样本 | `datasets/rot_stairs/raw.pcd`，头部记录 5,886,268 点、binary PCD，字段 XYZ 与法向 |
| 编码器源码 | `utils/PointTransformerV3`，已有 Utonia、Sonata 等预设 |
| 代价模型 | `Traversability/models/model.py`，编码器、查询聚合、FiLM 与 CostHead |
| 本地环境 | `/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python`，沿用已有环境 |
| VST 教师代码 | `common/vst_global_planner`，集中提供标签内核与原生规划器 |

条件化 cost 的训练、推理与三维点云规划已接入正式入口。行人清理和结构补全保留为后续 Reconstruction 阶段。

## 第一阶段：清理与结构修复

输入为带缺损的整张场景 PCD。先输出 filtered.pcd，再输出由原始保留点及补全新点组成的 repaired.pcd。

| 模块 | 监督目标 | 输出 |
|---|---|---|
| 固定 E_shared + FilterHead | 动态来源为0，其余按静态表面误差生成连续系数 | 每个原始点的 keep_score |
| 保留特征 + 重建解码器 | 完整 Mesh 的单元相交标签及真实表面代表点 | 表面存在概率与新增 XYZ |

FilterHead 学习正向保留结构，不要求输出人物语义。当前局部训练保留地面、墙、顶面、楼梯、斜坡、栏杆、柱梁、静止人物及家具。运动来源全部低分，配准副本按最终静态表面距离连续评分。准确断连结构保持高分。宽大规则障碍也进入干净 GT。keep_score 不是通行分数，墙和不可通行障碍同样应该保留。

过滤头训练后冻结，重建器使用其实际预测保留坐标和共享特征训练，不拥有第二个点云编码器。硬筛点发生在重建之前，完整 GT 不因误删缩减。所有部署输入均为离线 PCD，时间戳、位姿和运动判断不进入模型接口。

RECON 保留已有点坐标，以 24 → 12 → 6 → 3 → 1.5 cm 层级查询生成新点。完整 PCD 经 PTv3 和空间编码器形成几何 latent，条件 Flow Matching 从缺损点云特征生成该 latent，完整 Mesh 提供唯一 GT。当前先导出补全 PCD，下游特征传递延后。设计详见 [RECON 架构](../Reconstruction/docs/ARCHITECTURE.md) 与 [解码器候选](../Reconstruction/docs/RECON_HEAD.md)。

预训练编码器只提供特征。`features[inverse]` 只把特征分配回已有点，不会生成缺失的踏面或栏杆。

优先从干净建筑几何构造配对数据：保留干净目标，向输入加入行人、杂波和测量噪声，并模拟局部缺失、遮挡及不同密度。目标保留真实的楼梯井、平台边缘和结构开口，人工制造的缺损才具有对应的补全表面。真实扫描通过人工标注、重复扫描或高质量参考模型提供监督与验证。

栏杆、薄踏步和边缘属于需要保留的建筑结构。点云简化放在结构识别和修复之后，按结构尺度控制采样，分辨率由最细的目标结构决定。

RECON 按完整 terrain seed 划分，同一大地形的全部区块与删点变体留在同一组。CostModel 按建筑划分，同一建筑的相邻块保持在同一划分中。

## 第二阶段：机器人条件化 trav_cost

```text
保留特征 -> 重建器 -> 可查询补全特征 H
机器人能力参数 r -> 参数编码器 -> 条件特征
H(Tomogram 查询坐标) + 条件特征 -> 适配器 / cost 头 -> Tomogram trav_cost
```

本版学生条件固定为 `[clearance_min, slope_max, step_max, stand_ratio]`，分别控制水平安全距离、允许坡度、台阶高度和可站立网格比例要求。interval_free 在教师侧固定为 0.5 m。教师侧映射和固定项见 [教师—学生参数对照](TEACHER_STUDENT_PARAMETERS.md)。

教师保留完整 VST 参数，配置为 `common/vst_global_planner/configs/teacher_building.json`。kernel_size=7、standable_ratio=0.1、inflation=0.1 等参数全部生效。interval_min=0.5 与 interval_free=0.5 独立保留，学生不再调节净空参数。

同一建筑需要在多组机器人条件下生成不同的 VST 目标。仅提供一种机器人条件，不能监督模型学会随条件改变代价。第一版输出定义为对应参数下的 VST 代价近似，不将该标签直接称为真实机器人执行成功率。

第二阶段先用干净目标几何建立基线，再输入第一阶段实际修复结果，测量并适配上游残余噪声、删除错误和补全错误。cost 监督来自干净参考几何对应的教师目标。

## 大点云与坐标接口

- 原始 PCD 保留米制世界坐标。分块保存世界原点和点索引，模型内部使用块内坐标，融合后恢复世界坐标。
- 重力方向和机器人尺度在所有数据增强、几何目标及 cost 标签间同步。用于 cost 的真实尺寸保持可恢复。
- 编码器采用体素采样和可变长合批。粗尺度提供上下文，楼梯边缘、薄栏杆由足够精细的分辨率处理。
- 输出结构 PCD 带实测/生成来源信息。未知空间记录在配套空间表示中，不通过随意添加表面点表示。
- 最终点云输出 `[x,y,z,trav_cost]`，cost 为 `[0,1]`。A* 的点云邻接与 GPMP 的连续代价查询在规划器内部完成。

## 执行顺序

1. 固定工作空间文件架构，明确源码、工具、数据和运行产物的职责。
2. 定义已有修复 PCD、机器人能力参数与 VST 监督之间的数据契约。
3. 确定 cost 的预测位置、编码器输入和机器人条件化结构，再实现第二阶段训练。
4. 在独立建筑与机器人条件上评价 cost，并可视化与 GT 的差异。
5. 直接使用已有完整三维代价点云，完成三维代价点邻接图、A*、三维 GPMP 与 RViz 展示。
6. 再完成第一阶段清理与结构修复，接入相同 PCD 接口评价完整流程。

当前 cost 已完成训练和新 seed 测试，三维点云规划已接入。filter → recon 处于部件数据与训练方案阶段，尚未训练。

## 代码依据

- `utils/PointTransformerV3/src/point_transformer_v3/presets.py`：本地模型预设与任务头类别。
- `utils/PointTransformerV3/src/point_transformer_v3/heads.py`：编码器多尺度特征恢复。
- [Utonia 官方推理接口](https://github.com/Pointcept/Utonia#quick-start)：预训练编码器、体素采样和特征映射。


| 步骤 | 具体操作 | 产物与检查内容 |
|---|---|---|
| **1. 生成场景 ✅** | 固定一个 seed，用 Terrain Generator 生成几何 | `terrain.obj`，先看平台、楼梯、桥和障碍是否符合预期 |
| **2. 转成点云 ✅** | 在 Mesh 表面采样，生成完整场景点云 | `terrain.pcd`，检查结构和采样密度。无需路线或 LiDAR 采集 |
| **3. 确定机器人条件 ✅** | 学生顺序：`clearance_min、slope_max、step_max、stand_ratio`，教师 interval_free=0.5 | 基准加三组单参数变化，旧 tall_robot 留作教师诊断 |
| **4. 生成 Tomogram 教师监督 ✅** | 同一份 PCD，分别按各组条件运行 VST，直接生成原生代价点云 | `tomogram_<ID>.pcd`，字段为 `[x,y,ground,cost/50]`；pickle 保留为中间地图 |
| **5. 可视化监督 ✅** | 展示各组原生 Tomogram 代价点云 | `tomogram_overview.png`，核对楼层、边界和条件影响 |
| **6. 多批次生成与分块 ✅** | 三批六个 seed 按场景划分 train/val/test。输入几何与 Tomogram 独立分块，6×6×4 m 核心、1.5 m 上下文 | manifest、blocks.json、521 个块、mmap 读取和可变点数合批，每场景附分块图 |
| **7. 接入代价模型 ✅** | 局部/全楼几何 → 共享 PTv3；Tomogram 三维近邻特征 → 四维条件 FiLM → CostHead | 已完成 GPU 训练，新 seed 20261127 MAE 4.40/50，完整建筑逐点输出已核验 |

当前数据统一在 `datasets/wfc_capability_dr_v2/scene_<seed>/`。每个 seed 的占地面积保留原参考的 40%，楼梯井与落脚平台重新紧凑排布。保留 3–4 层、每对楼层五条连接，以及米制楼梯宽度、步高和栏杆。每处楼梯或斜坡独立采样 XY 和 0°、90°、180°、270° 朝向。`terrain.pcd` 是世界坐标系下完整 Mesh 的 3 cm 表面采样，binary float32 XYZ。栏杆、楼板上下表面和源 Mesh 的拼接接触面均参与采样，未执行布尔并集。seed 20261109 为 8,909,815 点。

复现入口与产物说明见 [Mesh 转点云](../common/geneator/README.md#第二步mesh-表面转成点云)。完整 PCD 再以 5 cm 体素降采样，seed 20261109 得到 2,223,167 点的 `terrain_downsampled.pcd`，作为该场景所有教师的共用输入 PCD。

第三、四步按 `Traversability/configs/robots.json` 的 nominal、wide_clearance、low_slope、low_step、tall_robot 五组条件生成同名 `vst_*.pickle`。基准条件为 `[0.5,0.4,0.3,0.5]`，四组对照各只改变一个参数。完整原生教师参数保留，距离曲线 p=2。几何建图复用，保存全部切片的原始 cost、ground、ceiling 和代价梯度。来源文件和 PCD 哈希写入每组 pickle。条件与产物见 [参数对照](TEACHER_STUDENT_PARAMETERS.md#当前基准与单参数对照)。

以上为历史三参数教师对照。当前四参数版本见 `datasets/wfc_capability_dr_stand_v3`，每个场景生成24组学生条件（含联合随机组合），输入为 `[clearance_min,slope_max,step_max,stand_ratio]`，interval_free固定0.5m；tall_robot仅作教师诊断。

VST 原生 `/tomogram` 直接提供有效地面候选网格的三维位置与代价。紧凑 seed 20261109 每组 199,477 点，其他 seed 的点数见 manifest.json。世界 XY 为网格坐标，Z 为 ground，相邻 ground 层按原生规则合并并传递最小代价，再将 cost 除以 50。零代价和障碍代价均保留，未知位置不生成点。

建筑输入 PCD 与 Tomogram 监督是两个不同的点集。原有最近 ground 层回查原始 PCD 的步骤已删除。两个点集已完成独立分块，共 521 个核心块，全部 1,463,543 个原生查询各归属一个核心。分块缓存保留五条件的 7,317,715 个 cost 标签，零代价和障碍代价均保留。输入上下文允许重叠，查询核心不重复。

部署目标为建筑 PCD + 机器人参数 -> 完整建筑三维代价点云。Tomogram XYZ 仅提供训练监督位置，部署查询直接取建筑原始点，输出全部原始 XYZ 和对应 cost。所有坐标基于同一地图原点，不是 LiDAR 或机器人局部坐标。

独立执行分块用 `/home/cjh/glocal2-nav/.venv/bin/python -m Traversability.data.blocks`，完整生成入口在最后自动执行同一步。缓存按场景保存在 blocks/，根索引为 blocks.json，读取和合批接口见 [分块数据契约](TRAVERSABILITY_BLOCKS.md)。Tomogram 查询模型已接入，局部 K=8、全楼 K=4，几何近邻预计算一次，不改动监督 cost。实现与真实 GPU 前向结果见 [代价模型](TRAVERSABILITY_MODEL.md)，训练留在后续阶段。
