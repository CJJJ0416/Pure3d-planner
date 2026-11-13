> 2026-11-10：当前已扩展stand_ratio四参数及lambda=1训练。下文第51轮best.pt及三维接口结果属于旧三参数基线，四参数权重保存到runs/cost_stand_lambda_20261110。

四参数、lambda=1的新模型已完成20轮训练，最佳第13轮。两个留出测试建筑Cost MAE由4.6153/50降至3.7727/50，条件差异MAE由0.0626降至0.0493。权重与完整记录见 四参数训练报告（历史结果已清理）。

# 第 7 步：条件化 cost 模型

已实现并在本机 RTX 4060 上完成短批次训练、完整数据训练与新 seed 测试。编码器加载 Utonia Tiny 预训练权重，查询聚合、FiLM 和 CostHead 从新参数开始训练。当前最佳权重为 runs/cost_full100_20261107/best.pt，完整第 51 轮。

部署目标固定为：地图原点坐标系下的建筑 PCD + 机器人四维参数 -> 每个建筑原始点的 [x,y,z,trav_cost]。Tomogram XYZ 只用于训练和监督评估时查询 cost，不是部署输入。部署输出继承建筑 PCD 的全部 XYZ、点数和原始行序。

## 实际结构

```text
局部建筑输入 P_local [N,3] → 5 cm 体素 → PTv3/Utonia → 多尺度特征 → Linear(C,256)
全楼粗几何 P_global [G,3] → 50 cm 体素 → 同一套编码器与投影层

三维查询位置 S [L,3]：训练 S=Tomogram XYZ，部署 S=建筑 PCD 全部 XYZ
  ├─ 局部 8 个三维近邻 → 特征 + 相对 XYZ + 距离 → 邻域 Attention → f_local
  └─ 全楼 4 个三维近邻 → 特征 + 相对 XYZ + 距离 → 邻域 Attention → f_global

concat(f_local, f_global) → Linear(512,256) + LayerNorm + GELU → f_query [L,256]

条件 r [B,K,4] / condition_scale → MLP(4,64,512) → γ、β
f_conditioned = (1 + γ) × f_query + β
MLP(256,128,64,1) → Sigmoid → cost [L,K]
```

局部和全楼分别前向，共享同一套编码器权重。Utonia Tiny 多尺度特征为 486 维，Small 为 972 维，两者都投影到 256 维。所有 K 组条件共用查询特征，改变机器人参数只重新执行 FiLM 和 CostHead。

学生条件为 `[clearance_min,slope_max,step_max,stand_ratio]`，归一化尺度为 `[1.0,0.5,0.3,1.0]`。教师interval_free固定0.5m，当前每场景K=24组学生条件，含多档单参数与联合随机组合。旧三参数数据和权重保留作对照。

## 查询位置如何取得特征

训练输入点和 Tomogram 监督查询点是不同点集。部署以建筑输入点自身作为查询位置。两者使用同一个三维特征查询和 CostHead，不需要 XYZ 生成头。预处理只根据几何计算三维近邻：

1. 每个核心块内，对每个查询找上下文建筑点的 8 个最近邻。保存块内原始输入行号。
2. 每个查询再找全楼粗几何的 4 个最近邻。保存粗点云行号。
3. PTv3 prepare 返回输入点到体素代表点的 inverse。查询的体素特征索引为 `inverse[query_neighbors]`。
4. 聚合使用这些特征及原始近邻点到查询位置的相对 XYZ、欧氏距离。

inverse 只负责输入点特征恢复，不直接把查询点当作输入点。模型也不展开完整 `[N,486]` 原始点特征，而是按查询近邻直接取体素特征。

近邻表在生成数据时计算一次，训练和前向不重新进行 KNN，不读取 cost 决定邻域，不增加查询有效性筛选。现有数据补充近邻表：

```bash
/home/cjh/glocal2-nav/.venv/bin/python -m Traversability.data.blocks --neighbors-only
```

配置为 `Traversability/configs/dataset.json` 中 `blocks.query_neighbors`。全楼索引保存为 `blocks/global_neighbors.npy`，块内索引为 `block_*.npz` 的 query_neighbors。完整生成/分块入口自动准备这两组表。

## 数据与坐标

`Traversability.data.dataset.PointBlocks` 是接入模型的读取入口，`collate_blocks` 分别拼接局部/全楼点，并为查询索引添加各自体素偏移。不同块的近邻不会混入其他样本。

- 输入、查询、全楼粗几何共用核心块中心，保持米制，不统一缩放。
- PTv3 prepare 显式使用 scale=1、center=False。缺少的 RGB 和法向统一传零。
- 训练数据增强只绕 Z 轴旋转，三类坐标同步旋转。三维刚体旋转不改变已保存的近邻关系。
- target 为 `[M,K]` 原生归一化 cost，只用于损失。模型不读取 target、scene ID 或教师 cost 图。
- 查询头按 point_chunk_size 分段计算，保留全部查询点。分段不改变监督范围。
- 训练配置已指向新数据，旧输入点解码与任意点数截断代码已移除。模型已完成训练，见文末实验结果。

输入 PCD 和训练 Tomogram 的 XYZ 均基于同一个地图原点，不是 LiDAR 或机器人局部坐标。分块的 origin 仅是内部计算的平移，既不重新定义地图原点，也不缩放建筑。训练查询恢复地图坐标为 `map_xyz = query_coord @ rotation + origin`。部署导出直接使用原始建筑 PCD 数组，保证 XYZ 与输入逐项相同。

## 只运行前向

```bash
HF_HUB_OFFLINE=1 /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Traversability.training.forward --indices 0 171
```

forward.json 和 forward.npz 保存到 runs/cost_model_forward/。入口不建立优化器，不更新参数，不保存训练 checkpoint。默认使用本机缓存的预训练 Tiny 编码器。

首次四维条件接口的实际结果（历史记录，当前已改为三维）：

| 项目 | 结果 |
|---|---:|
| 输入块 | seed 20261109 和 20261111 各一块 |
| 局部建筑点 / 体素 | 106,964 |
| 全楼粗几何体素 | 35,773 |
| 查询位置 | 2,889 |
| 查询特征 | `[2889,256]` |
| 五条件 cost | `[2889,5]` |
| 复用同一特征输出单条件 | `[2889,1]` |
| 模型参数 | 9,825,099 |
| CUDA 分配峰值 | 0.414 GiB |

输出全部有限，实际网络调用中已移除 target。第一次前向约 4.94 秒，包含首次 CUDA/Triton 编译，不代表训练吞吐。新任务头尚未训练，输出不能用于评价预测精度。

部署入口 predict.py 只接收 --pcd 建筑文件和 --condition 按checkpoint的condition_names输入，当前为四维参数。inference.py 将全部建筑原始点用作输出位置，准备无标签三维分块和近邻，QueryBlocks/collate_queries 前向。输入点与输出位置共享同一份 XYZ 缓存。不读取 Tomogram、slice、ground 或训练标签。旧 --queries、--scene、--data 推理入口已移除。Traversability.data.queries 仅保留为教师监督提取或诊断工具。规划器暂不修改。

## 三维条件接口验证

2026-11-07 已将学生条件改为三维，更新训练、评估、预测和 checkpoint 元数据。教师完整参数与五组原始产物保持原样，数据适配器选取 interval_free=0.5 的四组，查询点数量不变。

```bash
HF_HUB_OFFLINE=1 /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Traversability.training.forward --indices 0 171 --output runs/cost_model_forward_3conditions
```

实际条件 `[2,4,3]`，target `[2889,4]` 与原始前四组教师标签逐项一致。RTX 4060 前向输出 `[2889,4]`，全部有限，仍可复用几何生成单条件 `[2889,1]`。参数量 9,825,035，CUDA 分配峰值 0.414 GiB。本次前向约 0.87 秒，已有内核缓存，不能与首次编译耗时或训练吞吐混用。优化器步骤为 0，没有保存 checkpoint。

以上为历史三参数结果。当前四参数训练启用条件差异损失，lambda=1.0：L=SmoothL1(pred,GT)+lambda*SmoothL1(pred(k)-pred(nominal),GT(k)-GT(nominal))。该项学习教师的相对代价变化，不强制全局单调。

## 当前部署入口

```bash
HF_HUB_OFFLINE=1 /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Traversability.inference.predict \
  --checkpoint runs/cost_full100_20261107/best.pt --pcd /path/to/building.pcd \
  --geometry-python /home/cjh/glocal2-nav/.venv/bin/python \
  --condition 0.5 0.4 0.3 --output runs/new_building_prediction
```

使用实际训练 checkpoint，例如 runs/cost_full100_20261107/best.pt。只验证接口时用 --pretrained 替换 --checkpoint PATH，加载现有预训练编码器和未训练头，不保存训练权重。输出 prediction.pcd、prediction.npz、prediction.json，坐标系保持 map 原点。训练 checkpoint 记录 blocks_config，使部署复用三维分块与近邻配置。

旧接口试验和独立推理 run 已清理。当前保留最新完整训练及其新场景评估，结果如下。

## 完整数据实验

用户将原 100 轮计划改为第 50 轮截止。停止时第 51 轮已完成，第 52 轮在验证前中断。准确测试完整第 51 轮 best.pt，没有第 50 轮独立权重。固定验证 MAE 为 4.51/50，新 seed 20261127 为 4.40/50。新 seed 危险低估率 6.35%，条件差值 MAE 0.0522。输入建筑 2,153,446 点全部输出，坐标和原点保持不变。完整记录见 报告（历史结果已清理）。

## 真实 rot_stairs 点云推理

输入为 datasets/rot_stairs/raw.pcd，共 5,886,268 点。使用同一 best.pt 和条件 [0.5,0.4,0.3]，全部原始点均输出 cost，地图 XYZ 和原始点序逐项一致。该独立推理 run 已清理，原始 datasets/rot_stairs/raw.pcd 和最新训练权重保留，可重新预测。近邻准备约 26.94 秒，推理与数据加载约 11.30 秒，CUDA 分配峰值约 2.18 GiB。真实数据没有参与训练，本次没有生成教师标签或计算准确率。

真实扫描有 9 个上下文块不足 8 个点，涉及 22 个输出位置。KNN 表对空邻居作索引填充，并提供 query_neighbor_mask；Attention 只使用实际存在的邻居。此有效位不筛选任何点或 cost 标签，模型参数和已训练权重不变。训练块原有 8 个近邻全部有效。

RViz 已发布和接收全部 5,886,268 点，话题 /pure3d/prediction，字段 x/y/z/trav_cost，固定帧 map，色标 0–1。该次截图随旧推理 run 一并清理。
