# Filter 训练档案：2026-11-12

## 版本与产物

版本 `filter_scoring_mesh_full_20261112`。目标是对单张融合 XYZ 点云中的残影生成连续保留系数。扫描缺失由后续 Reconstruction 处理。本轮训练和测试已经完成，用户已查看真实 ROT stair 的评分与 0.2 筛选结果。

| 项目 | 路径或数值 |
|---|---|
| 数据配置 | `Filter/configs/scoring_mesh_full.json` |
| 训练配置 | `Filter/configs/train_scoring_mesh_full.json` |
| 数据 | `datasets/filter_scoring_mesh_full_20261112` |
| 冻结特征 | `datasets/cache/filter_scoring_mesh_full_20261112` |
| 运行 | `runs/filter_scoring_mesh_full_20261112` |
| best.pt | epoch 24，step 672，38,609,094 bytes |
| last.pt | epoch 50，step 1400，38,676,230 bytes |
| best SHA256 | `e258833d2e4149ce8503bb1ecbf8ace89c49b970be6cfc8430957f2e99163b90` |
| last SHA256 | `ff3442c7f8cfc876f9cb5c98f9623d9dfe99d7f0c79130e8210f2412f858b46e` |

best 按固定验证来源均衡 Macro MAE 选择，次级比较全点 MAE。最后一轮不是最优轮。权重包含评分头、冻结编码器及恢复训练状态。已重新核对 best 的编码器所有张量与冻结缓存完全一致。

## 数据与监督

六个独立局部组件，4/1/1 场景划分。Train 为 821100 直梯、821101 往复梯、821102 旋转梯、821103 走廊。Val 为 822100 直梯，Test 为 823100 平台。训练 seed 为 42。

完整 Mesh 表面按 3 cm 间距采样，不降采样输入，不挖洞、不删条带、不主动制造断连。四变体为干净静态、0.5% 稀疏悬空、1% 悬空加短拖影、1.2% 悬空加少量运动残影。完整静态参考包括静止物体。

运动点目标为 0。其余点目标为 exp(-0.5*(到完整静态 Mesh 的三角面距离/0.03)^2)。离线标签和增强来源不进入模型。分块边界不参与标签判断。

总计 24 个变体、333 个空间盒、2,175,141 个唯一核心点。训练 1,490,564 点，验证 338,322 点，测试 346,255 点。核心 4×4×3 m、上下文外扩 1 m，训练和部署均使用 map 零相位网格。原始点与全部核心索引逐点对齐的记录为数据目录的 preparation_check.json。

## 模型与训练

官方 Utonia Tiny 冻结/eval，多尺度 486 维。可训练投影 486→128，几何分支 9→32→32，评分头 160→128→64→1，Sigmoid 输出 keep_score。Tiny 内部 2 cm 物理网格经 inverse 恢复到全部监督点。

来源组内部按结构语义均衡，再对当前存在的来源组平均 SmoothL1，beta=0.1。当前无主动断连组，来源均衡指标实际平均建筑、静止物体、悬空/配准和动态四组。浮空点计入悬空/配准来源桶，不代表均为刚体错位副本。

真实联合 batch=8、workers=4，场景均衡采样，AdamW LR 1e-3、weight decay 0.01、Dropout 0.1。5% 更新 warmup，cosine 至 1e-5。BF16 前向，float32 loss。总计 50 epoch、1400 次完整更新。

环境为 pure3d-ptv3，Python 3.11，Torch 2.10.0+cu128，NumPy 1.26.4，SciPy 1.15.3。数据依赖包括 trimesh 5.2.0、manifold3d 3.5.4、open3d-cpu 0.19.0、opencv-python-headless 4.11.0.86。本机 GPU 为 RTX 4060 Laptop 8 GiB。官方 Tiny 权重 SHA256 为 `e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137`。

## 损失与固定验证

| 项目 | 初始/第1轮 | best 第24轮 | last 第50轮 |
|---|---:|---:|---:|
| 训练 epoch 平均 loss | 0.330844 | 0.022648 | 0.012404 |
| 训练全点 MAE | 初始 0.573588 | 0.023278 | 0.008162 |
| 验证全点 MAE | 初始 0.572311 | 0.010328 | 0.005945 |
| 验证 Macro MAE | 初始 0.494278 | **0.026427** | 0.092267 |

训练 loss 和全点 MAE 显著降低。验证动态组只有 293 点，其 MAE 从 best 的 0.07555 升至 last 的 0.34948，导致 Macro 指标回升。因此部署沿用 best24。

## 独立合成测试

测试 seed 823100，四变体，共 346,255 点。没有使用测试数据选择权重或操作阈值。

| 指标 | best24 | 全点打 1 分基线 |
|---|---:|---:|
| 全点 MAE | 0.016872 | **0.008037** |
| 来源均衡 Macro MAE | **0.020233** | 0.495335 |
| 全点 RMSE | 0.093782 | 0.086662 |

各组 MAE：建筑 0.017834，静止物体 0.007373，悬空/配准 0.012651，动态 0.043075。模型均衡指标明显优于全保留基线，但全点 MAE 仍较差，真实结构评分仍有偏差。独立合成测试只有一个建筑 seed，本轮不能量化广泛跨建筑泛化。

## ROT stair 与 0.2 操作阈值

真实输入为 `datasets/rot_stairs/raw.pcd`，5,886,268 点。best24 在 424 个窗口中恢复所有原始 XYZ、点序与分数。没有逐点 GT，仅做可视化和下游对照。

用户指定 keep_score≥0.2 后，保留 5,328,865 点，删除 557,403 点，删除比例 9.46955%。0.2 是操作阈值，不是经过验证集选出的最优阈值。外围散点显著减少，主体通道仍保留，仍存在误删结构边缘与保留漂浮片段。

随后使用 `runs/traversability/best.pt`，条件 [clearance_min,slope_max,step_max,stand_ratio]=[0.5,0.4,0.3,0.1]，对保留点全部预测 cost。对应结果为 `predictions/rot_stair_traversability_tau020/`，同点索引和前后同权重对照已核对。

middle→lower 的支撑查询为 (0,4,3.97424578666687) 到 (0,4,-0.10288545489311218)，map 米制坐标：

| 方法与输入 | 结果 |
|---|---|
| 过滤后预测 cost + DC，阈值 0.5 | A* 30.747 m，GPMP 29.384 m，优化收敛 |
| 过滤后同一 ROI 点云 + 原生 VST15/25 | 两者均在 A* 阶段无路 |
| 未过滤原始点云 VST15 | A* 无路 |
| 未过滤原始点云 VST25 | A* 31.594 m，原生 GPMP 返回 27.661 m 轨迹 |

规划对照位于 `runs/planner/rot_stair_filtered020_middle_lower_20261112/`。两条成功路线输入、代价来源和规划器不同，不能作为只比较 DC/VST 的控制实验，也不能据此归因过滤后 VST 无路的具体原因。两种 GPMP 的路线参考高度均为 +0.5 m。原生 VST 未暴露收敛标志，DC 的速度惩罚是软约束，本次未做执行轨迹验收。

## 时间口径与后续使用

- 数据生成 14.98 s。冻结编码缓存循环 37.90 s，不含环境启动和模型加载。
- 50 轮训练循环累计 50.64 s，含循环内周期性保存，不含逐轮评估和轮末保存。CUDA 分配峰值 1.88 GiB。
- ROT 全点推理 283.94 s，其中 CPU 完整 KNN 统计 264.24 s，编码 10.75 s，评分头 1.24 s。端到端约 20,731 点/s，CUDA 分配峰值 0.925 GiB。
- 筛选后 Traversability 前向 8.22 s，不含 CPU 几何准备。原始点云同权重前向 11.82 s。

冻结并保留 best24 作为本轮评分头基线。当前不启动新训练、不改损失或阈值。后续优先扩大独立真实/合成结构评价，并检查低评分边缘对原生 VST 连通性的影响。

曲线和逐点图在本轮 runs/plots，实测环境、指标和逐轮记录分别见 environment.json、convergence_report.json、history.json。权重校验身份在 checkpoint_identity.json。当前可复现命令见 [操作流程](TRAINING_WORKFLOW.md)。
