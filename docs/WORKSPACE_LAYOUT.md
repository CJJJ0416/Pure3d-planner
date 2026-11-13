# Pure3D 文件架构

更新日期：2026-11-13。根目录：`/home/cjh/Pure3D_ws`。

```text
Pure3D_ws/
├── Traversability/
│   ├── configs/          # 数据批次、教师条件与训练设置
│   ├── data/             # 生成、教师、Tomogram、分块和数据加载
│   ├── models/           # PTv3 + QueryPool + FiLM + CostHead
│   ├── training/         # 训练、损失、评估和前向入口
│   ├── inference/        # 新建筑几何准备和完整原始点推理
│   ├── visualization/    # GT/预测图与 RViz
│   ├── experiments/      # 楼梯数据、连通指标和实验调度
│   ├── docs/             # 完整训练流程与历史设计
│   └── README.md
├── Filter/
│   ├── configs/          # 活动方案、数据和训练配置，历史配置另行保留
│   ├── data/             # Mesh 采样、增强、连续标签和固定特征缓存
│   ├── models/           # 投影、几何分支与连续保留评分头
│   ├── training/         # 损失、训练、评估和恢复
│   ├── inference/        # 全点保留系数与显式阈值后处理
│   ├── visualization/    # GT/预测图与 RViz
│   ├── docs/             # 架构、数据契约、操作流程与训练档案
│   └── README.md
├── Reconstruction/
│   ├── configs/          # Subtraction 配对、几何自编码与 Flow 配置
│   ├── data/             # Mesh 几何、表面标签、课程缺损与冻结缓存
│   ├── models/           # 空间编码、自编码几何场与条件 Flow
│   ├── training/         # 几何损失、训练与评估
│   ├── inference/        # 表面生成与窗口融合
│   ├── visualization/    # 已有悬空可疑度工具，后续重建对比
│   ├── experiments/      # 预留：GT-only / GT+KD 与 Tiny/Small 对照
│   ├── docs/             # 架构、数据契约、训练说明及旧方案参考
│   └── README.md
├── common/
│   ├── pointcloud_io/    # PCD 与 ROS PointCloud2 I/O
│   ├── Mid360_imu_sim/   # MID-360 模型与方向表
│   ├── scanning/        # 增量扫描、IMU、LIO、GT 和原生可视化
│   ├── vst_global_planner/
│   └── geneator/         # 原生 terrain-generator、WFC 与 Mesh/PCD 工具
├── planner/              # PCD → DC跨阶图 → A* → 反重力GPMP
├── datasets/             # 几何、标签、数据划分、固定回归案例与缓存
├── utils/                # PTv3 与共用几何工具
├── runs/                 # 各实验权重、日志、预测点云与对比图
├── docs/                 # 工程整体架构与跨模块契约
└── ptv3_tomography/       # 旧工程参考
```

## 代码入口

以工程根目录为工作目录，使用分类后的完整模块名。

| 工作 | 入口 |
|---|---|
| 批量生成 | `python -m Traversability.data.generate` |
| 原生教师 | `python -m Traversability.data.teacher` |
| 分块/近邻 | `python -m Traversability.data.blocks` |
| 训练 | `python -m Traversability.training.train` |
| 评估 | `python -m Traversability.training.evaluate` |
| 完整点云推理 | `python -m Traversability.inference.predict` |
| 绘制 GT 对比 | `python -m Traversability.visualization.visualize` |
| RViz | `python -m Traversability.visualization.rviz` |
| 多模型 RViz | `python -m Traversability.visualization.compare_rviz` |

完整命令与环境见 [Traversability 训练流程](../Traversability/docs/TRAINING_WORKFLOW.md)。

Filter 已完成全点评分数据、特征缓存、训练、测试与独立推理。活动入口为 `Filter.data.scoring`、`Filter.data.features`、`Filter.training.train`、`Filter.training.evaluate`、`Filter.inference.predict` 和显式后处理 `Filter.inference.threshold`。命令见 [操作流程](../Filter/docs/TRAINING_WORKFLOW.md)，实测见 [训练档案](../Filter/docs/TRAINING_ARCHIVE_20261112.md)。

Reconstruction 按相同八类目录组织。当前设计入口为 [README](../Reconstruction/README.md) 和 [plan.json](../Reconstruction/configs/plan.json)。当前使用完整 Mesh 监督的几何自编码与条件 Flow Matching，数据、训练、评估及 PCD 导出入口已实现。已迁移的几何工具使用 `python -m Reconstruction.visualization.inspect_floating`。

## 数据与结果

每个数据版本集中保存场景、输入 PCD、原生 Tomogram 标签与 blocks.json。按完整建筑或几何族划分 train/val/test。`datasets/cache/` 保存可复用缓存。

回归数据 `datasets/wfc_regression_20261127/` 保存大型测试建筑的条件标签和固定楼梯案例，其几何数组引用 datasets 中已有场景，不依赖 runs。

训练与推理写入独立的 `runs/<experiment>/`。Filter 当前数据位于 `datasets/filter_scoring_mesh_full_20261112/`，冻结特征位于 `datasets/cache/filter_scoring_mesh_full_20261112/`，结果位于 `runs/filter_scoring_mesh_full_20261112/`。训练过程中只使用 best.pt 和 last.pt，不增加按任务或阶段层层嵌套的 checkpoint 目录。

Reconstruction 当前使用连续 WFC 大地形的整图缺损，再按固定核心和上下文分块。原始地形与 RViz 预览保留在 `datasets/recon_terrain_blocks_preview_v1`，新的块监督和特征接口位于 `datasets/recon_terrain_curriculum_v1`。课程、Loss 和训练配置集中于 `Reconstruction/configs`。旧扫描、小样本、独立楼梯与语义挖空数据已清理。当前未训练，详情见 [RECON](../Reconstruction/README.md)。
