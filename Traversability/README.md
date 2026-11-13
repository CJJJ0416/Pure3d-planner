# Traversability

完整建筑 PCD 与四维机器人条件，生成地图坐标系下的可通行代价点云。

唯一训练指南：[冻结 Small 完整训练与服务器迁移](docs/TRAINING_WORKFLOW.md)。涵盖服务器环境查询、依赖安装、代码与权重迁移、训练数据生成、runs 命名、冻结训练、恢复、评估和独立推理。

当前实验使用官方 Utonia Small 预训练编码器，始终冻结并缓存 972 维特征。Adapter、四参数 FiLM 和 CostHead 随机初始化后训练。模板为 [full3d_small_frozen.json](configs/full3d_small_frozen.json)，预训练权重为 `weights/utonia_small.pth`。

完整三维 PCD 参与编码，GPU 体素哈希将原生 Tomogram 查询对应到真实输入点。条件顺序为 `[clearance_min, slope_max, step_max, stand_ratio]`，教师固定 `interval_free=0.5 m`，cost 归一化到 0–1。当前 `inference.full3d` 输出原生几何支撑 XYZ 与预测 cost。

| 目录 | 用途 |
|---|---|
| `configs/` | 数据、条件与实验配置 |
| `data/` | 场景、原生教师、分块与 GPU 对应 |
| `models/` | 特征任务头与 FiLM |
| `training/` | 特征缓存、冻结训练与恢复 |
| `experiments/` | 训练、恢复、独立推理和报告编排 |
| `inference/` | 新建筑推理 |
| `visualization/` | 报告、图表与本机 RViz |
| `docs/` | 单份完整训练指南 `TRAINING_WORKFLOW.md` |

从工程根目录执行模块。服务器解释器和数据路径按指南修改后再启动。训练数据保存在 `datasets/<RUN_NAME>/`，产物保存在 `runs/traversability/<RUN_NAME>/`，名字记录 `full3d`、`small`、`frozen`、实际轮数、seed、时间和实验标签。

Small 预训练权重已下载并通过 SHA256 校验。当前 Small 配置尚未启动正式训练。
