# Pure3D

大型建筑点云清理、结构修复与机器人条件化三维代价地图。

当前先做：**建筑 PCD + 机器人能力参数 → VST trav_cost 监督训练**。

| 目录 | 用途 |
|---|---|
| `Traversability/` | 条件化 cost 的数据读取、模型、训练、推理和绘图，代码直接放一层 |
| `Reconstruction/` | 点云清理与修复，保留已有教程和特征示例 |
| `common/` | VST、场景生成工具 |
| `planner/` | 最终代价地图与规划器的接口 |
| `datasets/` | 点云、VST 监督、数据划分和缓存，每个建筑的数据放一起 |
| `utils/` | PTv3、共用点云 I/O、坐标、分块和编码器封装 |
| `runs/` | 每次实验一个目录，集中放权重、日志和预测结果 |
| `docs/` | 架构、数据契约和训练说明 |
| `ptv3_tomography/` | 旧单帧工程参考 |

详细约定见 [文件架构](docs/WORKSPACE_LAYOUT.md)。整体目标见 [建筑 PCD 流程](docs/BUILDING_PCD_PIPELINE.md)。

真实点云：`datasets/rot_stairs/raw.pcd`。已有 Notebook 和 `Reconstruction/test.py` 中的路径已同步。

当前处于方案分析阶段，正式 cost 模型尚未训练。此前的草稿与三组 VST 标签保存在 `runs/conditioned_cost_local_20261107/`。

本地沿用 `/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python`。VST 标签生成继续使用已有 CuPy 环境。
