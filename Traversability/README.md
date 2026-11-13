# Traversability

建筑地图坐标系 PCD + 机器人参数 → 完整三维 `trav_cost` 点云。

2026-11-13 在线微调：[当前完整3D结构、Tiny从第1步参与反向、同起点同20个seed重训50轮](docs/FULL3D_TINY_UNFROZEN50.md)。几何与索引缓存，特征在线计算，旧冻结权重保留。

2026-11-13 完整3D输入：[冻结Tiny、GPU体素哈希监督对应、每10轮替换4栋建筑](docs/FULL3D_TINY_ROTATING100.md)。从原第90轮best初始化任务头，固定标签保留并重建特征缓存。本次按用户要求第50轮停止并测试，原100轮学习率调度保留。

2026-11-13 续训：[冻结Tiny、每5轮替换4个WFC建筑、累计训练至100轮](docs/SLICED_TINY_ROTATING100.md)。固定验证／测试复用，源50轮权重保留。

2026-11-13 新路径：[共享切片XYZ、冻结Tiny、本机50轮](docs/SLICED_TINY_LOCAL50.md)。输入和监督共用原生Tomogram几何，输出为切片支撑XYZ＋cost，去掉KNN查询聚合。运行记录在`runs/traversability/cost_sliced_tiny_local50_20261113/`。下文旧查询模型与历史权重保留作为参考。

完整操作流程见 [TRAINING_WORKFLOW.md](docs/TRAINING_WORKFLOW.md)。当前四参数解耦训练见 [PARAMETER_DECOUPLING_V4.md](docs/PARAMETER_DECOUPLING_V4.md)，实时状态位于 `runs/traversability/cost_parameter_decoupling_v4/progress.json`。上一轮全类型楼梯增强方案保存在 [ALL_STAIRS_DR_TRAINING.md](docs/ALL_STAIRS_DR_TRAINING.md)。

## 文件分类

| 目录 | 内容 | 主要入口 |
|---|---|---|
| `configs/` | 场景批次、教师条件、训练配置 | `dataset_response.json`、`robots_response.json`、`train_response.json` |
| `data/` | 场景调度、VST 监督、Tomogram 导出、分块与数据加载 | `generate.py`、`teacher.py`、`prepare.py`、`blocks.py`、`dataset.py` |
| `models/` | 共享 PTv3、QueryPool、FiLM、CostHead | `model.py` |
| `training/` | 训练、条件差值损失、评估、前向诊断 | `train.py`、`condition_response.py`、`evaluate.py`、`forward.py` |
| `inference/` | 新建筑几何准备与全部原始点推理 | `inference.py`、`predict.py` |
| `visualization/` | GT 对比图、overview、RViz 点云对比、文字进度JSON | `visualize.py`、`overview.py`、`rviz.py`、`compare_rviz.py`、`progress.py` |
| `experiments/` | 楼梯专项数据、固定连通指标及批量实验 | `stair_data.py`、`stair_metrics.py`、`stair_experiment.py`、`stair_report.py` |
| `docs/` | 当前完整流程和历史设计参考 | `TRAINING_WORKFLOW.md` |

从工程根目录通过模块运行，例如 `python -m Traversability.training.train`。分类后不再使用旧的 `python -m Traversability.train`。

## 当前保留的权重

`/home/cjh/Pure3D_ws/runs/cost_all_stairs_dr_v3/best.pt`是当前四参数解耦实验的初始化权重，来自全类型楼梯增强第20轮。新实验采用24个独立测试seed及原地图8处楼梯回归，原权重保留。

`/home/cjh/Pure3D_ws/runs/cost_wide_stair_v2/best.pt`是此前宽深高直梯增强第10轮权重，保留为历史对照。

`/home/cjh/Pure3D_ws/runs/traversability/best.pt`是原C_geometry_conditions基线，第10轮、step1710。其原始选择标准记录在`runs/traversability/model.json`。

上一轮增强的独立测试Cost MAE为2.23/50，已知目标楼梯在固定图诊断中连通，但其它部分楼梯的误阻断仍偏高。几何与前后误差见`runs/cost_wide_stair_v2/all_stairs_audit/REPORT.md`。旧数据清理之后新增的增强运行单独保留在各自runs目录。

## 数据与工具

学生条件固定为 `[clearance_min, slope_max, step_max, stand_ratio]`。教师条件为 `[clearance_min, slope_max, step_max, interval_free, stand_ratio]`，`interval_free=0.5 m`。

- 输入 PCD 与输出预测使用地图原点坐标系，单位为米。
- 教师监督直接来自原生 Tomogram XYZ + cost，cost 除以 50。
- 每个原生查询点保留，几何分块和体素化不增加监督筛选。
- 推理查询位置来自输入建筑 PCD，保留全部原始 XYZ 和点序。
- 场景、标签与缓存属于 `datasets/`，实验输出属于 `runs/`。
- 通用 I/O 使用 `common/pointcloud_io/`，VST 使用 `common/vst_global_planner/`，编码器使用 `utils/PointTransformerV3/`。

大型地图回归数据集中在 `datasets/wfc_regression_20261127/`。它保存标签、机器人条件、块索引和固定楼梯回归案例，几何数组引用已有 datasets。该数据已用于问题诊断，属于回归数据。

历史设计保存在 `docs/CONDITION_RESPONSE_ABLATION.md` 与 `docs/STAIR_STAND_RATIO_FINETUNING_PLAN.md`。其中的结果目录是历史记录，当前保留文件以 `runs/traversability/` 为准。
