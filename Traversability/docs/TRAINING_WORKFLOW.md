# Traversability cost 完整训练流程

更新日期：2026-11-10。工程目录：`/home/cjh/Pure3D_ws`。

当前宽深高直梯增强任务已启动，实时状态见 `runs/cost_wide_stair_v2/execution.json`，方案见 [WIDE_STAIR_AUGMENTATION.md](WIDE_STAIR_AUGMENTATION.md)。本文是通用操作说明，示例命令不会自动执行。

## 1. 输入、监督与输出

```text
terrain-generator 多层建筑 Mesh
  ↓ 3 cm 表面采样、5 cm 体素降采样
建筑 PCD（地图坐标系）
  ├─ + 教师机器人条件 → 原生 VST → Tomogram XYZ + cost/50
  └─ 局部精细几何 + 全楼粗几何 → 共享 PTv3
                                      ↓
                         在 Tomogram XYZ 查询几何特征
                                      ↓
                         FiLM（学生机器人条件）→ CostHead
                                      ↓
                             与原生教师 cost 计算损失
```

部署入口为：建筑 PCD + 学生参数 → 输入 PCD 全部原始点的 XYZ + `trav_cost`。部署时不提供教师 cost、不提供 Tomogram 查询点、不计算 slice。

学生固定顺序：`[clearance_min, slope_max, step_max, stand_ratio]`。

教师固定顺序：`[clearance_min, slope_max, step_max, interval_free, stand_ratio]`，其中 `interval_free` 固定为 0.5 m。条件值及该条件对应的教师标签必须一起使用。

输入 PCD 和教师 Tomogram 是不同点集。输入 PCD 表达建筑几何，Tomogram XYZ 指定监督查询位置。不能把两者的行索引混用，也不需要把教师代价最近邻回填到原始建筑点。

## 2. 目录与执行环境

| 目录 | 职责 |
|---|---|
| `Traversability/configs/` | 数据集、机器人条件、本地/服务器训练配置 |
| `Traversability/data/` | 场景调度、教师生成、Tomogram 导出、空间分块、数据加载 |
| `Traversability/models/` | PTv3、查询聚合、FiLM 与代价头 |
| `Traversability/training/` | 训练、损失、断点与评估 |
| `Traversability/inference/` | 新建筑全部原始点的独立推理 |
| `Traversability/visualization/` | GT 图、预测图和 RViz |
| `Traversability/experiments/` | 楼梯专项实验和连通指标 |
| `datasets/<version>/` | 输入、监督、划分与共享缓存 |
| `runs/<experiment>/` | 本次训练权重、记录和预测 |

本机环境沿用现有安装：

```bash
cd /home/cjh/Pure3D_ws
export PYTHONPATH=/home/cjh/Pure3D_ws
export MPLCONFIGDIR=/home/cjh/Pure3D_ws/datasets/cache/matplotlib
export CUPY_CACHE_DIR=/home/cjh/Pure3D_ws/datasets/cache/cupy

PTV3_PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
GEOMETRY_PY=/home/cjh/glocal2-nav/.venv/bin/python
```

`PTV3_PY` 负责训练和网络推理。`GEOMETRY_PY` 负责场景、NumPy/SciPy 几何缓存和 CuPy VST。RViz 使用 ROS 2 Humble 的系统 `/usr/bin/python3`。场景、教师、训练和绘图均可 headless 运行，RViz 仅用于本机查看。

## 3. 确定新实验的数据和划分

常规多层场景配置：

- `configs/dataset_response.json`：批次、seed、train/val/test、输出路径和几何处理参数。
- `configs/robots_response.json`：固定参数组、联合随机参数与单变量配对。
- `common/geneator/wfc_response_dr.json`：面积、楼层、楼梯、斜坡、栏杆和能力阈值案例的域随机化。
- `common/vst_global_planner/configs/teacher_building.json`：完整原生教师参数。

开始新数据实验时，复制数据配置并修改 `output` 与 `batches`。不同划分使用不同完整场景 seed。同一几何族的变体应属于同一划分。

```bash
cp Traversability/configs/dataset_response.json \
   Traversability/configs/dataset_experiment.json
```

编辑新文件，例如将 `output` 设为 `datasets/cost_experiment_v1`，并配置独立的训练、验证、测试 seed。生成入口会重建所选数据，输出目录应对应这次实验。

楼梯专项需检查生成后的实际尺寸分布。目标失败直梯的踏步高度约 0.384 m、深度约 0.617 m、宽度约 1.442 m。前次新增直梯数据仅有四个几何族，不能把同族的三个深度版本当作三套独立几何。新方案应独立随机化高度、深度、宽度、栏杆、井口、楼层上下文和朝向。

已诊断的大地图 seed 20261127 保留作回归，不并入新训练数据，也不用它选择最佳权重。

## 4. 生成场景、PCD 和原生监督

```bash
"$GEOMETRY_PY" -m Traversability.data.generate \
  --config Traversability/configs/dataset_experiment.json
```

顺序执行：Mesh → `terrain.pcd` → `terrain_downsampled.pcd` → 多条件 VST → 原生 Tomogram → 分块和近邻缓存。每个场景保存 overview，总目录保存 manifest、批次索引和 overview 合集。该命令不训练网络。

只替换一批：

```bash
"$GEOMETRY_PY" -m Traversability.data.generate \
  --config Traversability/configs/dataset_experiment.json --batch batch_00
```

单独为已有场景生成教师：

```bash
"$GEOMETRY_PY" -m Traversability.data.teacher \
  --scene datasets/cost_experiment_v1/scene_20261109 \
  --pcd terrain_downsampled.pcd \
  --robots Traversability/configs/robots_response.json \
  --teacher-config common/vst_global_planner/configs/teacher_building.json
```

场景名替换为新配置实际生成的 seed。教师保留完整原生规则，包括地图分辨率、层高、kernel_size、interval_min、inflation、standable_ratio，以及距离曲线 `p=2`。机器人条件只覆盖对应参数。

教师产物：

- `vst_<profile>.pickle`：原始 0–50 cost、梯度、ground、ceiling 和完整教师参数。
- `tomogram_<profile>.pcd`：原生合并后的 XYZ + 归一化 `trav_cost`。
- `teacher.json`、`robot_conditions.json`：条件顺序、代价尺度、输入来源和标签清单。
- `teacher_overview.png`、`tomogram_overview.png`：教师层和点云概览。

训练监督直接使用 Tomogram 点云。零代价、障碍代价和全部原生有效 ground 查询点均保留，不额外按高度连续、地表回波或邻域覆盖筛选。

## 5. 分块和数据准备

完整生成入口已包含这一步。需要单独重建时运行：

```bash
"$GEOMETRY_PY" -m Traversability.data.blocks \
  --config Traversability/configs/dataset_experiment.json
```

只补已有块的近邻：

```bash
"$GEOMETRY_PY" -m Traversability.data.blocks \
  --config Traversability/configs/dataset_experiment.json --neighbors-only
```

当前标准为 6×6×4 m 核心块、1.5 m 上下文、0.5 m 全楼粗几何、局部 K=8、全局 K=4。每个查询点只归属一个核心块，mmap 数组保存共享几何，块文件仅保存索引。近邻仅由 XYZ 计算。

`blocks.json` 组织 scenes、blocks、splits、条件名称和尺度。场景数组包括输入几何、查询位置、cost、conditions 和全局近邻。训练读取点数可变的块，合批时拼接点并维护 offset 与索引偏移。

## 6. 设置训练配置

```bash
cp Traversability/configs/train_response.json \
   Traversability/configs/train_experiment.json
```

在新配置中设置：

```json
{
  "data": "datasets/cost_experiment_v1",
  "validation_data": "datasets/cost_experiment_v1",
  "output": "runs/cost_experiment_v1",
  "epochs": 50,
  "batch_size": 2,
  "num_workers": 2,
  "cpu_threads": 4,
  "amp": "bf16",
  "freeze_encoder_epochs": 2,
  "learning_rate": 0.0001,
  "encoder_learning_rate": 0.00001,
  "condition_difference_weight": 1.0,
  "evaluation_split": "val"
}
```

这是需要修改的字段片段，其余字段保留在完整配置中。现有最优权重使用 `model.encoder=utonia_tiny`、局部体素 0.05 m、全楼粗体素 0.5 m、特征宽度 256。

`batch_size` 是真正并行的块数量。编码器学习率单独设置。当前损失为绝对 cost 的 SmoothL1 加单变量条件差值损失，四参数差值权重由 `balanced_response.parameter_weights` 指定。

`train_server.json` 的默认编码器为 Utonia Small。使用当前 Tiny checkpoint 初始化时，配置也使用 Tiny。Small 对照实验应使用相应 Small 预训练编码器，并重新训练任务头。

## 7. 从保留的最佳模型微调

```bash
HF_HUB_OFFLINE=1 "$PTV3_PY" -m Traversability.training.train \
  --config Traversability/configs/train_experiment.json \
  --initialize runs/traversability/best.pt
```

`--initialize` 读取已有模型权重，新建优化器和学习率调度，从新实验 epoch 1 开始计数。只要模型和条件契约一致，即可接续学习新数据。

从官方 Utonia 预训练编码器开始训练新 cost 头时，省略 `--initialize`。此时任务头随机初始化，加载编码器权重时可能需要已有权重缓存或网络。

训练输出直接位于 `runs/cost_experiment_v1/`：

```text
best.pt          验证标准选出的最佳完整 checkpoint
last.pt          最近保存的可恢复 checkpoint
config.json      本次实际配置
history.json     每轮损失、验证、吞吐和 CUDA 分配峰值
```

`save_every_steps` 控制定期更新 last.pt，每轮结束也保存。best 的更新由配置指标或楼梯专用选优规则决定，最后一轮不自动等于最佳轮。

前向诊断入口为 `Traversability.training.forward`，它运行预训练编码器加新任务头，不能把该输出当作已训练模型预测。

## 8. 中断与恢复

正常中断使用 Ctrl+C，训练入口会保存恢复状态。外部强制终止只能恢复到磁盘上已完成写入的最近 checkpoint。

```bash
HF_HUB_OFFLINE=1 "$PTV3_PY" -m Traversability.training.train \
  --resume runs/cost_experiment_v1/last.pt
```

`--resume` 恢复模型、AdamW、学习率、AMP、随机状态和 epoch 内进度。恢复使用 checkpoint 内保存的配置，`--config` 不覆盖这些配置。

延长总训练周期或更改输出位置：

```bash
HF_HUB_OFFLINE=1 "$PTV3_PY" -m Traversability.training.train \
  --resume runs/cost_experiment_v1/last.pt \
  --epochs 80 --output runs/cost_experiment_v1
```

当前清理保留的 `runs/traversability/best.pt` 为第 10 轮完整状态。若使用它执行 `--resume`，将从第 10 轮继续，默认输出目录已改为 `runs/traversability_resume`。更换训练数据时使用第 7 步的 `--initialize`。

## 9. 固定验证、独立测试与选优

训练根据 `evaluation_split=val` 定期验证。普通实验按 `checkpoint_metric` 选优。楼梯实验使用独立验证场景的固定邻接图，先满足错误放行与非楼梯误差标准，再比较断裂率、stand 条件差值误差和整体 MAE。

选定 best 后再执行测试：

```bash
"$PTV3_PY" -m Traversability.training.evaluate \
  --checkpoint runs/cost_experiment_v1/best.pt \
  --data datasets/cost_experiment_v1 --split test \
  --output runs/cost_experiment_v1/test.json \
  --predictions runs/cost_experiment_v1/test.npz
```

主要指标：

| 指标 | 含义 |
|---|---|
| `cost_mae_raw` | 归一化 MAE 乘 50，恢复教师代价值量纲 |
| `parameter_response` | 单参数改变时的教师/学生差值、方向与幅度 |
| `dangerous_underprediction_pct` | 教师 cost=1 的点被预测为 <0.6 的比例 |
| 楼梯断裂率 | GT 连通的固定端点对在预测图中断开的比例 |
| 错误放行率 | GT 阻断的固定端点对被预测图连通的比例 |
| 非楼梯 MAE | 普通区域能力是否退化 |

楼梯连通诊断当前使用半径 0.45 m、cost 阈值 0.3，并保留 0.3 m 半径对照。这些是点云图诊断，不是机器人执行成功率。

## 10. 绘制同一点集的 GT 与预测

```bash
"$GEOMETRY_PY" -m Traversability.visualization.visualize \
  --scene datasets/cost_experiment_v1/scene_20261114 \
  --prediction runs/cost_experiment_v1/test.npz --profile 0 \
  --output runs/cost_experiment_v1/plots/test_gt_prediction.png
```

场景名替换为实际 test seed。该入口从 evaluate 快照读取同一行序的 XYZ、GT、预测，输出统一坐标的 GT、预测和误差三联图。

建筑完整原始 PCD 的部署预测与原生 Tomogram 监督点集不同。完整部署输出用于观察地图，定量 GT 评分使用上述同点评估快照。

## 11. 新建筑 PCD 的完整独立推理

```bash
HF_HUB_OFFLINE=1 "$PTV3_PY" -m Traversability.inference.predict \
  --checkpoint runs/traversability/best.pt \
  --pcd datasets/wfc_unseen_20261127/scene_20261127/terrain_downsampled.pcd \
  --geometry-python "$GEOMETRY_PY" \
  --condition 0.5 0.6 0.4 0.05 \
  --output runs/cost_prediction
```

输出 `prediction.pcd`、`prediction.npz`、`prediction.json` 和纯几何缓存。每个输入点都有 cost，XYZ、地图原点、单位和点序保持。条件顺序与 checkpoint 的 `condition_names` 一致。

这里不调用 VST、不读取教师标签、不请求外部 Tomogram XYZ。几何分块后按原始 query_index 写回，避免因体素化造成只显示少量体素中心点。

## 12. RViz 查看

在本机加载 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m Traversability.visualization.rviz \
  --prediction runs/cost_prediction/prediction.pcd
```

查看原生教师：

```bash
/usr/bin/python3 -m Traversability.visualization.rviz \
  --scene datasets/cost_experiment_v1/scene_20261114
```

增加 `--raw-cost` 可将教师显示恢复到 0–50。

多个模型在相同查询 XYZ 上并排对照：

```bash
/usr/bin/python3 -m Traversability.visualization.compare_rviz \
  --model before "Before" runs/before/test.npz \
  --model after "After" runs/after/test.npz \
  --scene scene_20261114 --profiles nominal low_stand strict_stand \
  --output runs/rviz_comparison
```

每份 NPZ 由第 9 步的 evaluate 导出，参数和场景相同。VST 列来自快照中的原生 target，展示平移通过 TF 完成，不修改保存的地图 XYZ。

## 13. 楼梯专项代码与当前保留状态

`experiments/stair_data.py` 保存上一轮小型楼梯族生成实现。`experiments/stair_experiment.py` 可运行 A/B/C 同预算筛选并续训选优版本，支持 `--initial`、`--data`、`--output`。这些代码保留作实验入口，针对宽深高踏步的新数据增强入口为 `experiments/wide_stair_experiment.py`，配置为 `configs/wide_stair_v2.json`。

旧版生成范围与其实际采样结果需要区分。下一次针对失败楼梯的数据版本应独立命名，明确几何范围、独立 seed、教师条件范围及验证标准，再启动实验。

本次清理后 runs 只保留：

```text
runs/traversability/
├── best.pt
├── model.json
├── config.json
├── validation.json
├── selection_baseline.json
└── README.md
```

最优权重来自 C_geometry_conditions，第 10 轮、step 1710。按原固定验证标准保留，已知大地图右下直梯仍不连通。`model.json` 保存来源、SHA256、验证/测试指标和已知限制。原模型张量保持一致，checkpoint 配置仅修正输出目录与验证基线位置。

大型回归数据现在位于 `datasets/wfc_regression_20261127/`，教师标签、查询点、机器人条件和固定回归端点属于数据，不依赖已删除的运行目录。旧训练日志、测速、PCD 预测、图像以及其它 checkpoint 已清理。
