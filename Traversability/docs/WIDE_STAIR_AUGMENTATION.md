# 基于当前 best.pt 的宽深高楼梯增强训练

2026-11-10。已按用户指令启动，实时进度读取 `runs/cost_wide_stair_v2/execution.json`。

## 初始化与目标

初始化权重为 `runs/traversability/best.pt`，即上一轮 C_geometry_conditions 的第10轮。新实验加载模型参数，重新建立优化器与30轮学习率周期，新轮数从1开始。

目标失败设施为大地图 seed 20261127 的 F1_F2_straight_1，真实踏步高度0.3839m、深度0.6174m、宽度1.4418m。GT固定端点最低连通代价约0.1667，基线预测约0.4727，阈值0.3下基线断开。

不把该大地图或其点云、标签加入训练。新训练包含一个相同尺寸锚点，使用新的seed、朝向、井口、背景楼层和教师标签，其余训练尺寸独立分层采样。

## 几何增强

新增36个独立几何实例，train/val/test为24/6/6。每个seed独立选择踏步高度、深度和宽度，独立几何实例不再使用同族固定高度和宽度的三深度变体。

| 维度 | 聚焦尺寸范围 | 宽范围 |
|---|---|---|
| 踏步高度 | 0.34–0.425m | 0.16–0.44m |
| 踏步深度 | 0.50–0.75m | 0.18–0.85m |
| 楼梯宽度 | 1.20–1.70m | 0.55–2.00m |

每个划分约一半聚焦样本、一半宽范围样本。额外随机化井口侧向间隙1.0–2.1m、栏杆、墙、楼层高度、Z轴朝向、XY亚体素相位和两层/三层局部上下文。

使用原生 terrain-generator 楼梯部件，通过实际几何实现指定踏步深度。3cm表面采样、5cm输入体素，每个seed保存Mesh、完整PCD、降采样PCD和overview。

三层样本增加下方背景地板，目标直梯连接中层与上层，用于局部几何上下文学习，不生成采集路线或运行机器人。

## 旧场景回放与数据恢复

开始时，旧版 `wfc_stair_stand_v1`、`wfc_response_paired_v4` 和 `wfc_capability_dr_stand_v3` 缓存不在现存数据目录中。

本次从保留的 `datasets/wfc_capability_dr_v2` 的四个训练建筑和一个验证建筑重新生成四参数监督，保留原输入PCD，重新建立原生Tomogram和几何缓存。训练旧场景为seed20261109–20261112。seed20261113只用于验证保留能力。

大图回归标签继续使用 `datasets/wfc_regression_20261127`，几何引用已修正为现存的 `datasets/wfc_unseen_20261127`。其查询XYZ与保留GT PCD逐项一致。

## 教师参数增强

每个新几何都重新运行原生VST，沿用完整教师参数、cost_barrier=50和距离曲线p=2。全部原生Tomogram查询保留，不增加连续性、回波或邻域覆盖筛选。

学生顺序为 `[clearance_min,slope_max,step_max,stand_ratio]`，教师interval_free固定0.5m。

| 参数 | 范围 |
|---|---|
| clearance_min | 0.2–1.0m |
| slope_max | 0.18–0.70rad |
| step_max | 0.08–0.55m |
| stand_ratio | 0.02–0.50 |

每次前向使用40组条件。训练几何有两个条件包，合计76组真实教师结果，按epoch轮换。两个锚点覆盖宽范围，两个锚点聚焦目标能力附近。临界对照从目标楼梯几何中的原生梯度、最大高度差和站立格子数选择，标签由完整教师计算。

固定保留nominal、low_stand、strict_stand与relaxed条件，relaxed为 `[0.5,0.6,0.4,0.05]`。验证/测试另有固定stand扫描。临界区域只用于选择参数，不用于删去监督点。

## 训练设置

网络保持Utonia Tiny + QueryPool + 单层FiLM + CostHead。原SmoothL1与条件差值损失保持，lambda=1，响应权重保持 `[1,2,1,2]`。

- 新训练30轮，每轮342块、171次更新。
- 真正batch=2，workers=2，BF16。
- 每轮240块来自新增楼梯，102块来自旧建筑回放。
- 前2轮冻结编码器，随后联合微调。
- Head学习率5e-5，编码器5e-6，30轮余弦周期。
- 每5轮固定验证。只保存best.pt与last.pt。

训练标签与网络结构不变。新数据包含更宽的几何/条件范围，采用针对新旧数据的采样比例及较低微调学习率。

## 选优与评估

固定验证包含6个新直梯和1个旧验证建筑。先以初始化模型在同一验证集建立基线，再按既定规则选优：错误放行率不超过基线2个百分点、非楼梯误差不超过基线5%，满足后优先降低断裂率，再比较stand条件差值MAE和整体MAE。

30轮结束后：

1. 使用验证选出的best在6个独立新seed上测试。
2. 比较旧验证建筑的前后指标。
3. 在大地图原固定楼梯端点与同一XYZ上比较GT、Before、After。
4. 在一个新测试PCD和大型terrain完整PCD上输出所有原始点预测。
5. 生成summary.json、REPORT.md、训练曲线、GT对比图和局部通道图。

cost阈值0.3和0.45m邻接保持。基线 `runs/traversability/best.pt` 保留，增强实验单独输出。

## 文件与入口

- 配置：`Traversability/configs/wide_stair_v2.json`
- 数据：`datasets/wfc_wide_stair_v2/`
- 输出：`runs/cost_wide_stair_v2/`
- 几何：`common/geneator/stair_families.py`
- 数据准备：`Traversability/experiments/wide_stair_data.py`
- 流程调度：`Traversability/experiments/wide_stair_experiment.py`
- 报告：`Traversability/experiments/wide_stair_report.py`

从工程根目录运行整个流程：

```bash
PYTHONPATH=/home/cjh/Pure3D_ws HF_HUB_OFFLINE=1 \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
-m Traversability.experiments.wide_stair_experiment
```

已启动时读取execution.json和日志，不重复执行命令。数据准备完成记录位于data_ready.json，训练开始后epoch与step读取history.json。恢复流程使用现存last.pt。
