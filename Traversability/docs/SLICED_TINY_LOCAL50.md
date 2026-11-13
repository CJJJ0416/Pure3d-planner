# 共享切片坐标的 Tiny 本机50轮训练

日期：2026-11-13。运行目录：`runs/traversability/cost_sliced_tiny_local50_20261113/`。

本次训练只在本机执行。使用现存6栋建筑，沿原划分4栋训练、1栋验证、1栋测试，训练50轮。原先的服务器144栋方案由本次本机方案替代。

## 1. 网络与输入输出

```text
完整建筑 PCD
    ↓ 原生 CUDA 几何分层
ground / ceiling（教师几何中间量）
    ↓ 原生高度合并，确定共享 ground XYZ
切片支撑点云 XYZ
    ↓ 4×4×3m 核心块，1m 上下文
官方冻结 Utonia Tiny，每窗口一次
    ↓ 486维体素特征
Adapter：486 → 256，LayerNorm，GELU
    ↓ FiLM：四参数 → 64 → gamma/beta各256维
CostHead：256 → 128 → 64 → 1，Sigmoid
    ↓ inverse 恢复标量代价与原点序
切片 ground XYZ + trav_cost
```

Tiny全程`eval()`，权重与缓冲量固定。预处理沿用官方Tiny契约，`scale=0.5`、`grid_size=0.01`，物理体素2cm。Tiny使用BF16，Adapter、FiLM、CostHead与损失使用FP32。颜色与法向使用零占位。

第一版学生几何输入仅为切片ground XYZ，ceiling只参与原生教师计算。报告设施、背景和条件响应误差，以评价这种输入表示的拟合能力。

每窗口编码一次并缓存486维体素特征和inverse。所有条件共享该缓存。学生没有局部/全局QueryPool、KNN表、全楼第二次编码或外部查询XYZ。输出保留全部切片支撑点，包含教师高代价点。

## 2. 共用一次几何生成监督

来源为`datasets/wfc_capability_dr_v2`每个场景的完整`terrain.pcd`，教师和学生都使用该份PCD，不混用旧5cm输入生成的标签。

几何层使用`common/vst_global_planner/tomography/kernels.py`的原生CUDA核，分辨率0.1m、切片间隔0.5m。ground按原生float16导出精度生成XYZ，高度差小于0.5m的相邻ground按当前原生Tomogram规则合并。几何合并不依赖机器人参数或cost阈值。每场景保留layer/XY来源索引。

教师在同一次几何上计算不同参数下的原生VST cost。每条件沿同一高度合并关系传播最小cost，标签为`cost/50`。监督XYZ与学生输入XYZ逐项相同、点序相同。没有坐标近邻匹配、支撑可靠性筛选或新增监督掩码。

教师固定`interval_free=0.5m`、`p=2`、`cost_barrier=50`，其他原生参数来自`teacher_building.json`。

学生条件顺序与范围：

| 条件 | 范围 |
|---|---:|
| clearance_min | 0.2–1.0m |
| slope_max | 0.18–0.70rad |
| step_max | 0.08–0.55m |
| stand_ratio | 0.02–0.50 |

复用两套40槽条件库，每套含4个固定条件和四参数各9点扫描，两套共76个条件列。本批每栋有75个唯一参数组合，基准组合在扫描中重复一次。第一套以`[0.5,0.6,0.4,0.05]`为背景，第二套使用独立随机背景。step、slope与stand扫描覆盖教师实际局部门槛，训练、验证使用不同临界偏移。每个参数的9点扫描产生36对单参数监督。

所有场景已有WFC布局、楼层、楼梯、斜坡、栏杆和几何尺度随机化。本次重算标签、没有新建场景或混用建筑划分。旋转在原场景生成时完成，训练不再在线旋转。

每场景保存coord、layer_cell、cost、conditions、facility和纯整数分块索引。facility仅用于加权和统计，不删除监督点。原始PCD、历史权重与实验记录保留。

## 3. 优化与选优

只训练Adapter、FiLM、CostHead，共200,001个参数。新任务头随机初始化，旧查询模型作为历史参考。

\[
L=L_{absolute}+2L_{condition}
\]

两项都用`SmoothL1(beta=0.1)`。绝对损失对设施与背景等权。条件差值使用每参数36对，变化阈值为0.02，变化区与稳定区等权，四参数等权。全部原生有限支撑点参与。缺少某类点的单块按实际存在的区域归约。

- AdamW，学习率`1e-4`，weight decay `0.05`。
- 总周期50轮，5%更新warmup，随后余弦下降至`1e-6`。
- 真实batch为2个块。每轮完整遍历训练块，打乱顺序，条件库逐轮交替。奇数块数时重复一个块补齐最后一批。
- 每轮完整评估固定验证建筑的两套条件库。
- 第1轮结束保存断点，再从`last.pt`真实恢复至第50轮。恢复优化器、调度器和步数，50轮总周期不变。
- 每轮保存`last.pt`。初始化和各轮按相同验证分数选`best.pt`。

对每个参数，响应误差为变化区差值MAE与稳定区差值MAE的等权平均。选优分数为归一化Cost MAE加四参数平均响应误差。测试建筑不参与选优。

记录总loss、绝对loss、条件loss、四参数各自变化/稳定loss、验证原始Cost MAE、响应方向和幅度、设施/背景误差、耗时与显存。`progress.json`提供当前阶段、文字进度条、轮次和分项指标。

## 4. 执行入口

首次运行：

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 \
  /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Traversability.experiments.sliced_local \
  --config Traversability/configs/sliced_tiny_local50.json
```

已有本次`last.pt`时增加`--resume`。几何和GPU教师由配置中的`glocal2-nav/.venv`执行，缓存和训练使用本机`pure3d-ptv3`。

新PCD独立推理：

```bash
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 \
  /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Traversability.inference.sliced \
  --checkpoint runs/traversability/cost_sliced_tiny_local50_20261113/best.pt \
  --pcd /absolute/path/to/building.pcd \
  --condition 0.5 0.6 0.4 0.05 \
  --output runs/traversability/sliced_prediction
```

独立推理只读取PCD与模型。先运行几何分层，再编码和预测，不计算VST cost，也不读取训练查询XYZ。输出`prediction.pcd`包含所有切片支撑点，保持地图坐标。

## 5. 结果与核验

- 核验76列监督与共享XYZ一致。
- 核验Tiny权重与缓冲量逐项不变，并核验实际梯度和任务头更新。
- 保存真实50轮曲线、所选轮次、测试集完整指标及GT/预测/误差图。
- 独立从测试建筑完整PCD再预测一次，核对其XYZ及预测与缓存评估路径一致。
- checkpoint保存完整官方编码器、任务头、预处理契约与机器人参数顺序。新架构由独立入口加载，旧模型入口保留。

FiLM的`lambda=0`与参数拼接对照由配置项支持，本次只运行完整FiLM版本的50轮，不启动额外训练。规划器、Filter与Recon保持现状。
