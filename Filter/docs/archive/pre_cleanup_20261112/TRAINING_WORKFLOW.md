# Filter 局部块训练与测试流程

## 当前全点评分实验：2026-11-12

生成器 Mesh 提供完整静态参考，3 cm 表面采样后保留全部点。关闭孔洞、条带删除和主动制造断连，不执行输入体素降采样。四个训练 seed 为 821100–821103，验证 822100，测试 823100，每 seed 四个固定变体。变体包含干净结构、稀疏悬空点、悬空点加短拖影、悬空点加少量运动残影。静止同类物体均保留。

所有输入点先按完整 Mesh 生成连续标签，再使用与部署相同的 map 网格分块，核心 4×4×3 m、上下文 1 m。每个点有且只有一个核心归属。Tiny 自身的网格编码通过 inverse 恢复到全部监督点，未在输入端删减点数。KNN 统计也使用完整上下文。

```bash
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
$PY -m Filter.data.scoring --config Filter/configs/scoring_mesh_full.json
$PY -m Filter.data.features --config Filter/configs/train_scoring_mesh_full.json
$PY -m Filter.training.train --config Filter/configs/train_scoring_mesh_full.json
$PY -m Filter.training.evaluate --checkpoint runs/filter_scoring_mesh_full_20261112/best.pt --score-only
$PY -m Filter.visualization.results --run runs/filter_scoring_mesh_full_20261112 --score-only
```

使用现有 pure3d-ptv3 环境，NumPy 保持 1.26.4，Torch 保持 2.10.0+cu128。几何依赖为 trimesh、manifold3d、open3d-cpu 0.19.0、opencv-python-headless。terrain-generator 的 NumPy 数组 dataclass 默认值已改为 default_factory，以支持当前 Python 3.11。

冻结编码器缓存原始 486 维特征，任务头从头训练 50 epoch。真实联合 batch=8，AdamW 学习率 1e-3，5% warmup 后 cosine 至 1e-5。SmoothL1 beta=0.1，先在来源组内部按结构语义均衡，再平均当前存在的来源组。来源和语义只用于监督及统计，不进入模型。采样器均衡选择训练场景，不要求不存在的旧来源组。

每轮记录实际更新的平均 loss，并在 eval 模式完整遍历训练、验证集，输出全点 MAE、来源均衡 Macro MAE 及结构 MAE。验证 Macro MAE 最低保存 best.pt，每 100 更新及每轮末保存 last.pt。仅这两个 checkpoint 位于 run 中。测试 seed 仅在训练完成后评估，并与全点打 1 分基线比较。

预测入口默认只导出同点序 XYZ+keep_score，不输出 filtered/removed 子集。该实验的六场景规模用于检查可学习性与损失曲线，不代表广泛跨建筑泛化验证。

真实 ROT stair 全点评分与查看：

```bash
$PY -m Filter.inference.predict --checkpoint runs/filter_scoring_mesh_full_20261112/best.pt \
  --pcd datasets/rot_stairs/raw.pcd --output runs/filter_scoring_mesh_full_20261112/predictions/rot_stair
$PY -m Filter.visualization.results --run runs/filter_scoring_mesh_full_20261112 \
  --prediction runs/filter_scoring_mesh_full_20261112/predictions/rot_stair
```

真实扫描保持全部原始 XYZ 和点序。图像由全部点的深度投影生成，同一视角的原始图和评分图使用相同可见点。主体近景采用投影坐标的 0.1%–99.9% 显示范围，另存 full_extent 全范围图。此显示范围不影响评分点数和导出的 PCD。真实扫描没有逐点 GT，不报告真实数据 MAE。

## 全点标签示例，不启动训练

```bash
/home/cjh/glocal2-nav/.venv/bin/python -m Filter.data.local \
  --example --config Filter/configs/block_ghost_example.json
```

输出 `datasets/filter_block_ghost_example/seed_821010`。完整 Mesh 表面按 3 cm 间距采样，保留全部增强点，不执行 0.1 m 体素降采样。该命令只生成一个完整组件及块内外幽灵，不运行特征缓存、训练或模型预测。

先查看 overview.png，再比较 comparison.png、top.png 和 side.png。蓝色边框标识核心区，灰色边框标识上下文。块外真实地面与栏杆仍可高分，块内漂浮散点与错位片段按到真实表面的误差评分。report.json 汇总点数、分区域标签分布与各阶段耗时。

以下流程描述已有 10 cm 训练数据，示例输出不混入其 manifest。

### 直接从 Mesh 查看完整 GT PCD

```bash
/home/cjh/glocal2-nav/.venv/bin/python -m Filter.data.local \
  --mesh-gt-demo --config Filter/configs/block_ghost_example.json
```

使用已生成的 seed 821010 的 clean.obj，直接采样完整表面，不挖洞、不删条带、不做体素降采样。输出位于该案例的 mesh_gt_demo 子目录：mesh_gt.pcd 为完整几何参考，input.pcd 为完整参考加稀疏残影，input_gt_scores.pcd 为与 input 同点序的连续评分 GT。所有输入点都保留，不导出硬筛除点集。mesh_gt_comparison.png 展示 Mesh、完整 GT PCD、带残影输入与逐点评分。

更新日期：2026-11-11。最终增强输入统一做0.1m体素降采样，每体素保留真实点并同步来源，禁止完整密集点云进入训练编码器。以最新要求为准：**直接生成集中目标的局部组件，不生成整栋大场景再裁块。**

## 1. 数据与划分

六个 seed 是独立的组件随机化组。训练 821000、821001、821002、821003，验证 822000，测试 823000。同一个 seed 的全部案例与变体保持同一 split。

每个 seed 直接生成 8 个局部案例：带栏杆直梯、无栏杆直梯、往复梯、旋转梯、墙角、栏杆、带顶板走廊和平台。楼梯调用原生 terrain-generator 部件，局部组装加入墙、地面、细栏杆和物体。不运行整栋 WFC 求解。几何约 4 m，核心 4×4×3 m，上下文目标 6×6×5 m。额外配准副本可以伸出核心，作为上下文。

每案例四变体：静态、动态、配准、混合。总计 48 个局部案例，192 个标准样本。测试 seed 的 8 个案例各增加稀疏和少次数残影两个固定套件，再增加 64 个测试样本，总计 256 个样本。

输入 PCD 只有 XYZ。Mesh、实例、时间步、语义、source_role、loss_group 均只用于监督与统计。动态来源全部 y=0，包括末位置。其他点 y=exp(-0.5*(到静态表面的距离/0.03)^2)。准确断连结构与静止人物/家具高分。

域随机化包括尺寸、朝向、地图平移、物体尺度与摆放、运动路径、密度、缺口、噪声和配准变换。保留同类物体的静止与运动对照。稀疏测试保留率 0.25–0.45、缺口 0.7–1.2 m。少次数测试动态轨迹 3–7 个位置。标准轨迹 8–20 个位置。

## 2. 准备数据与冻结特征

工作区：`/home/cjh/Pure3D_ws`。沿用已有几何环境和 pure3d-ptv3 训练环境，训练环境补齐 scipy==1.15.3。

```bash
cd /home/cjh/Pure3D_ws
MPLCONFIGDIR=/tmp/pure3d_matplotlib /home/cjh/glocal2-nav/.venv/bin/python \
  -m Filter.data.generate --config Filter/configs/dataset.json
HF_HUB_OFFLINE=1 /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Filter.data.features --config Filter/configs/train_local.json
```

数据在 datasets/filter_local_10cm_v3，缓存位于 datasets/cache/filter_local_10cm_v3。旧 v2 示例保留。中止的大场景数据不属于训练清单。

每个案例有 overview，带栏杆直梯案例另保存四变体 GT 对比与来源近景。监督使用场景共享 NPY 数组，盒仅保存索引。上下文完整参与一次 Utonia Tiny 编码，缓存核心所需的 486 维特征、inverse、相对坐标和六维 XYZ 邻域统计。投影与评分头不缓存，几何归一化统计只用训练数据。

冻结权重来自 Pointcept/Utonia 的 utonia_tiny.pth，SHA256 e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137。checkpoint 包含冻结编码器，独立推理不依赖再次下载。

## 3. 网络与损失

Utonia Tiny 全程冻结/eval。任务头为特征投影 486→128，几何分支 9→32→32，拼接后 160→128→64→1 与 Sigmoid。

五组互斥来源：普通建筑、静止物体、准确断连结构、配准重影、动态残影。SmoothL1 beta=0.1，先对各组核心点平均，再对当前组平均。训练不做分数阈值筛点。

真实 batch=8，组成 2 普通、2 动态、2 配准、1 静止物体、1 断连结构盒。按场景均衡选取。每 epoch 采样数为训练样本数向上取整到 8 的倍数，允许重采样。验证与测试全部遍历核心点。

## 4. 训练、保存与恢复

```bash
HF_HUB_OFFLINE=1 /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Filter.training.train --config Filter/configs/train_local.json
HF_HUB_OFFLINE=1 /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  -m Filter.training.train --resume runs/filter_local_10cm_v3/last.pt
```

50 epoch，seed42，AdamW LR1e-3、weight decay0.01，5% warmup 后 cosine 到1e-5，Dropout0.1，BF16，float32 loss。workers4、锁页与预取。

前 5 轮逐轮验证，其后每 5 轮验证。按五组 Macro MAE 最低选 best，同分比较固定0.5预测阈值的保留F1。每100次完整更新及每轮结束保存last。Ctrl+C在完成当前参数更新后保存。恢复包含优化器、调度器、随机状态和批次游标。可用 --stop-after-epoch 5 演示保存/恢复，仍保留50轮调度。

输出 runs/filter_local_10cm_v3 下仅 best.pt、last.pt 两个训练checkpoint，以及 config.json、history.json、train.log、progress.json、evaluations、plots 和 predictions。

## 5. 最终测试与可视化

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Filter.training.evaluate \
  --checkpoint runs/filter_local_10cm_v3/best.pt
MPLCONFIGDIR=/tmp/pure3d_matplotlib /home/cjh/glocal2-nav/.venv/bin/python \
  -m Filter.visualization.results --run runs/filter_local_10cm_v3
```

GT保留判定固定 y>=0.5。只在验证集比较预测阈值0.3/0.4/0.5/0.6/0.7，按五组平均保留F1选取，同分比较静态误删率、再比较与0.5的距离。GT不会随预测阈值改变。

best与阈值固定后，测试标准、稀疏、少次数残影三套样本，输出全点/各组MAE、RMSE、Macro MAE，动态筛除率、配准点筛除与误删、静态结构保留率，并比较全点保留基线。这里的全点是全部有监督核心点，上下文不重复计分。

同点序图显示GT、预测、绝对误差和筛除结果。测试只覆盖一个独立seed随机化组的8种局部案例，不代表未见完整建筑泛化。

## 6. 单张完整 PCD 独立推理

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Filter.inference.predict \
  --checkpoint runs/filter_local_10cm_v3/best.pt --pcd datasets/rot_stairs/raw.pcd \
  --threshold 0.5 --output runs/filter_local_10cm_v3/predictions/rot_stair
```

threshold取evaluations/threshold.json中选定值。输入只需XYZ，不读取GT、来源或Mesh。完整PCD内部做空间块，按核心索引恢复所有原始点序。独立推理同样先按0.1m降采样，计算体素代表点分数后映射回全部原始点。输出 prediction.pcd（XYZ+keep_score）、filtered.pcd、removed.pcd 和阶段耗时。ROT stair没有完整GT，只做迁移查看。

本机数据查看：source /opt/ros/humble/setup.bash 后执行 python -m Filter.visualization.rviz --scene datasets/filter_local_10cm_v3/seed_821000_straight_railed --full-scene --variant 3。预测文件可用 --prediction 指定。服务器全部生成、训练、评估和绘图可headless运行。
