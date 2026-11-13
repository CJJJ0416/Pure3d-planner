# Filter 全点评分操作流程

当前配置为 `Filter/configs/scoring_mesh_full.json` 和 `Filter/configs/train_scoring_mesh_full.json`。运行前进入 `/home/cjh/Pure3D_ws`。本轮已训练完成，结果见 [训练档案](TRAINING_ARCHIVE_20261112.md)。复现为新实验时，复制配置并修改 data、cache、output，保留已归档权重。

## GPU 体素统计版本

已完成50 epoch及独立测试，best为epoch15，last为epoch50。结果见 [GPU体素训练档案](TRAINING_ARCHIVE_20261113_VOXEL.md)。速度目标达到，精度恢复目标未达到。

配置 `Filter/configs/train_scoring_voxel.json`，实验名 `filter_scoring_voxel_20261113`。六维统计按 0.05、0.15、0.5 m 交错排列，每个尺度为所在体素的 `log1p(count)` 和周围26个非空体素的比例。计算使用地图原点下未缩放、未居中的完整 XYZ，保留重复观测。每张 PCD 仅计算一次，再按原索引供各窗口使用。

```bash
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
RUN=runs/filter_scoring_voxel_20261113
export MPLCONFIGDIR=/tmp/pure3d_filter_matplotlib
$PY -m Filter.data.scoring --config Filter/configs/scoring_voxel.json
$PY -m Filter.data.voxel_statistics --config Filter/configs/train_scoring_voxel.json
$PY -m Filter.training.train --config Filter/configs/train_scoring_voxel.json
$PY -m Filter.training.evaluate --checkpoint "$RUN/best.pt" --score-only
$PY -m Filter.visualization.results --run "$RUN"
```

本轮按原 seed 重建六场景，逐点核对标签、结构类别、来源组和分块索引后，复用原冻结特征。新缓存只保存 `geometry.npy` 及自己的 encoding.json。`feature_cache` 指向原缓存，读取486维特征、inverse、relative、标签、原始索引和编码器权重。新统计的均值、标准差只来自 Train，并随 checkpoint 保存。

`geometry_mode` 为 knn、voxel 或 relative。历史断点未设置该字段时按原 use_knn 恢复。voxel 与 knn 分支都使用三维相对坐标加六维统计，但特征含义不同，需要分别训练评分头。Tiny 继续冻结/eval。

全点推理沿用下文命令，指定新的 best。XYZ 上传后由统计和编码共享，评分前不将特征返回 CPU。geometry_s 是统计墙钟耗时，geometry_cuda_s、encode_s、head_s 使用 CUDA Event，encode_wall_s/head_wall_s 记录主机提交区间。不同区间可能重叠，不能直接相加代替 total_s。

## 原六维 KNN 的 GPU 后端

保留原含KNN的 best.pt，不重新训练。GPU保持同一个上下文、相同原始点/代表点选择、64近邻、两个半径、自身排除和归一化参数。索引和距离运算使用float64，输出仍为原来的六维float32。

```bash
$PY -m pip install --no-deps -r Filter/requirements-gpu-knn.txt
export CUPY_CACHE_DIR=/tmp/pure3d_cupy_cache
$PY -m Filter.inference.predict \
  --checkpoint runs/filter_scoring_mesh_full_20261112/best.pt \
  --pcd datasets/rot_stairs/raw.pcd \
  --output runs/filter_knn_cuda_20261113/predictions/rot_stair \
  --knn-backend cuda
```

`--knn-backend` 默认cuda，cpu为显式参考入口。relative/voxel模式不使用该参数，也不导入CuPy。缓存生成同样使用GPU KNN，配置中可显式设置knn_backend=cpu生成CPU参考。不要用已有归档的缓存路径重建新实验。

后端依赖CuPy 13.6和fastrlock，保留现有NumPy/PyTorch版本。KDTree构建和精确近邻内核来自CuPy官方v14.0.1源码，许可证及CuPy 13兼容改动记录在third_party/cupy_kdtree。半径计数使用单独的CUDA子树边界和批量计数内核，仅保存每个查询的两个计数，不调用上游会分配查询数×点数矩阵的半径API。没有CPU自动回退。

首次运行会编译并缓存CUDA内核。geometry_cuda_s使用CUDA Event，geometry_s是主机提交区间，encode_wall_s可能包含等待前序GPU几何计算的时间。cuda_peak_GiB是Torch分配峰值，另记录cupy_pool_reserved_GiB，二者不是整机GPU占用。新的验证记录位于runs/filter_knn_cuda_20261113。

## 无 KNN 对照版本

已完成 50 epoch，结果见 [无 KNN 训练档案](TRAINING_ARCHIVE_20261113_NO_KNN.md)。best 为 epoch 34，last 为 epoch 50。

配置 `Filter/configs/train_scoring_no_knn.json` 使用相同数据与冻结缓存，从零初始化投影、三维相对坐标分支和评分头。`use_knn=false` 时不读取 `geometry.npy`，不应用六维统计归一化，完整 PCD 推理也不建立 KDTree 或查询邻域。旧 checkpoint 未设置该字段时仍使用原来的九维分支。

```bash
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
RUN=runs/filter_scoring_no_knn_20261113
$PY -m Filter.training.train --config Filter/configs/train_scoring_no_knn.json
$PY -m Filter.training.evaluate --checkpoint "$RUN/best.pt" --score-only
$PY -m Filter.visualization.results --run "$RUN" \
  --xyz-reference runs/filter_scoring_mesh_full_20261112/predictions
```

这一轮只读复用原 333 盒缓存，不运行数据生成或缓存重建。训练保持 seed 42、50 epoch、batch 8、相同来源/结构均衡损失和学习率调度。恢复使用 `$PY -m Filter.training.train --resume "$RUN/last.pt"`。全点推理、显式阈值后处理和 RViz 使用下文相同入口，指定新的 checkpoint 和输出目录。

本机原始合成场景目录目前缺失。评估导出直接从缓存汇总核心点数。上述绘图参数显式指定旧 run 的同点序 XYZ，连续标签从相同冻结缓存恢复，不重新生成监督数据。

## 1. 数据、冻结缓存、训练

```bash
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
export MPLCONFIGDIR=/tmp/pure3d_filter_matplotlib
$PY -m Filter.data.scoring --config Filter/configs/scoring_mesh_full.json
$PY -m Filter.data.features --config Filter/configs/train_scoring_mesh_full.json
$PY -m Filter.training.train --config Filter/configs/train_scoring_mesh_full.json
```

使用现有环境，不切换到旧 WFC 缓存环境。六个局部场景完整表面加稀疏残影，无输入体素降采样。训练/验证/测试按 seed 分开，数据生成后固定，不在 epoch 内重新生成。

Tiny 冻结/eval，单次编码一个完整上下文，缓存原始 486 维特征。训练只更新投影、几何分支和评分头，真实 batch=8。每个核心点只计算一次损失。

## 2. 恢复与测试

```bash
RUN=runs/filter_scoring_mesh_full_20261112
$PY -m Filter.training.train --resume "$RUN/last.pt"
$PY -m Filter.training.evaluate --checkpoint "$RUN/best.pt" --score-only
$PY -m Filter.visualization.results --run "$RUN"
```

训练 50 epoch、1400 更新，按每轮验证 Macro MAE 选 best，次级比较全点 MAE。每 100 更新及每轮末保存 last。Ctrl+C 在一次完整更新后保存并退出。断点包括模型、冻结编码器、优化器、调度器、随机状态、epoch 和批次位置。恢复训练沿用断点中的配置和调度周期。

评估同时输出全点和来源均衡 MAE、RMSE、结构误差及全点打 1 分基线。本轮默认是连续评分，不自动选择预测阈值。可选二值诊断仍由 evaluate 的阈值评估模式显式运行。

## 3. 全点推理

```bash
$PY -m Filter.inference.predict --checkpoint "$RUN/best.pt" \
  --pcd datasets/rot_stairs/raw.pcd --output "$RUN/predictions/rot_stair"
$PY -m Filter.visualization.results --run "$RUN" \
  --prediction "$RUN/predictions/rot_stair"
```

只读 XYZ，输出同坐标、同点序、同点数的 `prediction.pcd`、`prediction.npz` 和耗时 JSON。推理入口只打分，不带阈值参数，不自动生成 filtered/removed。

## 4. 显式阈值选择与代价预测

```bash
$PY -m Filter.inference.threshold --prediction "$RUN/predictions/rot_stair/prediction.pcd" \
  --threshold 0.2 --output "$RUN/predictions/rot_stair_tau020"
$PY -m Traversability.inference.predict --checkpoint runs/traversability/best.pt \
  --pcd "$RUN/predictions/rot_stair_tau020/filtered.pcd" --condition 0.5 0.4 0.3 0.1 \
  --output "$RUN/predictions/rot_stair_tau020/cost"
```

选择模块不重新推理。输出 filtered.pcd、removed.pcd、filter_mapping.npz 和 selection.json。0.2 为用户指定的操作阈值，评分模型和连续 GT 均不改变。

本轮实际 Filter→cost 对照位于 `runs/filter_scoring_mesh_full_20261112/predictions/rot_stair_traversability_tau020/`。该目录保留过滤前后的同权重 cost 预测与同视角图片。Traversability 条件为 clearance_min、slope_max、step_max、stand_ratio。

## 5. RViz

使用系统 Python 和 ROS2 Humble，不使用 Conda Python 导入 rclpy：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m Filter.visualization.rviz \
  --prediction "$RUN/predictions/rot_stair/prediction.pcd"
```

仅查看评分不要求已有筛除文件。需要切换筛选结果时，增加 `--selection "$RUN/predictions/rot_stair_tau020"`。默认只启用评分点云，保留与删除点可在左侧独立切换。

查看真实数据盒：

```bash
/usr/bin/python3 -m Filter.visualization.rviz \
  --scene datasets/filter_scoring_mesh_full_20261112/seed_821100 --seed 42 --with-trails
```

盒读取共享数组中的对应 offsets 区间，上下文与 core_mask 同步切片。全场景可加 `--full-scene --variant 2`。

图片主体视图使用 0.1%–99.9% 投影坐标显示范围，并另存 full_extent 图。该显示裁切不改变输出点云。真实 ROT stair 没有逐点 GT，不报告真实数据 MAE。

## 历史入口

组件构造和增强函数继续复用 `data/local.py`、`data/augment.py`、`data/prepare.py`。旧 10 cm 配置 `configs/train_local.json` 仅作历史复现，不是 features/train 的默认配置。原缺口、少次数残影和 WFC 流程说明已移到 [整理前快照](archive/pre_cleanup_20261112/README.md)。
# Incremental scan dataset pilot

The new `python -m common.scanning.pipeline` entry generates rolling LiDAR and
consistent IMU, replays an unmodified external Faster-LIO, exports dense
registered increments, builds static-Mesh retention targets and core/context
indices, and renders GT comparisons. See [scanning workflow](../../common/scanning/README.md).
The `--frames 200 --profiles coverage` pilot checks a 20-second prefix of the
configured trajectory. Current defaults use 1.5x movement speed and no initial
or final idle, with a 33.33-second continuous route. The frame limit itself
does not rescale motion. Historical actual configs retain their own timing.
No Filter or RECON training is started by this entry. Existing datasets and
the selected relative-only Filter baseline remain separate.
