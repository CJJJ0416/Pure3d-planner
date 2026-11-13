# Filter：单张 XYZ PCD 的残影评分

当前参考为无六维统计的 `filter_scoring_no_knn_20261113`，使用 relative 分支。权重身份与来源见 [baseline.json](configs/baseline.json)。历史训练数据为 `filter_scoring_mesh_full_20261112`。完整 Mesh 表面点加稀疏悬空点、短拖影和少量运动残影。输入不做 0.1 m 降采样，不制造扫描缺口，缺失结构交给后续 Reconstruction。

原KNN best24现支持精确GPU后端，无需重训。独立测试六维特征和评分与CPU缓存逐位一致，ROT stair与building_SUST共1438万点的完整评分也逐位一致。全流程耗时为86.18/111.41 s，相比历史CPU记录快3.29/3.40倍。入口默认 `--knn-backend cuda`，依赖见 `requirements-gpu-knn.txt`，使用命令见操作流程。实测产物为 `runs/filter_knn_cuda_20261113/`。

当前无 KNN 参考使用 `configs/train_scoring_no_knn.json` 和 `runs/filter_scoring_no_knn_20261113/`。相同数据与冻结特征缓存，评分头从零训练。三维相对坐标分支保留，六维邻域统计不读取、不计算。运行命令见操作流程。

该对照已完成 50 epoch。独立测试 Macro MAE 为 0.033010，原含 KNN 版本为 0.020233。两张真实地图全点推理约提速 9.9–13.7 倍，时间基线取原同机记录。详见 [无 KNN 训练档案](docs/TRAINING_ARCHIVE_20261113_NO_KNN.md)。

GPU体素统计版本也已完成50 epoch，配置为 `configs/train_scoring_voxel.json`。三个尺度的点数/占用率由PyTorch CUDA一次计算，统计本身在588万/849万点上耗时0.178/0.220 s。测试Macro MAE为0.036575，精度恢复目标未达到。详见 [GPU体素训练档案](docs/TRAINING_ARCHIVE_20261113_VOXEL.md)。

```text
地图系 XYZ → 4×4×3 m 核心 + 1 m 上下文
  → 冻结官方 Utonia Tiny → 多尺度体素特征 486 维
  → 特征投影 + inverse 恢复 + XYZ 几何分支
  → FilterHead → 全部原始点的 keep_score
```

默认只打分，保存全部 XYZ 和原始点序。`keep_score` 是学习的保留系数。需要筛选时，单独调用 `Filter.inference.threshold`，阈值由调用者指定。

- [训练档案](docs/TRAINING_ARCHIVE_20261112.md)：本轮配置、指标、权重身份、ROT stair 与规划对照。
- [操作流程](docs/TRAINING_WORKFLOW.md)：生成、缓存、训练、恢复、评估、推理和 RViz。
- [数据契约](docs/DATA_CONTRACT.md)：输入、连续标签、分块索引和输出字段。
- [模型结构](docs/ARCHITECTURE.md)：冻结编码器、评分头、损失与接口。

| 职责 | 当前入口 |
|---|---|
| 完整组件和稀疏残影数据 | `Filter.data.scoring` |
| Mesh 连续标签与索引分块 | `Filter.data.targets`、`Filter.data.boxes` |
| 冻结特征缓存 | `Filter.data.features` |
| GPU体素统计及独立缓存 | `Filter.data.voxel_statistics` |
| 场景均衡采样与评分头 | `Filter.data.dataset`、`Filter.models.filter` |
| 训练、恢复与测试 | `Filter.training.train`、`Filter.training.evaluate` |
| 全点打分、显式阈值选择 | `Filter.inference.predict`、`Filter.inference.threshold` |
| 曲线、逐点对照与 RViz | `Filter.visualization.results`、`Filter.visualization.rviz` |

当前训练配置为 `configs/train_scoring_no_knn.json`，不含六维统计。`configs/scoring_mesh_full.json` 是历史完整 Mesh 合成数据的复现配置，`train_scoring_mesh_full.json` 保留原 KNN 对照。数据、缓存和结果分别位于 `datasets/filter_scoring_mesh_full_20261112`、`datasets/cache/filter_scoring_mesh_full_20261112`、`runs/filter_scoring_mesh_full_20261112`。权重只保留本轮的 `best.pt` 与 `last.pt` 两个入口。

`data/local.py`、`augment.py`、`prepare.py` 与旧配置仍提供组件、增强和历史数据复现，不是当前默认训练流程。整理前文档见 [历史快照](docs/archive/pre_cleanup_20261112/README.md)。

新的增量扫描监督由 [common/scanning](../common/scanning/README.md) 组织，包含带噪 IMU、MID-360 盲区和真实 Faster-LIO 算法配准。当前仅完成数据与标签闭环，尚未用它替换训练数据或产生新权重。
