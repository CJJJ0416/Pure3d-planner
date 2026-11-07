# 当前帧 LiDAR 的 PTv3 / Tomography 训练

当前帧原始 LiDAR XYZ 和重力 up → 官方完整 PTv3 → 64 维空间特征 → 地图头 → cost、重力相对地面高度。骨干与地图头联合训练，当前不训练 SRU 或导航后端。

## 当前监督规则

从场景 `vst.pickle` 读取各原生层的 cost 与 ground，通过完整 LiDAR 位姿转换到局部网格。可用标签全部参与 Loss：

```python
mask_cost = target_available.copy()
mask_height = target_available.copy()
```

没有高度连续、当前回波高度匹配或邻域覆盖条件。当前帧未命中但已有 VST 标签的位置也参与监督。没有原生标签的位置留空，同一高度带存在不同地面高度时仍沿用现有网格表示规则。输入点数预算不改变监督范围。

输出 cost、高度各为 `[B,6,64,64]`。cost 以 60 归一化，两个目标使用按帧归一化的 SmoothL1。teacher、mask 与世界位姿只用于标签和 Loss，不作为模型输入。

## 维护入口

| 工作 | 入口 |
| --- | --- |
| Headless Warp 采集 | `scripts/data/collect_mid360_warp.py` |
| 原生 VST 配对 | `scripts/data/pair_mid360_lidar_reference.py` |
| 真实 batch 训练、恢复 | `scripts/training/run_pretraining.py` |
| 本地 / H100 配方 | `configs/perception_pretrain_rtx4060_batch.json` / `perception_pretrain_h100_batch.json` |
| 训练数组缓存 | `src/ptv3_tomography/data/training_cache.py` |
| portable 数据与登记 | `scripts/data/make_dataset_portable.py` / `register_perception_data.py` |
| 单帧 / 全序列地图推理 | `scripts/perception/predict_map.py` / `export_plan_a.py` |
| 全点特征恢复 | `scripts/perception/export_point_restoration.py` |
| CPU 监督可视化 | `scripts/visualization/audit_point_supervision.py` |
| Isaac 原生交互可视化 | `scripts/visualization/view_plan_a_isaac.py --mode 1` |

训练只保留 `runs/<实验名>/best.pt`、`last.pt`，日志位于 `logs/`，缓存默认位于 `cache/training_cache_teacher_v1/`。旧缓存的观测 mask 不再读取，当前 mask 由原始 cost、高度的标签可用性生成。后续修改根据实际运行中出现的问题处理，不增加预检查、哈希拒绝或重复回归入口。

项目由 glocal2-nav/master 管理。官方核心位于 `third_party/PointTransformerV3`，本次未修改。旧源码副本和历史运行结果留在本地忽略目录，不进入 Git。

服务器在现有 `rl_env_glocal2` 中运行纯 Warp 采集、PyTorch 训练和 matplotlib Agg 绘图。本次重建数据并从随机参数开始，完整步骤见 [SERVER_TRAINING.md](docs/SERVER_TRAINING.md)。

详细说明：[服务器训练](docs/SERVER_TRAINING.md)、[本次代码审查](docs/TEACHER_SUPERVISION_CLEANUP.md)、[目录](docs/PROJECT_LAYOUT.md)、[官方训练对照](docs/OFFICIAL_TRAINING_COMPARISON.md)。
