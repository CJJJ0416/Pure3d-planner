# 当前帧点云感知训练

当前入口只使用真实多帧 batch：当前帧原始 XYZ、重力 up → PTv3 → 地图头 → cost 与高度。每个样本仍是一帧，各帧通过官方 collate/offset 打包后共同前向与反向。

## 标签

读取 VST 原生分层数组，`data[0]` 是 cost，`data[3]` 是 ground。完整位姿映射到 LiDAR XY 与重力高度带后，`mask_cost = mask_height = target_available`。不再要求高度连续、地表回波或周围射线覆盖。

无标签的位置不参与 Loss。cost 以 60 归一化，高度为 `ground_world_z - lidar_world_z`，单位米。两个分支使用 SmoothL1。当前关闭支撑 loss。

## 启动

从仓库根目录执行：

```bash
python src/ptv3_tomography/scripts/training/run_pretraining.py \
  --registry <实际 registry.json> \
  --config src/ptv3_tomography/configs/perception_pretrain_rtx4060_batch.json \
  --output runs/local_run
```

H100 使用 `perception_pretrain_h100_batch.json`。默认训练缓存名为 `training_cache_teacher_v1`。缓存保存原始 cost、高度，加载时直接得到相同的标签可用性 mask，旧缓存中保存的稀疏观测 mask 不再参与训练。

本次服务器实验重建数据，模型与优化器从零开始。同一次新实验中断后，`--resume runs/local_run/last.pt` 恢复本次模型、AdamW、OneCycle、RNG 与 epoch/样本游标。服务器统一激活 `rl_env_glocal2`，完整从零启动命令见 [SERVER_TRAINING.md](SERVER_TRAINING.md)。

输出目录顶层保留 `last.pt` 与按 dev 结果选择的 `best.pt`。没有 dev 时只保存 last。SIGINT/SIGTERM 在 batch 边界保存，强制终止使用上一次已写入的断点。

同建筑与未见建筑的划分由 registry 显式指定。当前不再在运行时重复扫描文件哈希或自动拒绝配置。服务器方案见 [SERVER_TRAINING.md](SERVER_TRAINING.md)。
