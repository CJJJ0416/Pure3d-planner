# RECON：原始密度、替换式课程训练

流程为连续 WFC 地形 → 整图随机缺损 → 固定区块 → 冻结 PTv3／Filter 缓存 → 几何自编码 A → 条件 Flow Matching B。输出保留点与新增表面点，Filter 和 Traversability 权重不变。

正式配置采用六张独立地形、每图八块。数据全部集中在 `datasets/recon_terrain_curriculum_v1`，结果集中在 `runs/recon_terrain_curriculum_v1`。完整 GT、完整输入、验证／测试长期复用，训练缺损只保留当前阶段和切换时的临时新版本。

```bash
cd /home/cjh/Pure3D_ws
export PYTHONPATH="$PWD/.deps/recon${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/recon_mpl
PY=/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python
# 只准备固定数据和训练 stage 1，不预生成后续阶段
$PY -m Reconstruction.experiments.run initialize
# 显式训练。先 A 5→50 轮，固定验证达标后 B 5→50 轮
$PY -m Reconstruction.experiments.run train
# B 晋级自动调用此接口，也可在训练停止后手工准备
$PY -m Reconstruction.experiments.run prepare-stage --stage-id 2
```

断点使用每阶段的 `best.pt` 和 `last.pt`。重复执行 train 会读取现有 last.pt 继续。B 在第 10、20、35 轮结束后独立准备下一阶段，替换旧缺损和条件缓存。不会重置优化器、学习率或步数。A 的固定验证 Precision／Recall@3 cm 均须达到 90%，否则报告门槛未达并停止进入 B。

`status.json`、`ae.log`、`flow.log` 和每阶段 `history.json` 显示实际进度。固定验证每 5 轮执行，测试在阶段结束后对 best.pt 执行，测试指标不参与选权重。结果写到固定 validation_latest／test 目录，避免逐次追加大量点云。

详情见 [数据契约](docs/DATA_CONTRACT.md)、[监督与 Loss](docs/SUPERVISION_LOSS.md) 和 [训练说明](docs/TRAINING_PLAN.md)。
