# 本地真实 batch 训练与 GPU 峰值

历史实测记录。整理后的最新全流程实测见 [SHORT_PIPELINE_PROFILE.md](SHORT_PIPELINE_PROFILE.md)。目录整理后，当前安装与启动入口见 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。本地旧 outputs 在整理执行时已不存在，以下数值及产物路径保留为历史记录。

2026-11-05，RTX 4060 Laptop 8 GiB。完整 PTv3 + cost/height 地图头，共 46,163,618 个可训练参数。当前按用户要求只运行本地。

## 实际完成

- 复用九区域场景中 ramp_01、open_stairs_01 两条路线的 40 个真实扫描帧。
- 每帧 1000 条射线，有效点 302–404，平均 360.475，体素后平均 278.575 点。
- 输出 cost/height 各 `[B,6,64,64]`。真实合批前向、batch loss、反向与优化器更新均已完成。
- batch=4 完成 5 epoch / 50 次更新，batch=16 完成 30 epoch / 90 次更新。
- batch=32 先训练 10 epoch / 20 次更新，再在新进程恢复并接续到 30 epoch / 60 次更新。两段共 1200 帧次，恢复包含 AdamW、OneCycle、RNG 与 epoch 游标。

## 速度和峰值

| 真实 batch | workers | 稳定吞吐（帧次/秒） | GPU 采样均值 | GPU 采样峰值 | CUDA 分配峰值 |
| --- | --- | --- | --- | --- | --- |
| 4 | 2 | 62.52 | 84.6% | 92% | 0.82 GiB |
| 16 | 2 | 115.60 | 86.1% | 94% | 1.45 GiB |
| 32 | 2 | 118.42 | 83.1% | 100% | 2.30 GiB |

稳定吞吐使用后续 epoch，排除首个本地 epoch、checkpoint 与评估耗时。GPU 数值来自 nvidia-smi 采样，batch=32 的时间窗口包含一次中间 checkpoint 等待。整卡显存峰值包含桌面，三组分别为 1503、2139、3142 MiB，CUDA 分配峰值只统计 PyTorch。

本地默认选 batch=16。batch=32 的吞吐增加约 2.4%，CUDA 分配峰值增加约 59%。两者均实际完成训练。

同样的 40 帧重复 5 epoch，共 200 帧次，旧版真实 batch=1、累计4帧、workers=0 用时 148.74 秒，其中重复 checkpoint 写入 128.61 秒。新版 batch=4 用时 9.01 秒，保存耗时 0.48 秒，整段约快 16.5 倍。新版复用了训练数组与已编译 kernel，收益包含数据、batch、后端与 I/O 修改。首次编译记录单独保留在 `outputs/local_ptv3_batch_20261105/batch/`。

本地新缓存准备约 0.99 秒。batch=16 的第 30 epoch 训练 cost MAE 为 1.24/60，高度 MAE 为 0.96 cm。这些是当前 40 帧上的训练指标。

## 本地直接训练

以下命令从 glocal2-nav 根目录执行。首次自动生成 `registry.json` 同目录下的 `training_cache_v2`，后续直接复用 mmap 数组。输出目录使用新的名称。

```bash
export PTV3_PROJECT="$PWD/src/ptv3_tomography"
.venv_ptv3_gpu/bin/python "$PTV3_PROJECT/scripts/training/run_pretraining.py" \
  --registry outputs/local_ptv3_batch_20261105/registry.json \
  --config "$PTV3_PROJECT/configs/perception_pretrain_rtx4060_batch.json" \
  --output outputs/ptv3_local_batch_next
```

配置采用真实 batch=16、workers=2、BF16、auto attention/conv/LayerNorm、query_chunk=1024、30 epoch。末尾不足16帧的 batch 保留。已有 checkpoint 可通过 `--init-checkpoint` 进行仅权重初始化，或通过 `--resume` 恢复 v2 全部训练状态。分段接续时固定 `schedule_total_epochs`，各段只改变 `epochs`。

服务器后续显式选择 `configs/perception_pretrain_h100_batch.json`。当前没有连接服务器，也没有沿用旧版 9–11 天的时间估计。

## 精简后的执行流程

一次性准备点、up、地图目标及最终掩码 → mmap DataLoader → 官方点云 collate → PTv3 → 并行地图头 → masked SmoothL1 → backward → AdamW / OneCycle → checkpoint。

训练循环不重新生成标签、不重建射线掩码、不做逐帧 SHA256 或全量来源扫描。训练状态日志和 checkpoint 取消逐次 fsync。每个保存点只序列化一次，需要的阶段/best 副本采用同文件系统原子硬链接，避免再次复制模型字节。

已删除三个独立预检/守护入口：`scripts/diagnostics/check_environment.py`、`isaac_training/scripts/server/check_raycaster_server.py`、`isaac_training/scripts/server/run_monitored.py`。没有新增防御性测试或硬件配额。

## 本地产物

所有数据和运行结果保存在 Git 忽略的 `outputs/local_ptv3_batch_20261105/`。

| 产物 | 路径 |
| --- | --- |
| 汇总数据 | `performance_summary.json` |
| GPU 曲线与峰值图 | `gpu_training_comparison.png` |
| batch=16 模型 | `batch16/training/last.pt` |
| batch=32 接续模型 | `batch32_resume/training/last.pt` |
| 逐步吞吐与 GPU 原始记录 | 各运行目录的 `training/updates.jsonl` 与 `gpu.csv` |

## 与服务器比较所需的纯文本

```text
GPU型号：
代码版本：
稳定窗口实际处理帧数：
窗口总耗时（秒）：
其中 checkpoint / 评估耗时（秒）：
真实 batch / 梯度累计次数 / workers：
平均有效点数 / 体素后点数：
GPU利用率均值 / 显存峰值：
```

训练帧/秒 = 实际处理帧数 ÷（窗口耗时 − checkpoint 耗时 − 评估耗时）。不同 batch 使用帧/秒比较。H100 的实现优化倍率使用同卡、同数据的前后记录计算。
