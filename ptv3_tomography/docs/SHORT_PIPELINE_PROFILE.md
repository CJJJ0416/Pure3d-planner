# 本地短 epoch 全流程与重复开销清理

2026-11-05，目录整理后重新运行。RTX 4060 Laptop 8 GiB，完整 PTv3 + 地图头共 46,163,618 个可训练参数。没有施加硬件配额，没有新增预检、守护或防御性测试。

## 实际完成

- 九区域现有场景 `third_party/vst_global_planner/outputs/playground_rotated_seed20261104`，路线 `ramp_01`、`open_stairs_01` 各采集前 20 帧。
- 原生 Isaac Lab RayCaster / Warp，2 路合批，每帧 1000 射线。有效回波 302–404，平均 360.475，体素后平均 278.575 个 token。
- 40 帧完成 Tomography 配对、相对路径 portable 导出和一次性 mmap 缓存。
- 真实 batch=16、workers=2、BF16。3 epoch 共 9 次更新，再启动新进程恢复模型、AdamW、OneCycle、RNG 和游标，继续第 4 epoch，完成第 12 次更新。
- 恢复后的 checkpoint 实际加载并导出 0、10、19 三帧，地图形状 `[1,2,6,64,64]`。

最终产物位于仓库根目录的 `outputs/ptv3_tomography/short_pipeline_optimized_20261105/`。训练数据回放的三帧 cost MAE=8.69383/60，高度 MAE=8.14592 cm。没有运行独立 seed 测试集或导航策略。

## 实测时间

下表的进程耗时包含 Python 导入和进程启动。标签、portable 和预测两侧都启用 cProfile。训练与采集首轮更冷，不能把整段耗时差都归为删除检查的收益。

| 项目 | 首轮 | 最终版本 | 口径 |
| --- | ---: | ---: | --- |
| 采集进程 | 19.406 s | 11.962 s | 包含 Isaac 启动、网格加载和退出 |
| 实际扫描循环 | 0.192 s | 0.167 s | 40 帧，含压缩保存、单次 SHA256 |
| 标签生产 | 3.131 s | 2.337 s | 两条路线内部计时，减少 25.4% |
| portable 导出 | 0.413 s | 0.381 s | 两条路线进程累计 |
| 一次性训练缓存 | 1.075 s | 0.622 s | 内部准备时间 |
| 3 epoch 训练进程 | 10.073 s | 4.777 s | 9 次更新，包含启动和末尾保存 |
| 恢复并训练 1 epoch | 7.811 s | 4.568 s | 新进程恢复、3 次更新、保存 |
| 恢复段 checkpoint 保存 | 2.143 s | 0.426 s | 一次序列化及阶段保留 |
| 3 帧预测导出 | 4.673 s | 3.962 s | 包含加载 checkpoint、FP32 推理和导出 |

首轮使用默认线程独立计时缓存，最终缓存计时器按实际训练配置使用 CPU 4 线程，因此缓存行不用于宣称算法加速倍率。训练首轮还包含较多形状初始化或编译。最终后两个 epoch 共 80 帧次，窗口吞吐 120.49 帧次/秒，CUDA 分配峰值 1480.77 MiB，保留峰值 1532 MiB。

这次只有 3 个微小 epoch，GPU 采样窗口包含 worker 启动。200 ms 采样获得 12 个样本，均值 24.2%、峰值 58%，整卡显存峰值 2150 MiB。约 0.66 秒的后续训练窗口不足以作为持续 GPU 利用率测试，原始记录为 `final_gpu.csv`。

先完成了一次清理后的连续全流程，子进程累计 35.467 s。之后修改 checkpoint 保留和元数据 I/O，复用这次真实采集的 40 帧，最终标签准备到推理累计 17.837 s。没有把两个时间窗口拼成一次新的连续全流程总耗时。

## 已删除的重复操作

| 位置 | 删除内容 | 剖析证据 |
| --- | --- | --- |
| `sensors/batched_raycaster.py` | 每批回传整张世界射线方向表并与 CPU 旋转结果比较 | 生产扫描保留命中与原点回传，用于距离和点云计算 |
| `collect_mid360_raycaster.py` | CPU 回传后的额外 Warp 同步、末尾重复来源扫描 | 新增初始化、位姿、查询、压缩、哈希分阶段计时 |
| `pair_mid360_lidar_reference.py` | 导出后的全帧读取、SRU 序列重放、重复来源和原始扫描哈希 | 每条路线 step 读取由 42 次降至首帧 1 次，SHA256 由 145 次降至 71 次 |
| `make_dataset_portable.py` | 每个复制目标已校验后，再用完整 loader 重复扫描 | 每条路线 SHA256 由 91 次降至 50 次 |
| `export_plan_a.py` | 扫描未导出的帧、末尾重读整个 checkpoint 和数据集、解析高度恒等式 GPU 断言、逐帧 fsync | 三帧导出 SHA256 由 93 次降至 10 次 |
| `io.atomic_json` | 每次元数据更新强制 fsync | 文件关闭后原子替换，保留完整文件可见性 |
| `training/batch_pretrain.py` | 为阶段/best 副本再次同步复制约 555 MB 模型 | 同文件系统原子硬链接，实测保留耗时 0.000060 s |

以下为 2026-11-05 的历史测量。当前输出已改为 `runs/<实验名>/best.pt` 和 `last.pt`，不再生成编号副本。历史版本 checkpoint 每次只序列化一次，并用新文件原子替换 `last.pt`。阶段/best 硬链接保留旧 inode，因此后续保存不会改写已经保留的模型。`last.pt`、`checkpoints/` 与 `best_dev.pt` 位于同一输出文件系统。此次 `final_resume/last.pt` 和 `checkpoints/epoch_0004.pt` 共享 inode，文件大小 554,964,309 bytes。新进程实际恢复与导出均已完成。

两条路线的监督格总数保持 cost=1899、height=2693，导出三帧的 cost/height 误差指标与修改前一致。坐标转换、重力高度定义、直接查表、可见性掩码、独立 cost/height loss 和官方骨干均未改变。

## 仍保留的实际工作

一次性准备时的 CPU 坐标/地图契约检查、复制目标的一次哈希、所选预测帧的校验、日志窗口/epoch 边界同步均保留。非阻塞采集进程锁只执行一次，没有等待轮询或硬件配额。训练支持 SIGINT/SIGTERM 和显式 STOP 请求。

v2 训练循环不读取原始 NPZ、不重新生成标签、不逐帧哈希、不调用 fsync，也不调用 `collection.verify_unchanged()`。遗留 v1 路径仍有末尾全量核验，本次 v2 调度未进入该分支。

标签阶段当前主要时间花在真实查询和压缩，两条路线合计约 query=1.145 s、NPZ 压缩=0.750 s。训练缓存只准备一次，训练期间不重复付出这些成本。较大规模采集应复用同一个 Isaac 进程，正式训练应连续运行并降低无必要的进程重启频率。

## 产物

- [机器可读汇总](../../../outputs/ptv3_tomography/short_pipeline_optimized_20261105/performance_review.json)
- [最终数据登记](../../../outputs/ptv3_tomography/short_pipeline_optimized_20261105/final_registry.json)
- [3 epoch 报告](../../../outputs/ptv3_tomography/short_pipeline_optimized_20261105/final_training/report.json)
- [恢复报告](../../../outputs/ptv3_tomography/short_pipeline_optimized_20261105/final_resume/report.json)
- [恢复后的 checkpoint](../../../outputs/ptv3_tomography/short_pipeline_optimized_20261105/final_resume/last.pt)
- [预测报告](../../../outputs/ptv3_tomography/short_pipeline_optimized_20261105/final_prediction/report.json)

日志、cProfile、GPU CSV 和执行脚本一起保存在上述本地 outputs。配置仅缩短 epoch 数，使用真实全骨干、全地图前向和反向。未提交 Git。
