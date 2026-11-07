# 当前训练代码与官方 PTv3 对照

日期：2026-11-05。比对起点是用户指定的 [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)。本轮为源码比对、中文注释和文档，训练行为保持不变。

## 官方来源与版本

| 来源 | 固定版本 | 用途 |
| --- | --- | --- |
| PointTransformerV3 | `4e277e6802eb3a6299fd71f707394d918b96f5fc`，包版本 0.1.2 | 实际调用的模型、体素化、点云合批和加速后端 |
| Pointcept | `77f87abe69bf098ebccf3e9b08400963954af6ac` | 官方训练组织、ScanNet 配方和 checkpoint 流程参考 |

官方 PTv3 的 [README](https://github.com/Pointcept/PointTransformerV3/blob/4e277e6802eb3a6299fd71f707394d918b96f5fc/README.md#model-zoo) 将训练配置与实验记录指向 Pointcept。本地直接调用 PTv3 包，在 `src/ptv3_tomography` 中实现地图回归任务的训练器。

已从官方固定提交逐文件读取并比较 `TOMOGRAPHY_UPSTREAM_PIN.json` 中的 46 份文件：**本地文件与官方 46/46 相同，差异为 0**。原始比对记录位于本地 `outputs/ptv3_official_code_review_20261105/source_comparison.json`。中文注释只加入项目适配层。

## 逐项对照

| 部分 | 官方代码 | 当前项目实现与差异 |
| --- | --- | --- |
| 骨干 | [PTv3 model.py](https://github.com/Pointcept/PointTransformerV3/blob/4e277e6802eb3a6299fd71f707394d918b96f5fc/src/point_transformer_v3/model.py) | `models/dense_perception.py` 直接构造官方 `PointTransformerV3`，完整编码器/解码器共同训练 |
| 体素化 | [prepare](https://github.com/Pointcept/PointTransformerV3/blob/4e277e6802eb3a6299fd71f707394d918b96f5fc/src/point_transformer_v3/preprocess.py#L46) | `data/training_cache.py` 调用同一函数。项目选择 XYZ、0.1m、首点代表、保留雷达原点 |
| 点云合批 | [collate](https://github.com/Pointcept/PointTransformerV3/blob/4e277e6802eb3a6299fd71f707394d918b96f5fc/src/point_transformer_v3/preprocess.py#L147) | `collate_frames` 调用官方 collate。各帧点连续拼接，offset 保存边界，一次骨干前向处理整个 batch |
| 数据加载 | [Trainer.build_train_loader](https://github.com/Pointcept/Pointcept/blob/77f87abe69bf098ebccf3e9b08400963954af6ac/pointcept/engines/train.py#L284) | 均使用真实 batch、多 worker、pin_memory、持久 worker。项目增加 CPU mmap、显式 spawn 和 prefetch_factor |
| 洗牌与尾 batch | 同上 | 项目用 seed+epoch 的 sampler 支持帧游标恢复，在 sampler 中处理 drop_last。本地保留尾 batch |
| GPU 传输 | [Trainer.run_step](https://github.com/Pointcept/Pointcept/blob/77f87abe69bf098ebccf3e9b08400963954af6ac/pointcept/engines/train.py#L185) | 同样使用 non_blocking 传输，worker 只读取 CPU 数据 |
| 前反向 | 同上 | 当前 v2 每个真实 batch 一次前向、一次 backward、一次 optimizer.step，梯度累计次数为 1 |
| AMP | [PTv3 加速说明](https://github.com/Pointcept/PointTransformerV3/blob/4e277e6802eb3a6299fd71f707394d918b96f5fc/docs/acceleration.md#precision) | 项目使用 BF16，无 GradScaler。Pointcept 通用 AMP 分支使用 GradScaler，两者实现有区别 |
| 优化器 | [build_optimizer](https://github.com/Pointcept/Pointcept/blob/77f87abe69bf098ebccf3e9b08400963954af6ac/pointcept/utils/optimizer.py#L23) | 同为 AdamW，block 使用低学习率组。项目按官方 Block 类型分组，CUDA 使用 fused 实现 |
| 学习率 | [ScanNet 配方](https://github.com/Pointcept/Pointcept/blob/77f87abe69bf098ebccf3e9b08400963954af6ac/configs/scannet/semseg-pt-v3m1-0-base.py) | 沿用 OneCycle/cos 调度方式。项目峰值 0.001/0.0001，官方该配方为 0.006/0.0006 |
| epoch 定义 | [default_config_parser](https://github.com/Pointcept/Pointcept/blob/77f87abe69bf098ebccf3e9b08400963954af6ac/pointcept/engines/defaults.py#L108) | 项目一个 epoch 对应一次数据遍历，整段预算由 schedule_total_epochs 固定。官方还通过 data.train.loop 重复数据集 |
| 地图头 | 官方该配方使用逐点语义分类头 | `ChunkedMapDecoder.forward_batch` 是项目新增的三近邻插值与地图 MLP，输出 cost/height 各 `[B,6,64,64]` |
| 损失 | 官方该配方为 CrossEntropy+Lovasz | `batched_perception_loss` 对 cost、height 使用标签可用性掩码的 SmoothL1，先按帧归一化，support loss 关闭 |
| 保存与恢复 | [CheckpointSaver/Loader](https://github.com/Pointcept/Pointcept/blob/77f87abe69bf098ebccf3e9b08400963954af6ac/pointcept/engines/hooks/misc.py#L169) | 沿用临时文件替换，保存模型/优化器/scheduler。项目增加 RNG 与帧游标，BF16 scaler 为 None。同文件系统的阶段/best 采用原子硬链接，避免复制模型字节 |
| 计时与日志 | 官方 InformationWriter 按步写标量 | 项目按窗口同步和归约。batch_time_s 是窗口摊销时间，GPU 事件时间包含区间内执行与等待 |
| 多 GPU | 官方训练器支持 DDP | 当前项目 v2 使用单 GPU，尚未接入 DDP |

骨干行为以固定 PTv3 包及其 Config 为准。当前配置使用新训练路径的默认兼容开关，旧 ScanNet 权重的 legacy CPE 行为由官方 preset/转换机制处理。

## 输入与数据增强的任务适配

官方上述 ScanNet 配方使用 RGB+normal 六通道、0.02m GridSample、随机代表点和多种空间/颜色增强。本项目的输入是传感器系 XYZ，体素 0.1m，域随机化在场景和采集阶段完成，缓存中保存固定代表点及原始 cost、高度，掩码直接来自标签可用性。

点云、重力方向与地图查询共享雷达参考系，因此训练时保持原点和尺度。项目地图/up 按 batch 轴堆叠，合批不跨帧混合空间。骨干输出 `[sum(N_i),64]`，随后由任务地图头预测全部高度带和栅格。

## 精度和梯度注释

`SingleFramePerception` 声明 `precision="fp32"`，用于没有外部 autocast 的独立调用。训练器在前向外层进入 BF16，官方 [PointTransformerV3._autocast](https://github.com/Pointcept/PointTransformerV3/blob/4e277e6802eb3a6299fd71f707394d918b96f5fc/src/point_transformer_v3/model.py#L400) 保留调用者已建立的上下文，因此当前训练确实使用混合精度。

KNN 距离、回归输出与损失归约使用 FP32，主要矩阵计算使用 BF16。地图头插值读取的骨干 features 保持可微。日志中的 detach 只用于统计，未知目标处的 detach 只排除无监督位置。骨干和地图头均参与训练。

## GPU 利用率与显存

`nvidia-smi` GPU 利用率表示采样区间内有 kernel 执行的时间比例，显存使用量表示存储占用，二者衡量不同内容。该定义见 [NVIDIA nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)。多 worker、预取与减少同步 I/O 能缩短 GPU 空闲间隙。

显存实际从 batch=4 的 0.82 GiB 增长到 batch=16 的 1.45 GiB、batch=32 的 2.30 GiB。本地每帧平均约 360 个有效回波，体素后约 279 个 token，输入规模较小。BF16 缩小部分激活占用，CPU mmap 和预取队列使用主机内存。

当前模型共 46,163,618 个参数。按 FP32 参数、梯度和两份 Adam 矩估算，主要固定状态约为 `46,163,618 × 16 bytes ≈ 0.69 GiB`。此外还有激活、混合精度缓存和算子工作区。batch 增大主要增加这些随样本变化的部分，固定模型状态只保存一份。

代码报告的 max_memory_allocated 是 PyTorch 分配峰值，max_memory_reserved 还包含分配器缓存。整卡 nvidia-smi 数字还包含 CUDA 上下文与其他进程。不同统计口径分别记录。

## 本轮修改范围

中文注释已加入 `training/batch_pretrain.py`、`data/training_cache.py`、`models/dense_perception.py`、`models/perception.py`、`training/runtime.py` 和 `training/pretrain_state.py`。对注释前后的可执行语法结构作了只读比对，结果相同。本轮没有新增测试或启动训练，既有吞吐记录见 [LOCAL_BATCH_TRAINING.md](LOCAL_BATCH_TRAINING.md)。

后续生产 I/O 调整与真实训练结果见 [SHORT_PIPELINE_PROFILE.md](SHORT_PIPELINE_PROFILE.md)。上面的注释前后逻辑一致结论对应首次注释整理，后续已修改任务层的 checkpoint 保留与数据导出，官方核心保持原样。

2026-11-06：任务监督已切换为 `mask_cost = mask_height = target_available`，三重观测筛选与 v1 串行训练路径已删除。最新清理和实际运行见 [TEACHER_SUPERVISION_CLEANUP.md](TEACHER_SUPERVISION_CLEANUP.md)。
