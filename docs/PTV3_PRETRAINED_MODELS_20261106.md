# Pure3D：PTv3 预训练权重调研与项目起点

记录日期：2026-11-06。目标：对已扫描的大型建筑点云进行串行清理与修复，随后结合机器人能力预测三维可通行代价。本文是模型调研，不代表已完成模型下载、训练或真实数据验证。

## 项目流程

```text
大型建筑 PCD
  -> 分块与重叠上下文
  -> 识别并过滤行人、临时物体和杂波
  -> 恢复有证据的地面、楼梯与栏杆结构
  -> 修复后的静态建筑点云，保留实测/预测/未知信息
  -> 结合机器人能力预测逐点或表面 trav_cost
  -> 三维地图和楼层连接信息 -> 规划器
```

预训练编码器输出特征，不会自动完成上述任务。语义、去噪、补全和 trav_cost 需要有对应任务头与训练数据。点上色也不能直接说明未知空间可通行或楼层之间存在连接。

## 当前源码

- 本地源码：`/home/cjh/Pure3D_ws/PointTransformerV3`，git commit `4e277e6`，包版本 `0.1.2`。
- 本机默认 `python3` 查询不到 torch 或 point_transformer_v3，本项目目录下未发现 `.venv`。没有做导入、权重加载或模型前向验证。
- 当前 `nvidia-smi` 无法连接 NVIDIA 驱动，本机 CUDA 状态未验证。用户的训练服务器为 A100，尚未连接或核对其显存、驱动和依赖。
- 代码要求 `torch>=2.10`，当前 uv 锁定 `torch==2.12.*`。不改现有导航/Isaac 环境，后续使用独立环境并核对 A100 服务器驱动与 CUDA 构建兼容。
- 本次未在默认 Hugging Face hub 缓存目录发现相应模型缓存目录。其他缓存或服务器上的状态未知。首次下载的可用性、耗时和磁盘空间尚未验证。

源码：[预设配置](../PointTransformerV3/docs/presets.md) · [模型预设和类别](../PointTransformerV3/src/point_transformer_v3/presets.py) · [依赖](../PointTransformerV3/pyproject.toml)。

## 本地 package 支持的模型

| 加载名 | 参数量约 | 特征维度 | 输出及用途 |
|---|---:|---:|---|
| `ptv3` | 46.2M | 依模型输出层 | 随机初始化，用于训练，无预训练知识 |
| `ptv3_scannet` | 46.2M | 64 | 已训练的 ScanNet-20 语义分割，默认可 `predict` |
| `sonata` | 108.5M | 1232 | 自监督 3D 特征编码器，可选 ScanNet-20 线性探针 |
| `sonata_small` | 38.6M | 992 | Sonata 较小编码器，本地预设不含 ScanNet 头 |
| `concerto` | 108.5M | 1232 | 2D–3D 自监督衍生的默认 base 编码器 |
| `concerto_origin` | 108.5M | 1232 | Concerto 论文原始 base 权重 |
| `concerto_siglip` | 108.5M | 1232 | 基于 SigLIP 2 的 base 变体 |
| `concerto_large` | 207.7M | 1728 | 较大编码器，可选 ScanNet-20 线性探针 |
| `utonia` | 137.3M | 1386 | 跨室内、室外 LiDAR、物体等域的编码器，可选 ScanNet-20 线性探针 |
| `utonia_small` | 37.2M | 972 | 较小跨域编码器，本地预设不含 ScanNet 头 |
| `utonia_tiny` | 9.3M | 486 | 更小跨域编码器，本地预设不含 ScanNet 头 |

维度是该 package 的 `embed` 接口对上述编码器返回的逐输入点特征宽度。`ptv3_scannet` 的默认语义预测维度为 20 类，不是 64 类。参数量和速度不可直接等同。

Concerto 的 Hugging Face 仓库还提供 `concerto_small`、`concerto_tiny` 和 `concerto_large_outdoor` 检查点。本地 `docs/presets.md` 将它们列为尚无普通预设的权重，户外 demo 有专门路径，不应假设均可直接 `ptv3.load(name)`。

## 输入和输出

`coord` 是 `(N,3)` 的米制 XYZ，`color` 是 `(N,3)`、范围 0–255 的 RGB，`normal` 是 `(N,3)` 法向。Sonata 官方示例的 `(N,1232)` 是特征，不是代价值或语义标签。`ptv3.load("sonata", head="scannet20")` 才加载对应线性分割头。

本地 `prepare` 对缺少模型要求的 `color` 或 `normal` 会报错。Utonia 和默认 Concerto 权重官方支持缺失颜色/法向时传入同形状全零数组，但这不等于缺失信息没有性能影响。真实 LiDAR XYZ-only 的 Sonata 使用方式不能从前述承诺推断，需要单独核验和比较。不把 LiDAR 强度伪装成 RGB。

ScanNet-20 在本地代码中的类别为 wall, floor, cabinet, bed, chair, sofa, table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator, shower curtain, toilet, sink, bathtub, otherfurniture。**无 person、railing、stair 类**。所以 `ptv3_scannet.predict` 及已有 ScanNet-20 线性头无法直接输出专门行人、栏杆和楼梯类别。独立的人体/栏杆/楼梯标签和任务头仍然需要设计和训练。

## 对本项目的建议

优先比较 `utonia_small`、`utonia` 与 `sonata`。Utonia 对 XYZ-only 输入有官方缺失模态处理，适合先做 LiDAR 特征基线。Sonata 作为室内预训练特征对照，需要真实输入条件和效果验证。保留当前 `ptv3_scannet` 作为 ScanNet-20 示例，不能当成行人分割器。上述只是假设和候选，并非性能排名。

第一阶段比较相同建筑场景、相同点云块、观测条件、分辨率和一致标签。单独检查细栏杆保留、楼梯踏面与真实缺口、行人过滤、耗时与峰值显存。补全区域区分观测、推测和未知。对大楼采用带重叠的分块并记录世界坐标与融合方式。

第二阶段以修复后的 PCD 为输入，融合机器人能力参数。用高质量场景在多组 VST 参数下生成 trav_cost 和独立有效/未知标记。当前 VST 无地面观测也可能映射为障碍代价，因此不能只凭最终 cost 恢复支撑真值。模型模仿 VST 参数规则不等于真实机器人已验证可通行。

优先固定同一 PCD 和同一模型，只改变机器人能力条件，检查输出是否与相应 VST 变化一致。再用独立楼宇、不同扫描质量和机器人执行结果验证。

## 参考

- [PTv3 官方仓库与模型列表](https://github.com/Pointcept/PointTransformerV3)
- [Sonata 官方仓库](https://github.com/facebookresearch/sonata)
- [Concerto 权重说明](https://huggingface.co/Pointcept/Concerto)
- [Utonia 权重说明](https://huggingface.co/Pointcept/Utonia)
- 权重与代码许可不同。`ptv3_scannet` 权重在本地预设标为 MIT，其余编码器权重多为 CC BY-NC 4.0 或带额外许可，具体以各自许可证及本地 `NOTICE` 为准。
