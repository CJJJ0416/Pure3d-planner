# Filter 逐点打分器：文献与方案依据

调研日期：2026-11-10。状态：文献调研。后续用户决定采用连续保留系数，并将 ROT stair 图中指定的外围漂浮扫描残片纳入低分目标。当前工程设计见 [ARCHITECTURE.md](ARCHITECTURE.md)，历史论文中的二分类不作为当前回归目标的限定。当前训练数据采用离线 Mesh 采样增强，论文中的扫描流程仅作方法参考。

阅读目的为 literature-review，输入范围为论文正文、摘要及官方代码的混合摘录，论文类型为 algorithmic，提取重点为 method-pipeline、limitations 和 reproducibility，深度为 standard。以下分别注明论文内容与本项目建议。

## 1. 本项目实际包含三种判断

| 判断 | 例子 | 单纯密度过滤的不足 |
|---|---|---|
| 是否属于目标结构 | 楼梯、墙、地面、顶面、栏杆、柱梁、约定保留的规则障碍 | 人物和杂物也可能形成密集、连续的表面 |
| 是否为扫描伪影 | 离表面噪点、漂浮噪声小簇 | 噪声簇自身也可能很密集 |
| 是否为漏扫留下的真实结构 | 断开的扶手、薄墙残片、仅扫描到局部的踏步 | 稀疏、体量小或断连都可能来自真实结构 |

因此，任务更接近“目标结构保留评分”，同时学习扫描伪影。完整表面补点仍交给后续 RECON。

## 2. 优先参考的工作

### 2.1 PointCleanNet：最直接的离群点评分参考

论文为 *PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds*，Computer Graphics Forum，2020。预印本发表于 2019 年。

**论文与代码内容：**先预测离群点并删除，再估计保留点的修正位移。网络采用基于 PCPNet 的局部点云 patch 编码。官方离群训练入口设置一个标量输出，使用 L1Loss，评估代码以 0.5 划分离群点。本项目的连续保留系数与损失定义属于工程改造。

**本项目建议：**借鉴“每个已有点一个删除/保留分数”的接口。其几何离群定义不直接覆盖移除人物、保留建筑结构的语义选择。已有共享 PTv3 可以替代它的独立 patch 编码器，这属于本项目改造。

来源：[论文](https://arxiv.org/abs/1901.01060)、[官方代码及预训练模型下载入口](https://github.com/mrakotosaon/pointcleannet)、[离群训练代码](https://github.com/mrakotosaon/pointcleannet/blob/master/outliers_removal/train_pcpnet.py)、[评估阈值代码](https://github.com/mrakotosaon/pointcleannet/blob/master/outliers_removal/eval_pcpnet.py)。

### 2.2 Sonata：冻结编码器、训练轻量任务头的直接依据

论文为 *Sonata: Self-Supervised Learning of Reliable Point Representations*，CVPR 2025。

**论文内容：**linear probing 冻结编码器，将多尺度特征 up-cast 到输入分辨率，再训练线性层。论文也报告了稀疏 SLAM 点云上的域差异，不能由室内分割成绩推定在任意扫描上都有效。

**本项目建议：**用来支持“公共编码器固定，FilterHead 单独训练”的路径。先做简单读出，再判断是否需要增加空间聚合。共享权重可缓存特征，但任务头仍需要本项目保留标签。

来源：[论文，第 4.3 节与附录 B.1](https://arxiv.org/html/2503.16429v1)、[官方代码及权重](https://github.com/facebookresearch/sonata)。

### 2.3 Utonia：当前共享底座的首选候选

论文为 *Utonia: Toward One Encoder for All Point Clouds*，ICML 2026。

**论文内容：**以 PTv3 为底座进行跨域预训练，并处理颜色、法向缺失。包含 linear probing、decoder probing 和全量微调比较。部分细粒度任务上，空间读出头优于单一线性层。

**本项目建议：**沿用已规划的 Utonia 共享编码器。Filter 可先采用小型非线性评分头，Tiny 与 Small 分别训练适配头。不将预训练权重直接视为楼梯、栏杆或扫描伪影分类器。论文整体结果也不能当作 Tiny/Small 在本项目上的结果。

来源：[论文](https://arxiv.org/html/2603.03283v2)、[官方代码、输入缺模态说明及模型入口](https://github.com/Pointcept/Utonia)。

### 2.4 Superpoint Transformer / EZ-SP：大场景结构与上下文参考

论文分别为 *Efficient 3D Semantic Segmentation with Superpoint Transformer*，ICCV 2023，以及 *EZ-SP: Fast and Lightweight Superpoint-Based 3D Segmentation*，ICRA 2026。

**论文内容：**前者以多层 superpoint 关系进行大场景语义分割。后者将分区改为可学习的 GPU 流程，减少 CPU 预处理开销，输出仍是场景语义类别。

**本项目建议：**借鉴局部结构与更大范围上下文联合判断的思路。它们本身是完整分割流程，不能直接称为接在固定 PTv3 后的现成 FilterHead。首版不因这两篇工作再引入第二套编码器。

来源：[SPT 论文](https://openaccess.thecvf.com/content/ICCV2023/html/Robert_Efficient_3D_Semantic_Segmentation_with_Superpoint_Transformer_ICCV_2023_paper.html)、[EZ-SP 论文](https://arxiv.org/html/2512.00385v1)、[统一官方代码与模型入口](https://github.com/drprojects/superpoint_transformer)。

### 2.5 BIM-Guided Virtual-to-Real：最接近合成建筑监督的数据路线

论文为 *A BIM-Guided Virtual-to-Real Framework for Component-Level Semantic Segmentation of Construction Site Point Clouds*，Sensors，2026。

**论文内容：**BIM 组件赋标签，经 HELIOS++ 虚拟扫描得到带身份的点，再分块训练语义网络并应用到真实建筑扫描。涉及密度、视点、噪声等域差异。其主要类别为墙/柱、梁、顶面、地面和 other，不能据此认为已经验证了楼梯、栏杆与漂浮点过滤。

**本项目建议：**借鉴面片/构件身份随射线命中传到点标签的制作方式，使用本项目 Warp 扫描与结构定义。来源仅用于生成监督，不作为离线 PCD 推理输入。

来源：[论文全文](https://www.mdpi.com/1424-8220/26/1/308)。本次未核验可直接复用的官方训练代码和权重。

## 3. 名称相近但任务不同的工作

| 工作 | 实际输出与用途 | 与当前 Filter 的差别 |
|---|---|---|
| [Score-Based Point Cloud Denoising，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_Score-Based_Point_Cloud_Denoising_ICCV_2021_paper.html) | 估计三维 log-density 梯度并迭代移动点 | score 是三维方向，不是一个 0–1 保留分数 |
| [IterativePFN，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/de_Silva_Edirimuni_IterativePFN_True_Iterative_Point_Cloud_Filtering_CVPR_2023_paper.html) | 多个内部迭代模块把点移向干净表面 | 主要解决坐标去噪，不能直接替代建筑结构保留判断 |
| [SSPCN，Pattern Recognition 2023](https://www.sciencedirect.com/science/article/pii/S0031320323000675) | 自适应下采样、特征补偿、上采样与坐标重建 | 是联合清理与重建网络，输出接口不同于保留原点并打分 |
| [Confidence-guided progressive denoising，2026](https://www.sciencedirect.com/science/article/pii/S1110016826004746) | 逐阶段预测位移，用置信度调节修正幅度 | 置信度用于位移更新，不能直接等同于目标结构标签 |

本节仅用于区分任务。没有在本机安装或复现这些方法，也不依据跨论文指标排序。

## 4. FOV 断连片段如何处理

以下是本项目的目标约定，不是上述论文已替我们验证的结论。实际监督使用连续保留系数。

- 主体目标结构的准确回波获得高保留系数，正常测距误差连续影响质量。
- 用户在 ROT stair 图中指出的主体外围漂浮扫描残片为低保留目标，包含局部像墙或栏杆的片段。
- 当前 v2 的运动来源全部低分，配准重影按静态表面距离评分，静止物体和准确断连结构保留。
- 对仍属于目标几何的缺测断连片段，不仅凭稀疏或断连就改为低分。
- KNN、密度和局部形状可以输入网络，但不能直接充当“真伪”标签。

理由是扫描缺失改变了观测集合，没有改变剩余回波原本所属的构件。单份 XYZ PCD 也不能保证区分几何上相同的真实残片和伪影，网络判断仍依赖训练分布与上下文。

应分别保留两组难例：稀疏且断开的目标结构，以及内部密集或呈平面状的漂浮扫描残片。用户图示明确指定了后一类的低分目标，不将该图像标记冒充三维逐点标注。

## 5. 下一步 Filter 结构的候选起点

```text
rough XYZ
    → 固定 E_shared：Utonia，每窗口一次前向
    → 多尺度 upcast 特征 F_voxel
    → 用 inverse 对齐到原始点
    → 融合局部相对坐标与几何描述
    → 轻量 MLP → keep_logit → Sigmoid → keep_score [N,1]
    → 同一保留索引选择 XYZ_keep 与 F_keep
    → 后续冻结的接口连接 RECON
```

这个组合是基于文献的工程候选，不是某一篇论文的原样复现。分数只表示目标结构保留倾向，与 trav_cost 分开。F 阶段先训练任务头，完成后冻结，再训练 R。模块之间继续传递坐标与共享特征，不重新编码过滤后的点云。

设计时比较单层读出与小型 MLP，并单列栏杆、楼梯边缘、真实断连片段和漂浮噪声簇的表现。是否增加邻域聚合由这些结果决定。具体通道数、几何特征和损失权重在 FilterHead 设计中确定，不由本次调研直接锁定。

本地接口已核对：[多尺度 upcast](../../utils/PointTransformerV3/src/point_transformer_v3/heads.py) 与 [prepare 返回的 inverse](../../utils/PointTransformerV3/src/point_transformer_v3/preprocess.py)。体素特征恢复是索引操作，不会产生缺失几何，也不会让预训练特征自动拥有本项目的二元标签含义。

现有 [inspect_floating.py](../../Reconstruction/visualization/inspect_floating.py) 是几何可疑度可视化工具，不是已训练 Filter，也不作为本方案的真值来源。
