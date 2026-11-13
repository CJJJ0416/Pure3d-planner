# Filter：单张 XYZ 的连续保留系数预测

更新日期：2026-11-11。D0/D1/D2 数据程序已实现，Utonia Tiny 冻结编码与 FilterHead 已实现。现有 Traversability best.pt 不是 Filter 权重。

## 当前任务与 DUFOMap 参考边界

输入单张地图系 XYZ PCD，内部按盒读取。运动物体全部运动位置低分，静止人物、家具和准确断连建筑结构保留。配准重影按最终几何误差连续评分。

[DUFOMap](https://arxiv.org/html/2403.01449v2#S3) 使用逐帧点云和传感器位姿的射线空闲证据。当前学生仅有融合 XYZ，通过几何特征学习残影线索，不复现其射线判定。PCD 默认 VIEWPOINT 不是采集位姿。

## 数据与教师

当前版本为 filter_local_10cm_v3，直接生成集中目标的局部组件。局部楼梯、墙面、栏杆及走廊 → 含静止物体的静态 Mesh → 表面采样 → 密度、缺口和轻微噪声 → 动态轨迹与配准副本 → 最终每点连续标签 → 小盒。具体接口见 [数据契约](DATA_CONTRACT.md)，命令见 [数据准备](DATA_PREPARATION.md)。

四变体为静态、动态、配准、混合。人物、椅子及可移动物体的同一模板同时用于静止对照和运动轨迹。墙、楼梯、栏杆的正确断连点通过删除周围环带构造，不加入连通性删点规则。

静态参考包括建筑和静止物体，运动实例不加入参考。每场景一次 Open3D 三角面索引，批量计算 d。运动来源 keep_target=0，其他 keep_target=exp(-0.5*(d/0.03)^2)。精确为 0 或 1 的端点允许存在，训练仍为连续回归。

语义、来源、轨迹、时间步、Mesh 和标签都不进入编码器。四变体的基础观测共享固定随机种子。输入和参考采用同一地图坐标，所有增强属性按最终 PCD 行序保存。裁剪和邻居缺失不降低剩余准确点的目标。配准副本重新落回正确表面时不因副本身份被强制标零。

原 v1 语义非目标删除方式已退出当前任务，旧数据保留用于参考。完整 Mesh 与人工缺口不是真实扫描可见性。真实迁移能力后续单独评价。

## 4. 网络和损失接口

```text
rough XYZ → 固定 Utonia Tiny → 多尺度特征
    → inverse 对齐 + 相对坐标 + 局部几何统计
    → 轻量 FilterHead → Sigmoid → keep_score [N,1]
    → 保留索引同时选择 XYZ_keep 与共享特征 F_keep
```

### 4.1 首版通道和可训练参数

`N` 是当前窗口的原始输入点数，`M` 是编码器体素代表点数。Tiny 五层通道为 `18+36+72+144+216=486`。使用全部层的 upcast，保留细层几何与粗层上下文。

| 分支 | 输入 | 层与输出 |
|---|---|---|
| 固定编码器 | 同一窗口 XYZ，缺失 RGB/normal 置零 | `F_voxel[M,486]` |
| 可训练特征投影 | F_voxel | Linear 486→128、LayerNorm、GELU，随后按 inverse 恢复为 `[N,128]` |
| 可训练几何分支 | 3 维点到体素代表点相对坐标 + 6 维局部统计 | 9→32→32，各隐藏层 LayerNorm、GELU |
| 可训练评分分支 | 两路拼接 `[N,160]` | 160→128→64→1，隐藏层 LayerNorm、GELU、Dropout 0.1 |
| 输出 | `[N,1]` logit | Sigmoid → keep_score |

点到代表点的偏移在原米制坐标计算，再除以物理体素宽度。六维几何统计为近邻 16 平均距离、第 16/64 邻居距离、0.15/0.5 m 半径计数、平均近邻距离与第 64 邻居距离比。先做稀疏代表点统计，再按自己的映射恢复到原始点，不能混用几何统计和 PTv3 的 inverse。统计归一化参数仅由 train 估计。

Tiny 使用当前本地 preset 的 grid_size=0.01、scale=0.5，物理体素约 0.02 m。地图坐标与编码器内部的缩放、居中单独记录，输出保持原地图坐标。该结构已实现并完成局部0.1 m输入训练，实测结果见 runs/filter_local_10cm_v3/REPORT.md。输入先做0.1 m体素代表点降采样，编码器内部仍沿用Tiny的原预处理尺度。

训练缓存保存冻结编码器的 `[M,486]` 特征，而不是活动投影产生的 128 维特征。投影仍需反向更新。不同点数的窗口用 offset 拼接，inverse 按各样本的 M 前缀和修正，不把窗口补零成固定点数。

### 4.2 损失与筛选

F 阶段只更新评分头和其投影。首版损失为按结构与伪影来源均衡归约的 `SmoothL1(keep_score, keep_target)`，beta 从 0.1 起步。R 阶段使用已冻结评分头的实际选择结果，不再次运行点云编码器。

具体归约为 `loss = sum(sample_weight_i * smooth_l1(w_pred_i - w_target_i)) / sum(sample_weight_i)`。sample_weight 负责平衡地面、墙、细栏杆与各类干扰的数量，不改变它们的连续目标系数。训练使用全部有标签的输入点。

训练前向输出系数后直接计算损失，不先执行 `keep_score >= threshold` 再只监督保留下来的点。硬阈值仅用于验证、导出与部署。Mesh 距离可作为离线目标和验证依据，不作为过滤后点集的唯一训练目标。单独最小化 `sum(w_pred_i * distance_i^2)` 会出现将所有权重压向零的解。把完整目标 Mesh 转成点云后要求 Filter 单独补足其覆盖，也超出了只删除已有点的能力。

几何统计作为可学习输入，不附加最大连通分量删除或固定比例删点。目标与伪影的空间关系由输入上下文提供，需要覆盖漂浮残片与主体同时可见的训练窗口。

### 4.3 接入 RECON

导出 `keep_score[N]`、`keep_indices[K]` 和 `XYZ_keep[K,3]`。公共特征仍按体素保存，RECON 使用 `F_voxel[inverse[keep_indices]]` 或其延迟索引视图取得原始 486 维共享特征，随后训练自己的投影。评分头的 128 维内部表示不替代公共特征。

一个原始点属于唯一核心窗口，外扩点只提供上下文。最终系数和保留索引恢复到整张输入 PCD 的行序。首版用离线训练的固定前级连接后续 RECON，不新增第二次 PTv3 编码。

## 5. 权重来源

| 部分 | 初始权重 | F 阶段更新 |
|---|---|---|
| E_shared / Utonia Tiny | 官方 Pointcept/Utonia 的 utonia_tiny.pth | 冻结，保持 eval |
| Filter 特征投影 | 随机初始化 | 更新 |
| 几何分支与评分 MLP | 随机初始化 | 更新 |

官方 Tiny 权重提供几何与语义特征，不包含本项目的连续保留评分头。官方 ScanNet 读出权重对应另一规格的 Utonia，不能直接作为 Tiny 的 FilterHead。现有 Traversability best.pt 继续用于 cost，不能当成已训练的 Filter 权重。

训练完成后，Filter 的 best.pt / last.pt 保存本次训练的 head、投影和所依赖的固定编码器身份。后续再冻结它，训练 RECON。

来源：[Utonia 官方加载及 head 兼容说明](https://github.com/Pointcept/Utonia#quick-start)、[Trimesh 表面采样与面片索引](https://trimesh.org/trimesh.sample.html)、[SmoothL1](https://docs.pytorch.org/docs/2.10/generated/torch.nn.SmoothL1Loss.html)。

## 6. 查看与评估

- 评分误差：连续保留系数 MAE。
- 用户指定漂浮残片：低分覆盖率、筛除率、残留点数。
- 主体结构：楼梯、栏杆、墙面等分别统计误删和保留率。
- 固定近景：原始点、GT 保留系数、预测保留系数、筛除结果。

Filter 数据 RViz 使用独立 keep_target 色标，并分别显示两类重影及静态对照。未来预测使用 keep_score。Traversability 的 trav_cost 不作为漂浮点标签，高通行代价墙面仍可能需要保留。
