# 当前数据准备

活动入口为 `python -m Filter.data.scoring --config Filter/configs/scoring_mesh_full.json`。使用现有 `pure3d-ptv3` 环境，不使用已删除的 `datasets/cache/wfc_runtime` 路径。

1. 直接生成六个集中目标的局部组件，楼梯复用 terrain-generator 部件，走廊、栏杆和静止物体由组件工具组装。不是先生成整栋 WFC 建筑再裁剪。
2. 每个组件输出完整 Mesh、面片身份和 overview。表面按 3 cm 采样，不对最终 XYZ 降采样。
3. 每场景固定四变体：静态、稀疏悬空、悬空加短拖影、悬空加少量运动残影。保持完整可靠背景，只有轻微测量噪声，不制造缺口和条带。
4. 使用同一完整静态 Mesh 计算连续标签，保存逐点来源和变换记录。
5. 按 map 网格切成 4×4×3 m 核心和 1 m 上下文，保存索引，不为每盒重复保存 PCD。
6. 冻结 Tiny 编码各上下文一次，缓存原始 486 维特征和核心映射，几何归一化仅统计训练集。

| split | seed | 组件 |
|---|---:|---|
| train | 821100 | 带栏杆直梯 |
| train | 821101 | 往复梯 |
| train | 821102 | 旋转梯 |
| train | 821103 | 走廊 |
| val | 822100 | 带栏杆直梯 |
| test | 823100 | 平台 |

总计 24 个变体、333 个空间盒、2,175,141 个核心点。Train/Val/Test 分别为 1,490,564 / 338,322 / 346,255 点。只固定一套 nominal 测试，不将旧版稀疏缺口套件列为本轮已完成测试。

完整命令见 [操作流程](TRAINING_WORKFLOW.md)，字段见 [数据契约](DATA_CONTRACT.md)。旧 WFC 和 10 cm 数据生成器留作复用与历史复现，旧说明见 [整理前快照](archive/pre_cleanup_20261112/README.md)。
