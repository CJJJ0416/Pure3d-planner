# 当前训练版本

本轮训练已经完成。活动配置为 `configs/scoring_mesh_full.json` 和 `configs/train_scoring_mesh_full.json`。

- 四个训练 seed、一个固定验证 seed、一个独立测试 seed。完整组件加稀疏残影，无输入体素降采样。
- 官方 Utonia Tiny 冻结/eval。缓存原 486 维特征，训练投影、9 维几何分支和评分头。
- 核心 4×4×3 m + 1 m 上下文。场景均衡采样，真实 batch 8，workers 4。
- 来源组内按结构均衡，再对现存来源组均衡 SmoothL1，beta=0.1。
- AdamW 1e-3、weight decay 0.01，50 epoch、1400 更新，5% warmup + cosine 到 1e-5，BF16。
- 每轮完整验证，以来源均衡 Macro MAE 选 best，次级指标为全点 MAE。只保留 best.pt 和 last.pt。
- 默认只输出全点评分。GT 0.5 是可选二值诊断分界，不改变连续监督。0.2 后处理由用户指定。

本轮 best 为 epoch 24，last 为 epoch 50。量化结果、环境、真实点云与规划记录见 [训练档案](TRAINING_ARCHIVE_20261112.md)，可复现命令见 [操作流程](TRAINING_WORKFLOW.md)。
