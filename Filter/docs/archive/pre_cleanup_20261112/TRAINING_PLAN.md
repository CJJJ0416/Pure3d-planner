# 当前训练计划

最新工作流与执行命令统一在 [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)。2026-11-11已按用户最新要求调整为直接局部目标块，不使用整栋WFC大场景训练。

- Train seed：821000、821001、821002、821003。
- Val seed：822000。
- Test seed：823000。
- 各seed组8种局部案例，四个标准变体。测试另有稀疏、少次数残影套件。
- Frozen Utonia Tiny，原486维核心特征缓存，训练投影、几何分支和FilterHead。
- 五组均衡SmoothL1，beta0.1。训练seed42，真实batch8，AdamW，50epoch。
- GT保留分界固定0.5。验证选best与预测阈值，独立测试不调参。
- runs/filter_local_10cm_v3保存best.pt、last.pt、记录和可视化。

局部几何及已知来源监督不证明完整新建筑上的泛化能力。真实ROT stair只做迁移查看。
