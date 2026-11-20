# Pure3D 规划器

| 组件 | 来源与职责 |
|---|---|
| [light_planner](light_planner/README.md) | 原 planner：表面图、DC/稀疏表面、A*、候选选择和样条/GPMP 优化 |
| [learning_planner](learning_planner/README.md) | 本分支的方法归入 `neupan/`，按数据、模型、训练、推理、评估和显示分类 |

两套实现各自保留配置与说明。在线目标接入和跟踪组合位于 `common/planning/`，启动入口位于 `Entrance/`。

Usage（工作区根目录，使用对应组件要求的 Python 环境）：

```bash
python -m planner.light_planner.plan --help
python -m planner.light_planner.candidate_demo --help
python -m planner.learning_planner.neupan --help
python -m planner.learning_planner.neupan.training.landing_train --help
```

源码移动后，轻量规划器仍在工作区 `.cache/` 与既有缓存目录中编译原生代码。学习规划器继续使用既有数据、权重与 `outputs/planner/` 结果目录。
