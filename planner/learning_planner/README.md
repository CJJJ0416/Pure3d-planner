# 学习规划器

本分支的学习规划方法按 `neupan/` 管理。方法内部的职责目录、命令和已有实验说明见 [NEUPAN](neupan/README.md)。

```text
learning_planner/
└── neupan/
    ├── data/            GT/VST 路线、支撑监督、回放样本与数据集
    ├── geometry/        Tomogram、分层支撑图与落脚区域几何
    ├── models/          支撑策略、落脚模型与支撑序列网络
    ├── training/        路径、落脚和支撑序列训练
    ├── inference/       路径生成、beam 搜索、可达性与 rollout
    ├── evaluation/      对照、修订审计与序列 benchmark
    ├── visualization/   ROS/RViz、预测预览和训练进展绘图
    ├── verification/    组件回归检查
    └── __main__.py      方法命令行入口
```

Usage（工作区根目录，使用相应的 Python 环境）：

```bash
python -m planner.learning_planner.neupan --help
python -m planner.learning_planner.neupan.training.landing_train --help
python -m planner.learning_planner.neupan.data.landing_gpmp --help
python -m unittest planner.learning_planner.neupan.verification.component_tests
```

调用推理接口示例：

```python
from planner.learning_planner.neupan.inference.path_generator import PathGenerator
```

数据、权重和已有结果继续使用工作区的既定存储目录；方法内部按本目录的公开命令选择数据与输出位置。
