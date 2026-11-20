# Pure3D 文件架构

更新日期：2026-11-20。根目录：`/home/cjh/Pure3D_ws`。

```text
Pure3D_ws/
├── Entrance/                # 导航框架统一启动入口
│   ├── launch/              # sim/real 单项方案，含中文注释、Usage 和配置路径
│   ├── config/              # 场景、LIO、cost、SCAN、导航与 RViz 配置
│   └── shell/               # 整体启动、构建与内部环境库
├── utils/                   # 核心组件与共用工具
│   ├── Filter/              # 点云保留评分与过滤
│   ├── Traversability/      # 条件化可通行代价的数据、训练与推理
│   └── PointTransformerV3/  # 固定版本编码器源码
├── common/                  # 原 Entry 拆分后的共用组件
│   ├── cost/                # 局部/整图 cost、worker 与监测
│   ├── mapping/             # 地图、稳定上下文与体素盒
│   ├── planning/            # 目标接入、参考与跟踪
│   ├── visualization/       # ROS/RViz 与离线显示
│   ├── bag_tools/           # 采集、回放与离线数据处理
│   ├── native/              # C++ 数组、体素与过滤
│   └── pointcloud_io/       # PCD/PointCloud2 I/O
├── planner/
│   ├── light_planner/       # 原 planner，表面图/A*/优化与候选选择
│   └── learning_planner/
│       └── neupan/          # 本分支学习规划方法，内部按 data/models/training/inference 等分类
├── Simulator/               # 仿真、机器人与场景/扫描生成
│   ├── SCAN-Planner/
│   ├── quadruped_ros2_control/
│   ├── Mid360_imu_sim/
│   ├── livox_ros_driver2/
│   ├── unitree_gazebo/
│   ├── scanning/
│   └── geneator/
├── third_party_lib/         # 独立参考工程与外部基线
│   ├── ptv3_tomography/
│   └── vst_global_planner/
├── datasets/                # 数据与教师标签
├── weights/                 # 实际 checkpoint
├── runs/                    # 训练记录与权重链接
├── outputs/                 # 独立推理、ROS 与导航输出
├── .cache/                  # 编译与仿真工作目录
└── docs/                    # 跨组件说明
```

## 代码入口

导航运行入口见 [Entrance](../Entrance/README.md)。各业务模块保留实现代码，导航运行 YAML 统一放在 `Entrance/config/`。

以工程根目录为工作目录，使用分类后的完整模块名。

| 工作 | 入口 |
|---|---|
| 批量生成 | `python -m utils.Traversability.data.generate` |
| 原生教师 | `python -m utils.Traversability.data.teacher` |
| 分块/近邻 | `python -m utils.Traversability.data.blocks` |
| 训练 | `python -m utils.Traversability.training.train` |
| 评估 | `python -m utils.Traversability.training.evaluate` |
| 完整点云推理 | `python -m utils.Traversability.inference.predict` |
| 绘制 GT 对比 | `python -m utils.Traversability.visualization.visualize` |
| RViz | `python -m utils.Traversability.visualization.rviz` |
| 多模型 RViz | `python -m utils.Traversability.visualization.compare_rviz` |

完整命令与环境见 [Traversability 训练流程](../utils/Traversability/docs/TRAINING_WORKFLOW.md)。

Filter 已完成全点评分数据、特征缓存、训练、测试与独立推理。活动入口为 `utils.Filter.data.scoring`、`utils.Filter.data.features`、`utils.Filter.training.train`、`utils.Filter.training.evaluate`、`utils.Filter.inference.predict` 和显式后处理 `utils.Filter.inference.threshold`。命令见 [操作流程](../utils/Filter/docs/TRAINING_WORKFLOW.md)，实测见 [训练档案](../utils/Filter/docs/TRAINING_ARCHIVE_20261112.md)。


## 数据与结果

每个数据版本集中保存场景、输入 PCD、原生 Tomogram 标签与 blocks.json。按完整建筑或几何族划分 train/val/test。`datasets/cache/` 保存可复用缓存。

回归数据 `datasets/wfc_regression_20261127/` 保存大型测试建筑的条件标签和固定楼梯案例，其几何数组引用 datasets 中已有场景，不依赖 runs。

训练记录写入 `runs/{filter,cost,reconstruction}/<experiment>/`，实际checkpoint写入 `weights/<module>/<experiment>/`，runs中的best.pt和last.pt为相对符号链接。独立推理、ROS及导航写入outputs，编译和仿真工作目录使用.cache。数据和教师标签保存在datasets，详见[存放规则](STORAGE_LAYOUT.md)。Filter历史数据路径仍以实际数据包为准。
