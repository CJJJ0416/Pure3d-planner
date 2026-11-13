# Pure3D workspace

建筑点云处理与三维可通行性：rough PCD → Filter → Reconstruction → 条件化 trav_cost → A* → GPMP。
各阶段分别训练与验收。数据均使用地图原点坐标，单位为米，不把预期的模块串接描述为已经完成的端到端训练。

| 模块 | 当前代码职责 |
|---|---|
| [Filter](Filter/README.md) | 冻结 Utonia Tiny，输出全部原始点的 keep_score。当前参考为不含六维统计的 relative 版本，筛除属于显式后处理 |
| [Reconstruction](Reconstruction/README.md) | 完整 Mesh 监督、缺损点云分块、几何自编码与条件 Flow Matching、表面点生成 |
| [Traversability](Traversability/README.md) | 建筑 PCD 与机器人条件 → FiLM → 完整三维归一化代价点云，包含训练、恢复、评估与推理 |
| [Planner](planner/README.md) | DC 三维表面图、A* 搜索和 GPMP 优化，以及交互式起终点 |
| [增量扫描](common/scanning/README.md) | Warp 逐点射线、连续轨迹与 IMU、Faster-LIO 配准、Mesh 连续监督、分块及原生可视化 |

当前扫描默认每帧10,000射线、LiDAR 10 Hz、IMU 200 Hz、pitch=0.785 rad、运动速度1.5倍，初末额外静止均为0秒。数据生成使用纯 Warp，Isaac Sim 用于原生可视化。Faster-LIO 使用现有 ROS 2 underlay。运行入口和可配置路径见扫描说明。

通用 PCD／ROS I/O 集中于 [common/pointcloud_io](common/pointcloud_io/)，VST 和 terrain-generator 位于 common，PTv3 与共用分块工具位于 utils。各模块按 configs、data、models、training、inference、visualization、experiments、docs 组织。

## 源码与本地产物

- [源码检出说明](docs/SOURCE_CHECKOUT.md)：PTv3 固定子模块、terrain-generator 源码快照及外部运行环境。
- [文件架构](docs/WORKSPACE_LAYOUT.md)和[跨模块架构](docs/NETWORK_ARCHITECTURE.md)：目录职责与接口。
- [Filter 操作流程](Filter/docs/TRAINING_WORKFLOW.md)和[Cost 操作流程](Traversability/docs/TRAINING_WORKFLOW.md)：训练、恢复、测试和独立推理。
- datasets、runs、weights、缓存、编译产物和本地 Seen2Scene 参考 checkout 不随源码提交。已完成实验档案中的本地结果链接需配套实验数据才能打开。

默认只保留 best.pt、last.pt 两个训练权重入口。所有权重与数据在本地独立管理，不纳入 Git。
