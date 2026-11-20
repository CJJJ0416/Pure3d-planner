# Pure3D workspace

建筑点云处理与三维可通行性：观测点云 → KNN / Filter → 条件化 trav_cost → A* → SCAN。独立静态工具保留 GPMP。
各阶段分别训练与验收。数据均使用地图原点坐标，单位为米，不把预期的模块串接描述为已经完成的端到端训练。

| 模块 | 当前代码职责 |
|---|---|
| [Filter](utils/Filter/README.md) | 冻结 Utonia Tiny，输出全部原始点的 keep_score。当前参考为不含六维统计的 relative 版本，筛除属于显式后处理 |
| [Traversability](utils/Traversability/README.md) | 建筑 PCD 与机器人条件 → FiLM → 完整三维归一化代价点云，包含训练、恢复、评估与推理 |
| [Planner](planner/light_planner/README.md) | DC 三维表面图、A* 搜索和 GPMP 优化，以及交互式起终点 |
| [增量扫描](Simulator/scanning/README.md) | Warp 逐点射线、连续轨迹与 IMU、Faster-LIO 配准、Mesh 连续监督、分块及原生可视化 |
| [Gazebo 仿真](Simulator/unitree_gazebo/README.md) | Building-no-rails 场景、A1 控制、MID360、SCAN 与 Faster-LIO 在线建图，保留 terrain 场景入口 |
| [共用在线组件](common/README.md) | [人工目标 → 每次 goal 整图 cost → DC+A* → SCAN](common/planning/docs/EXPLORATION.md) |

当前扫描默认每帧10,000射线、LiDAR 10 Hz、IMU 200 Hz、pitch=0.785 rad、运动速度1.5倍，初末额外静止均为0秒。数据生成使用纯 Warp，Isaac Sim 用于原生可视化。Faster-LIO 使用现有 ROS 2 underlay。运行入口和可配置路径见扫描说明。

全部导航启动入口统一在 [Entrance](Entrance/README.md)，按 `launch/`、`config/`、`shell/` 管理。切换默认场景修改 `Entrance/config/startup.yaml`，修改场景资源与出生点使用 `Entrance/config/scenes/`。

当前 Gazebo／SCAN 局部 cost 实验可用 `./Entrance/shell/total_terminal.sh`：先清理本工作区旧会话，再按就绪状态和间隔打开 run、keyboard、Faster-LIO、cost 四个独立终端。独立 cost 入口为 `ros2 launch Entrance/launch/sim/cost.launch.py`；仿真、LIO 和键盘分别使用 Entrance 下的单项 launch。命令、间隔与数据目录见 [局部 cost 启动说明](common/cost/docs/LOCAL_COST.md)。

局部 cost 默认缓存查询 core 的完整已观测上下文与 1 m halo，输出限制在 LIO 独立维护的 6×6×5 m 盒子。实验模块与对照脚本仅保留在本地，不随源码提交。

通用 PCD／ROS I/O 集中于 [common/pointcloud_io](common/pointcloud_io)。仿真、扫描与场景生成位于 [Simulator](Simulator/README.md)，VST 基线和独立 PTv3 历史工程位于 [third_party_lib](third_party_lib/README.md)。Pure3D 使用的 PTv3 与共用分块封装位于 utils。核心组件位于 `utils/Filter` 与 `utils/Traversability`；规划器按 `planner/light_planner`、`planner/learning_planner` 分类。

## 源码与本地产物

- [源码检出说明](docs/SOURCE_CHECKOUT.md)：PTv3 固定子模块、terrain-generator 源码快照及外部运行环境。
- [真机安装与启动](docs/REAL_WORLD_SETUP.md)：MID360 外参、重力方向、独立 LIO 启动和 共用在线组件与 SCAN 接口。
- [文件架构](docs/WORKSPACE_LAYOUT.md)和[跨模块架构](docs/NETWORK_ARCHITECTURE.md)：目录职责与接口。
- [Filter 操作流程](utils/Filter/docs/TRAINING_WORKFLOW.md)和[Cost 操作流程](utils/Traversability/docs/TRAINING_WORKFLOW.md)：训练、恢复、测试和独立推理。
- datasets、runs、weights、缓存、编译产物和本地 Seen2Scene 参考 checkout 不随源码提交。已完成实验档案中的本地结果链接需配套实验数据才能打开。

默认只保留 best.pt、last.pt 两个训练权重入口。所有权重与数据在本地独立管理，不纳入 Git。

本地产物按[存放规则](docs/STORAGE_LAYOUT.md)分类：监督和原始采集放datasets，实际checkpoint放weights，runs只保留filter/cost/reconstruction的训练记录与权重链接。独立推理及ROS输出使用outputs，编译产物使用.cache。
