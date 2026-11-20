# 一次搜索生成多个局部候选

本文描述已有代价点云上的离线工具。现有 `SparseSurfacePlanner` 的 cost 阈值、分层、相邻柱配对、对角约束和端点吸附保持原样。同一搜索模块现已接入 flight type 4 的 light 后端，在线续接、RViz和切换说明见 /home/cjh/Pure3D_ws/common/planning/docs/PURE_PATH_FOLLOW.md。SCAN跟随接口保持不变。

`planner/light_planner/graph/candidates.py` 接收已经构建好的 CSR 图。每个查询调用一次 SciPy 原生 Dijkstra，得到从起点出发的最短路树，再回溯不同终点。边权沿用 A* 的 `距离 × (1 + 5 × 两端平均 cost)`。加权 CSR 在同一图版本内复用。

默认局部路径长度为 0.8–3 m，近目标候选可以短于 0.8 m。搜索 cost 上界为 `3 × (1 + 5 × cost_limit)`，之后按真实累计路径长度筛选。每个终点只有一条树路径，这不是同一终点的多种绕障拓扑搜索。也不是有长度约束的多标签最短路，如果该终点的最便宜树路径超过 3 m，另一条更短但更贵的路径不会成为候选。

评价函数全部使用点云的支撑坐标，不混入机身高度偏移。设起点为 s，终点为 e，人工目标为 g，反重力单位向量为 u，H=3 m，通行阈值为 c_max。

```text
Pz   = (|(s-g)·u| - |(e-g)·u|) / H
Pxyz = (||s-g|| - ||e-g||) / H
C    = Σ[边长 × 两端平均 cost] / (H × c_max)
score = 8 × Pz + 2 × Pxyz - C
```

分数越大越优。Z 使用接近目标高度的进展，上楼、下楼都可奖励，同高度目标下偏离该高度会扣分。8 是最大的权重，但不是任何微小高度收益都压倒其他项的硬性字典序。cost 使用整条路径积分，不只看终点，也不把无点区域视为空闲。权重均可通过命令行调整。

先对全部合格终点评分，再按分数从高到低回溯并去重。路径按归一化弧长重采样 24 点。两条路径的 XYZ 均方根距离不超过 0.25 m 且终点反重力高度差不超过 0.1 m 才视为重复，保留分数较高者，最多保留 16 条。达到数量上限后的终点不再处理，计数记录为 `unexamined_due_to_cap`。最佳候选就是执行参考选择，输出保存为 `selected.csv`，工具不发布运动指令。

在线 `LocalCandidateQueries` 启用 DC 式踏面跨阶检查，具体规则见 [在线路径接口](../../../common/planning/docs/PURE_PATH_FOLLOW.md)。独立 `candidate_demo` 沿用其显式构图选项。

默认 `cost_limit=0.8`，对应当前 mode 4 的设置。所有已有分层与连边规则仍由原始 `planner/light_planner/surface.py` 与原生构图器决定，评分不会补边或修复跨层问题。

## 复现实验

```bash
cd /home/cjh/Pure3D_ws
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  MPLCONFIGDIR=/tmp/pure3d_candidate_matplotlib  python3 -m planner.light_planner.candidate_demo  --pcd datasets/benchmarks/surface_planning/reference/navigation/maps/9241/input_world.pcd  --cases planner/light_planner/config/local_candidates_cases.json  --repeat 20 --rollout-steps 24  --output outputs/planner/local_candidates_new
```

使用新的输出目录，避免覆盖历史结果。示例包含保存的查询 23、24、查询 22/23 的反向查询，以及一个同高度人工构造目标。统一在保存的地图 9241 上运行，因此查询 22/23 的输入地图与其历史记录中的 8209 不同，不能直接当作历史同图性能对照。

单次查询可以改用 `--start X Y Z --goal X Y Z`，不传 `--cases`。坐标必须与 PCD 一致。默认 frame 为 `world`，盒尺寸为 6×6×5 m，中心为起点支撑坐标。这里没有读取 IMU，反重力方向使用已对齐 world 的 +Z。

`--rollout-steps 24 --advance 0.5` 会在固定保存地图上模拟执行选中路径的约 0.5 m 图边前缀，然后更新局部盒并再次选择。每一步仍只有一次搜索。它使用理想图路径跟随，不包含机器人动力学、SCAN、实时感知或模型重算。三次回到相同位置会报告循环，超出步数则报告未完成，不将候选生成成功视为目标到达。

每组保存：

- `local_input.pcd`、`surface.pcd`，局部原始代价点云与原图搜索节点。
- `candidate_00.csv` 等，保留路径的 XYZ、原始节点 cost 与 node ID。
- `selected.csv`、`no_z_selected.csv`、`candidates.png`，选择与关闭 Z 权重的对照。
- `search_tree.npz`、`endpoints.csv`，完整搜索父指针、距离与合格终点评分。
- `result.json`，候选、去重计数、可重复性、各阶段耗时与选中路径检查。
- 可选的 `rollout.json`、`rollout.csv`，理想移动的逐步决策与真实完成状态。

关闭 Z 的消融从同一份终点特征重新评分，不增加图搜索。实验还单独调用一次已有 A*，检查选中终点的树路径是否具有相同搜索目标值。该检查不属于候选生成耗时。候选逐边检查原图成员关系，另记录选中路径的现有插值检查，均不代表机身碰撞或实际跟踪验收。

`total_query_s` 包含一次搜索、树路径特征累计、评分、回溯去重和原图边检查。不包含 PCD 读取、全图裁盒、表面图准备、网络预测、绘图或磁盘保存，这些时间单独记录。离线从约 4000 万点扫描裁盒明显慢于搜索，不应把查询毫秒数解释成完整感知规划链路耗时。接入现有在线局部体素盒时，应直接使用其数组与同版本图缓存。

本机检查命令：

```bash
python3 -m unittest discover -s tests -p test_local_candidates.py -v
```

检查包括单次搜索调用、低 cost 绕行、上楼与下楼偏好、同层高度偏离、路径积分、路径去重、断开图、长度上限、近目标、坐标旋转和平移，以及输入图不被修改。
