# 建筑点云过滤、条件化代价与规划

更新日期：2026-11-17。工作目录为 `/home/cjh/Pure3D_ws`。

## 当前处理顺序

```text
在线：注册扫描 → LIO 独立盒子与稳定上下文 → KNN → PTv3 + cost head → A* → SCAN
离线：原始 PCD → 可选 Filter → 条件化 cost → 三维 A* → 可选 GPMP
```

在线 KNN 使用保守的邻域统计删除孤立点，学习式 Filter 仍是独立模块。保留点保持原始坐标。没有观测的区域保持未知。

## 编码与点序

输入 XYZ 使用米制、重力对齐的地图坐标。编码使用 `4×4×3 m core` 和 `1 m` 上下文，每个输出点只归属一个 core。上下文可以重复参与不同 core 的编码，最终输出不拼入重复上下文点。

GPU worker 在启动时加载确定的 checkpoint 并记录 SHA256。默认总入口选择 Tiny best49，Small best56 可显式选择。代表点采样与原始点对应关系由现有预处理维护，实际推理精度单独记录，不以训练配置代替运行值。

## 机器人条件与教师

学生条件顺序为 `[clearance_min,slope_max,step_max,stand_ratio]`。当前局部运行值为 `[0.5,0.4,0.2,0.0]`。教师 `interval_free` 固定为 `0.5 m`，参数对应见 [教师—学生参数对照](TEACHER_STUDENT_PARAMETERS.md)。

VST 教师的归一化 cost 是训练参考，不是实测机器人执行成功率。按完整建筑或几何族划分训练、验证与测试，同一建筑的相邻块保持在同一组。

## 过滤与代价分工

Filter 的 `keep_score` 描述观测应保留的程度，墙体和不可通行障碍也可能有高保留分数。`trav_cost` 描述机器人条件下的地形代价。当前在线通过 KNN 在编码前去除孤立点，再计算 cost。

幽灵点过滤需同时评价孤立噪声去除、楼梯点误删和保留地面的 cost 变化。完整处理时间包括上下文维护、快照、KNN、进程传输、编码、cost head、拼接和发布。

## 输出与验收

- `/entry/local/cost_cloud` 为 XYZ + `trav_cost`，RViz 使用其降采样显示版本。
- 模式 4 的 A* 采用严格 `cost < 0.8`，输出 `/pure_path`，SCAN 保留原有占用与局部轨迹优化。
- 静态规划工具保留 A* → GPMP 和独立地图后端。
- 空间连通、预测精度、计算耗时和实际到达分别记录。

## 使用入口

[Entry](../Entry/README.md)、[在线启动与耗时](../Entry/docs/LOCAL_COST.md)、[静态 planner](../planner/README.md)、[Cost 训练流程](../Traversability/docs/TRAINING_WORKFLOW.md)、[Filter 流程](../Filter/docs/TRAINING_WORKFLOW.md)、[文件布局](WORKSPACE_LAYOUT.md)。
