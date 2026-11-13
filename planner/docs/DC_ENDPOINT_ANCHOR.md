# PCD锚点与DC端点吸附

起终点统一为地图坐标系中的输入PCD支撑位置，单位米。输入不是机体中心，也不要求地图地面Z为0。

## 地图加载

DC顶点的 `surface_source_voxel` 复用cost映射已有的最近邻编号，指向 `planner.xyz`，也就是导出 `voxel_cost.pcd` 的点序。来源关系描述与输入几何的最近邻关联。

节点先沿用现有cost阈值剪枝，端点候选再要求法向与反重力方向夹角不超过 `slope_max`。将候选按来源体素分组，每组固定选择距离该PCD体素质心最近的DC顶点，距离完全相同时选择原始DC顶点编号最小者。跨阶边的来源净空条件只约束新增边，不改变本端点索引或原三角网格。

对具有候选的来源体素质心建立KD-tree。来源映射、分组和KD-tree均在加载地图时构建一次，构建耗时记为 `setup.endpoint_index_s`。

## 查询

```text
输入PCD支撑位置XYZ
    ↓ 最近邻查询
PCD体素锚点
    ↓ 固定来源映射
对应DC支撑点
    ↓ 原有法向偏移
A*搜索坐标
```

选择端点不读取另一个端点、连通分量或搜索结果。没有换层重试，也不改变cost阈值。

`anchor(points)` 返回与输入形状一致的PCD体素锚点坐标。`snap(points)` 仍返回 `(distance, node, support_xyz)`，其中距离是原查询到DC支撑点的欧氏距离。`astar()` 和 `plan()` 的调用方式不变。

RViz初始化读取原 `queries.json` 并转成PCD锚点。拖动释放时保存、显示PCD锚点，规划时再由同一来源映射定位DC节点，不把DC支撑坐标再次当作PCD输入。

## 保存

成功和失败查询均输出 `endpoint_snap.json`，数组依次对应起点和终点：

- `query_xyz`：传给本次规划的PCD支撑位置。
- `pcd_anchor_xyz`、`source_voxel_indices`：输入体素锚点与来源编号。
- `dc_support_xyz`、`node_ids`、`surface_vertex_ids`：选中的DC支撑点和图节点、原表面编号。
- `query_to_anchor_m`、`anchor_to_dc_m`、`query_to_dc_m`：三个坐标阶段之间的距离。

搜索计时中的 `endpoint_snap_s` 为本次两端点解析耗时。结果仍分别保存DC支撑路径、A*搜索路径和现有GPMP轨迹。
