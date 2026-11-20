# Pure3D 共用组件

原 Entry 按职责拆入本目录。启动入口与运行配置统一由 [Entrance](../Entrance/README.md) 管理。

| 目录/文件 | 职责 |
|---|---|
| `cost/` | 局部/整图 cost 接收、推理、GPU worker、会话与耗时监测 |
| `mapping/` | 点云地图、LIO 盒子、稳定上下文与体素过滤 |
| `planning/` | 人工目标接入、表面查询、局部参考与轨迹跟踪 |
| `visualization/` | RViz 显示、点云/Marker 发布与离线绘图 |
| `bag_tools/` | bag 采集、回放、分段、监督和离线处理 |
| `native/` | 共用 C++ 数组、体素与过滤实现及数组绑定 |
| `pointcloud_io/` | PCD 与 ROS PointCloud2 读写 |
| `io.py`、`paths.py` | JSON/NumPy、ROS 消息转换与统一目录定位 |
| `experiments/` | 本地对照实验，按既有规则不随源码提交 |

实际运行入口示例：`common.cost.local`、`common.cost.session`、`common.cost.worker`、`common.planning.navigation` 和 `common.bag_tools.real_bag`。GPU worker 由调用模块管理。

Usage（工作区根目录）：

```bash
source /opt/ros/humble/setup.bash
ros2 launch Entrance/launch/sim/cost.launch.py model:=tiny rviz:=true
ros2 launch Entrance/launch/sim/navigation.launch.py --show-args
./Entrance/shell/total_terminal.sh --scene terrain-prepared --dry-run
```

[局部 cost](cost/docs/LOCAL_COST.md)、[规划与跟踪](planning/docs/PURE_PATH_FOLLOW.md)、[整图工具](cost/docs/WHOLE_MAP.md)、[bag 工具](bag_tools/docs/BAG_WORKFLOW.md)。

仿真与场景资源位于 [Simulator](../Simulator/README.md)，两套规划器位于 [planner](../planner/README.md)，Filter 和 Traversability 核心组件位于 [utils](../utils/README.md)。

所有任务共用 `common.pointcloud_io`，不在模型或可视化代码里另写 PCD 解析器：

```python
from common.pointcloud_io import read_pcd_xyz, read_pcd_fields, read_pcd, write_pcd

xyz = read_pcd_xyz("building.pcd")         # 全部原始 XYZ，float32 [N, 3]
fields = read_pcd_fields("prediction.pcd") # 全部命名字段，含 trav_cost
xyz, normals = read_pcd("building.pcd")   # 无法向量字段时 normals 为零
write_pcd("prediction.pcd", xyz, cost)
```

读写保持源坐标系、点数与行序，不做降采样、分块或代价归一化。支持 binary 和 ASCII PCD 读取，写出 binary float32 PCD。ROS 转换使用 `pointcloud2_xyz` 和 `pcd_to_pointcloud2`，仅调用时导入 ROS。后者沿用现有二进制 float32 PCD 到消息的直接转换，供 RViz 发布使用。

