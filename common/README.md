# Common

集中管理通用点云 I/O、VST 与场景生成工具。

| 目录 | 用途 |
|---|---|
| `pointcloud_io/` | PCD 读写、XYZ 提取及 ROS PointCloud2 转换 |
| `vst_global_planner/` | VST 内核、原生规划器和已有组合场景工具 |
| `geneator/` | 官方 terrain-generator 与 WFC 粗糙多层场景生成入口 |
| `Mid360_imu_sim/` | MID-360 方向表、模型与已有 Gazebo ROS 2 插件 |
| `scanning/` | 逐点 Warp 扫描、带噪 IMU、Faster-LIO、Mesh 监督、分块及 Isaac/RViz 查看 |

预训练 PTv3、坐标处理、分块和编码器封装放在 `../utils/`。

所有任务共用 `common.pointcloud_io`，不在模型或可视化代码里另写 PCD 解析器：

```python
from common.pointcloud_io import read_pcd_xyz, read_pcd_fields, read_pcd, write_pcd

xyz = read_pcd_xyz("building.pcd")         # 全部原始 XYZ，float32 [N, 3]
fields = read_pcd_fields("prediction.pcd") # 全部命名字段，含 trav_cost
xyz, normals = read_pcd("building.pcd")   # 无法向量字段时 normals 为零
write_pcd("prediction.pcd", xyz, cost)
```

读写保持源坐标系、点数与行序，不做降采样、分块或代价归一化。支持 binary 和 ASCII PCD 读取，写出 binary float32 PCD。ROS 转换使用 `pointcloud2_xyz` 和 `pcd_to_pointcloud2`，仅调用时导入 ROS。后者沿用现有二进制 float32 PCD 到消息的直接转换，供 RViz 发布使用。

场景脚本从工程根目录通过 `common/geneator/...` 调用，VST 自带入口通过 `common/vst_global_planner/scripts/...` 调用。具体运行参数在后续数据契约和运行说明中确定。

WFC 多层场景入口为 geneator/generate_wfc_multifloor.py，调用方式与分层装配说明见 [WFC 场景说明](geneator/README.md)。

