# 点云 I/O

所有任务通过 `common.pointcloud_io` 调用同一套点云读写与转换。

| 文件 | 用途 |
|---|---|
| `pcd.py` | binary/ASCII 字段读取、binary_compressed XYZ 读取、XYZ/法向量提取、binary float32 PCD 写出 |
| `ros.py` | PointCloud2 提取 XYZ，以及项目二进制 float32 PCD 转 PointCloud2 |
| `__init__.py` | 统一导出公开函数，ROS 包仅在转换函数调用时导入 |

```python
from common.pointcloud_io import read_pcd_xyz, read_pcd_fields, write_pcd

xyz = read_pcd_xyz("building.pcd")
fields = read_pcd_fields("prediction.pcd")
write_pcd("copy.pcd", xyz, fields["trav_cost"])
write_pcd("diagnostic.pcd", xyz, scores, field_name="floating_cost")
```

保持原始坐标系、点数和行序。体素采样、分块、坐标变换与代价归一化由调用方处理。

`read_pcd_xyz` 读取 binary_compressed 时延迟调用 Open3D Tensor I/O，不删除非有限点，不读取 VIEWPOINT 作为坐标变换。普通 binary/ASCII 读取不依赖 Open3D。`read_pcd_fields` 的完整字段读取范围仍为 binary/ASCII。
