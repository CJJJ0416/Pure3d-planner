"""ROS PointCloud2 与 PCD/XYZ 转换，ROS 依赖按需导入。"""
from pathlib import Path

import numpy as np


def pointcloud2_xyz(message):
    """提取 PointCloud2 的全部 XYZ，不使用 cost，也不筛选 NaN。"""
    from sensor_msgs_py import point_cloud2

    return point_cloud2.read_points_numpy(
        message, field_names=["x", "y", "z"], skip_nans=False,
    ).astype(np.float32, copy=False)


def pcd_to_pointcloud2(path, header, cost_multiplier=1.0):
    """将项目输出的二进制 float32 PCD 转为 ROS 消息，可缩放代价。

    默认直接使用文件中的点数据字节，只有缩放 trav_cost 时复制数组。
    ROS 依赖只在调用时导入，普通 PCD 读写不需要 ROS。
    """
    from sensor_msgs.msg import PointCloud2, PointField

    with Path(path).open("rb") as stream:
        metadata = {}
        for raw_line in stream:
            key, *values = raw_line.decode("ascii").strip().split()
            metadata[key] = values
            if key == "DATA":
                break
        data = stream.read()
    names = metadata["FIELDS"]
    count = int(metadata["POINTS"][0])
    if cost_multiplier != 1.0:
        values = np.frombuffer(data, dtype="<f4").reshape(count, len(names)).copy()
        values[:, names.index("trav_cost")] *= cost_multiplier
        data = values.tobytes()
    return PointCloud2(
        header=header, height=1, width=count,
        fields=[PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(names)],
        is_bigendian=False, point_step=4 * len(names), row_step=count * 4 * len(names),
        is_dense=True, data=data,
    )
