"""通用 PCD 读写和 ROS PointCloud2 转换，保留原始坐标与点序。"""
from pathlib import Path

import numpy as np


def _read_header(stream):
    header = {}
    for raw_line in stream:
        line = raw_line.decode('ascii').strip()
        if not line or line.startswith('#'):
            continue
        key, *values = line.split()
        header[key] = values
        if key == 'DATA':
            break
    return header


def read_pcd_fields(path):
    """读取二进制或 ASCII PCD 的全部命名字段，保留字段类型。"""
    with Path(path).open("rb") as stream:
        header = _read_header(stream)
        types = {"F": "f", "I": "i", "U": "u"}
        fields = []
        for name, size, kind, count in zip(
            header["FIELDS"], header["SIZE"], header["TYPE"], header["COUNT"]
        ):
            shape = () if int(count) == 1 else (int(count),)
            fields.append((name, "<" + types[kind] + size, shape))
        dtype = np.dtype(fields)
        encoding = header["DATA"][0]
        if encoding == "binary":
            rows = np.fromfile(stream, dtype=dtype, count=int(header["POINTS"][0]))
        elif encoding == "ascii":
            rows = np.loadtxt(stream, dtype=dtype, ndmin=1)
        else:
            raise NotImplementedError(f"PCD encoding: {encoding}")
    return rows


def fields_xyz(fields):
    """从命名字段中提取 [N,3] float32 XYZ，不变换坐标或点序。"""
    return np.column_stack([fields[name] for name in ("x", "y", "z")]).astype(np.float32)


def read_pcd_xyz(path):
    """读取 PCD 的全部 XYZ，其他字段不进入坐标数组。"""
    with Path(path).open('rb') as stream:
        header = _read_header(stream)
    if header['DATA'][0] == 'binary_compressed':
        import open3d as o3d

        cloud = o3d.t.io.read_point_cloud(str(path), remove_nan_points=False,
                                        remove_infinite_points=False)
        return np.ascontiguousarray(cloud.point.positions.numpy(), dtype=np.float32)
    return fields_xyz(read_pcd_fields(path))


def read_pcd(path):
    """读取 XYZ 和法向量，文件未提供法向量时返回等形状零数组。"""
    rows = read_pcd_fields(path)
    coord = fields_xyz(rows)
    if "normal_x" in rows.dtype.names:
        normal = np.column_stack([rows[name] for name in ("normal_x", "normal_y", "normal_z")]).astype(np.float32)
    else:
        normal = np.zeros_like(coord)
    return coord, normal


def write_pcd(path, coord, cost=None, *, field_name="trav_cost"):
    """按输入点序写出二进制 float32 XYZ，可指定附带标量字段的名称。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["x", "y", "z"] + ([] if cost is None else [field_name])
    width = len(fields)
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS " + " ".join(fields)
        + "\nSIZE " + " ".join(["4"] * width)
        + "\nTYPE " + " ".join(["F"] * width)
        + "\nCOUNT " + " ".join(["1"] * width)
        + f"\nWIDTH {len(coord)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        + f"POINTS {len(coord)}\nDATA binary\n"
    )
    values = coord if cost is None else np.column_stack((coord, cost))
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(np.asarray(values, dtype="<f4").tobytes())
