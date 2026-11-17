"""Atomic array/JSON IO and ROS message conversions with lazy ROS imports."""
from pathlib import Path
import json
import os
import numpy as np


class JsonlTail:
    """Read only complete new records, retaining a partial trailing line."""
    def __init__(self, path, start_at_end=False):
        self.path = Path(path)
        self.position = 0
        self.partial = b''
        if start_at_end and self.path.exists():
            self.position = self.path.read_bytes().rfind(b'\n') + 1

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open('rb') as stream:
            stream.seek(self.position)
            data = self.partial + stream.read()
            self.position = stream.tell()
        lines = data.split(b'\n')
        self.partial = lines.pop()
        return [json.loads(line) for line in lines if line]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


def ros_stamp(value):
    from rclpy.time import Time
    return Time(seconds=value).to_msg()


def transform_value(message):
    p, q = message.transform.translation, message.transform.rotation
    return np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])


def pose_value(pose):
    p, q = pose.position, pose.orientation
    position, quaternion = np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(np.r_[position, quaternion]).all() or norm < 1e-6:
        raise ValueError('Nonfinite or invalid pose')
    return position, quaternion/norm


def cloud_xyz(message):
    """Read strided/organized PointCloud2 without assuming packed XYZ fields."""
    fields = {f.name: f for f in message.fields}
    formats = []
    for name in ('x', 'y', 'z'):
        field = fields[name]
        if field.datatype not in (7, 8) or field.count != 1:
            raise ValueError('XYZ fields must be scalar float32/float64')
        formats.append(('>' if message.is_bigendian else '<') + ('f4' if field.datatype == 7 else 'f8'))
    dtype = np.dtype(dict(names=['x', 'y', 'z'], formats=formats,
                         offsets=[fields[n].offset for n in ('x', 'y', 'z')], itemsize=message.point_step))
    rows = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                      strides=(message.row_step, message.point_step))
    return np.column_stack([rows[n].ravel() for n in ('x', 'y', 'z')]).astype(np.float32)


def cloud_message(coord, stamp, cost=None, frame='map'):
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    values = np.asarray(coord, dtype='<f4') if cost is None else np.column_stack((coord, cost)).astype('<f4',copy=False)
    names = ['x', 'y', 'z'] + ([] if cost is None else ['trav_cost'])
    return PointCloud2(header=Header(frame_id=frame, stamp=stamp), height=1, width=len(values),
        fields=[PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)],
        is_bigendian=False, point_step=values.shape[1]*4, row_step=values.nbytes,
        is_dense=True, data=values.tobytes())
