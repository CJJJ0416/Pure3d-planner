"""Gravity-aligned local bounds maintained from LIO positions."""
import numpy as np


class LioBox:
    def __init__(self, size=(6., 6., 5.), resolution=.05, shift=.2):
        self.size = np.asarray(size, dtype=float)
        if self.size.shape != (3,) or not np.isfinite(self.size).all() or np.any(self.size <= 0):
            raise ValueError('Expected positive finite XYZ box dimensions')
        if not np.isfinite([resolution, shift]).all() or resolution <= 0 or shift <= 0:
            raise ValueError('Expected positive box resolution and shift')
        self.resolution = float(resolution)
        self.shift_cells = max(1, int(np.ceil(shift / resolution)))
        self.index = None
        self.lower = self.upper = None

    def update(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Expected finite LIO XYZ position')
        index = np.floor(position / self.resolution).astype(np.int64)
        if self.index is not None and np.max(np.abs(index - self.index)) < self.shift_cells:
            return False
        self.index = index
        center = index * self.resolution
        self.lower, self.upper = center - self.size / 2, center + self.size / 2
        return True


def box_marker(lower, upper, stamp, frame='world'):
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker
    lower, upper = np.asarray(lower), np.asarray(upper)
    corners = np.array([[x, y, z] for x in [0, 1] for y in [0, 1] for z in [0, 1]])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.abs(corners[a] - corners[b]).sum() == 1]
    xyz = lower + corners * (upper - lower)
    m = Marker()
    m.header.frame_id, m.header.stamp = frame, stamp
    m.ns, m.id, m.type, m.action = 'entry_lio_box', 0, Marker.LINE_LIST, Marker.ADD
    m.pose.orientation.w = 1.
    m.scale.x = .04
    m.color.g, m.color.b, m.color.a = .8, 1., 1.
    m.points = [Point(x=float(xyz[i, 0]), y=float(xyz[i, 1]), z=float(xyz[i, 2]))
                for edge in edges for i in edge]
    return m
