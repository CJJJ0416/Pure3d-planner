"""坐标变换与独立原始三角网格高度检查；仅依赖 NumPy。"""
import numpy as np


def rotation_rpy(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

def transform(position, rpy):
    result = np.eye(4)
    result[:3, :3] = rotation_rpy(rpy)
    result[:3, 3] = position
    return result

def sensor_extrinsics(config):
    return transform(config["position_body_m"], np.deg2rad(config["rpy_body_deg"]))

def mesh_surface_residual(points, triangles):
    """到同 XY 原始向上三角面的最近高度差；不读取 VST 或路径高度。"""
    residual = np.full(len(points), np.inf)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for tri in triangles[normals[:, 2] > 1e-9]:
        a, b, c = tri
        edge1, edge2 = b[:2] - a[:2], c[:2] - a[:2]
        determinant = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        delta = points[:, :2] - a[:2]
        u = (delta[:, 0] * edge2[1] - delta[:, 1] * edge2[0]) / determinant
        v = (edge1[0] * delta[:, 1] - edge1[1] * delta[:, 0]) / determinant
        inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        z = a[2] + u[inside] * (b[2] - a[2]) + v[inside] * (c[2] - a[2])
        residual[inside] = np.minimum(residual[inside], np.abs(points[inside, 2] - z))
    return residual
