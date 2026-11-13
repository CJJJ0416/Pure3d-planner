"""占用体素场的Dual Contouring与表面法向，不读取外部Mesh。"""
import itertools

import numpy as np
from scipy import ndimage


def _solve_qef(q, b, mass):
    """从交点质心求最小位移的 QEF 解，平面零空间保留质心坐标。"""
    values, vectors = np.linalg.eigh(q)
    projected = np.einsum('nji,nj->ni', vectors, b - np.einsum('nij,nj->ni', q, mass))
    delta = np.zeros_like(projected)
    np.divide(projected, values, out=delta, where=values > values[:, -1:] * 1e-6)
    return mass + np.einsum('nij,nj->ni', vectors, delta)


def _bounded_qef(q, b, mass, resolution):
    """枚举盒约束的面、边和角，求每个活动单元内的最小 QEF。"""
    solution = _solve_qef(q, b, mass)
    outside = ((solution < 0) | (solution > resolution)).any(axis=1)
    qq, bb, mm = q[outside], b[outside], mass[outside]
    best = mm.copy()

    def objective(x):
        return np.einsum('ni,nij,nj->n', x, qq, x) - 2 * np.einsum('ni,ni->n', x, bb)

    best_value = objective(best)
    for state in itertools.product((-1, 0, 1), repeat=3):
        free = np.flatnonzero(np.asarray(state) == 0)
        fixed = np.flatnonzero(np.asarray(state) != 0)
        candidate = np.zeros_like(mm)
        candidate[:, fixed] = (np.asarray(state)[fixed] == 1) * resolution
        if len(free):
            rhs = bb[:, free] - np.einsum('nij,nj->ni', qq[:, free][:, :, fixed], candidate[:, fixed])
            candidate[:, free] = _solve_qef(qq[:, free][:, :, free], rhs, mm[:, free])
        feasible = ((candidate >= -1e-10) & (candidate <= resolution + 1e-10)).all(axis=1)
        value = objective(candidate)
        take = feasible & (value < best_value)
        best[take], best_value[take] = candidate[take], value[take]
    solution[outside] = best
    return solution, int(outside.sum())


def dual_contouring(field, origin, resolution):
    """真正的 Hermite/QEF Dual Contouring，每个活动单元一个顶点。

    Hermite 交点由场值线性插值，法向取同一 EDT 的单元三线性插值梯度。
    QEF 约束在当前单元内。相交网格边周围四个单元构成一个四边形，
    统一三角化后构图，搜索边始终属于输出三角网格。
    """
    negative = field < 0
    cell_shape = np.asarray(field.shape) - 1
    first = negative[:-1, :-1, :-1]
    active = np.zeros(tuple(cell_shape), dtype=bool)
    for offset in itertools.product((0, 1), repeat=3):
        slices = tuple(slice(o, o + size) for o, size in zip(offset, cell_shape))
        active |= negative[slices] != first
    cells = np.argwhere(active)
    count = len(cells)
    corner_offsets = np.asarray(list(itertools.product((0, 1), repeat=3)))
    corner_values = np.stack([field[tuple((cells + offset).T)] for offset in corner_offsets], axis=1)
    q = np.zeros((count, 3, 3), dtype=np.float64)
    b = np.zeros((count, 3), dtype=np.float64)
    mass = np.zeros((count, 3), dtype=np.float64)
    hits = np.zeros(count, dtype=np.int32)
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for pair in itertools.product((0, 1), repeat=2):
            offset = np.zeros(3, dtype=int)
            offset[other] = pair
            a = cells + offset
            z = a.copy()
            z[:, axis] += 1
            va, vz = field[tuple(a.T)], field[tuple(z.T)]
            cross = (va < 0) != (vz < 0)
            ids = np.flatnonzero(cross)
            weight = va[cross] / (va[cross] - vz[cross])
            p = np.broadcast_to(offset, (len(ids), 3)).astype(np.float64).copy()
            p[:, axis] += weight
            p *= resolution
            # 三线性单元场在相交边上的精确梯度。相交轴导数必为非零，
            # 不会因薄壁两侧中心差分相消而失去 Hermite 法向。
            n = np.zeros_like(p)
            n[:, axis] = (vz[cross] - va[cross]) / resolution
            for dimension in other:
                lo, hi = offset.copy(), offset.copy()
                lo[dimension], hi[dimension] = 0, 1
                lo_end, hi_end = lo.copy(), hi.copy()
                lo_end[axis], hi_end[axis] = 1, 1
                code = np.array([4, 2, 1])
                d0 = corner_values[ids, hi @ code] - corner_values[ids, lo @ code]
                d1 = corner_values[ids, hi_end @ code] - corner_values[ids, lo_end @ code]
                n[:, dimension] = ((1 - weight) * d0 + weight * d1) / resolution
            n /= np.linalg.norm(n, axis=1)[:, None]
            q[ids] += np.einsum('ni,nj->nij', n, n)
            b[ids] += n * np.einsum('ni,ni->n', n, p)[:, None]
            mass[ids] += p
            hits[ids] += 1
    local, bounded = _bounded_qef(q, b, mass / hits[:, None], resolution)
    vertices = origin + cells * resolution + local
    mapping = np.full(tuple(cell_shape), -1, dtype=np.int32)
    mapping[tuple(cells.T)] = np.arange(count, dtype=np.int32)
    quads = []
    offsets = [np.array([[0,-1,-1], [0,0,-1], [0,0,0], [0,-1,0]]),
               np.array([[-1,0,-1], [-1,0,0], [0,0,0], [0,0,-1]]),
               np.array([[-1,-1,0], [0,-1,0], [0,0,0], [-1,0,0]])]
    for axis in range(3):
        a, z = [slice(None)] * 3, [slice(None)] * 3
        a[axis], z[axis] = slice(None, -1), slice(1, None)
        edges = np.argwhere(negative[tuple(a)] != negative[tuple(z)])
        other = [i for i in range(3) if i != axis]
        edges = edges[((edges[:, other] > 0) & (edges[:, other] < cell_shape[other])).all(axis=1)]
        around = edges[:, None, :] + offsets[axis]
        quad = mapping[tuple(around.transpose(2, 0, 1))]
        reverse = ~negative[tuple(edges.T)]
        quad[reverse] = quad[reverse, ::-1]
        quads.append(quad)
    quads = np.concatenate(quads)
    triangles = np.concatenate((quads[:, [0,1,2]], quads[:, [0,2,3]])).astype(np.int32)
    return vertices, triangles, dict(
        implementation='uniform-grid Hermite Dual Contouring, box-constrained QEF',
        normals='analytic gradient of the common EDT trilinear cell interpolant at edge intersections',
        active_cells=count, quads=len(quads), bounded_qef_cells=bounded,
        triangulation='consistent 0-2 quad diagonal, graph uses triangle edges only')


def outward_vertex_normals(vertices, triangles, field, origin, resolution):
    """面积加权网格法向，用负内正外的体素场校正方向，不使用重力。

    在顶点两侧各探测四分之一个体素。两侧场值相同的接触位置保留
    原提取网格绕序给出的方向，并在返回统计中明确记录。
    """
    points = vertices[triangles]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals = np.column_stack([
        sum(np.bincount(triangles[:, corner], weights=face_normals[:, axis], minlength=len(vertices))
            for corner in range(3)) for axis in range(3)])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    epsilon = resolution / 4
    plus = ndimage.map_coordinates(field, ((vertices + epsilon * normals - origin) / resolution).T,
                                   order=1, prefilter=False)
    minus = ndimage.map_coordinates(field, ((vertices - epsilon * normals - origin) / resolution).T,
                                    order=1, prefilter=False)
    difference = plus - minus
    flip = difference < -1e-10
    normals[flip] *= -1
    return normals, dict(
        algorithm='area-weighted triangle vertex normals, oriented by signed field probes',
        gravity_used=False, probe_distance_m=epsilon,
        flipped_vertices=int(flip.sum()),
        equal_probe_vertices=int((np.abs(difference) <= 1e-10).sum()),
        equal_probe_rule='retain the orientation of the extracted mesh winding')
