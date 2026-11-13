"""Exact triangle/cell labels and first-hit scans, on the same Warp mesh.

The clipping kernel labels surface intersection, not solid occupancy/free space.
No PCD nearest-neighbour threshold is used to construct geometry supervision.
"""
import numpy as np
import warp as wp

Polygon = wp.types.matrix(shape=(12, 3), dtype=wp.float32)


@wp.func
def vertex(p: Polygon, i: int):
    return wp.vec3(p[i, 0], p[i, 1], p[i, 2])


@wp.func
def clipped_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, center: wp.vec3, half: float):
    poly = Polygon()
    for j in range(3):
        poly[0, j] = a[j]
        poly[1, j] = b[j]
        poly[2, j] = c[j]
    count = int(3)
    for plane in range(6):
        axis = plane // 2
        sign = float(1.0)
        if plane % 2 == 1:
            sign = -1.0
        out = Polygon()
        nout = int(0)
        for i in range(count):
            u = vertex(poly, i)
            v = vertex(poly, (i + 1) % count)
            du = sign * (u[axis] - center[axis]) - half
            dv = sign * (v[axis] - center[axis]) - half
            if du <= 0.0:
                for j in range(3):
                    out[nout, j] = u[j]
                nout += 1
            if (du < 0.0 and dv > 0.0) or (du > 0.0 and dv < 0.0):
                p = u + (v - u) * (du / (du - dv))
                for j in range(3):
                    out[nout, j] = p[j]
                nout += 1
        poly = out
        count = nout
    best = center
    distance = float(1.e20)
    if count > 0:
        normal = wp.normalize(wp.cross(b - a, c - a))
        projection = center - normal * wp.dot(center - a, normal)
        inside = bool(True)
        for i in range(count):
            u = vertex(poly, i)
            v = vertex(poly, (i + 1) % count)
            edge = v - u
            if wp.dot(wp.cross(edge, projection - u), normal) < -1.e-10:
                inside = False
            alpha = wp.clamp(wp.dot(center - u, edge) / wp.max(wp.dot(edge, edge), 1.e-20), 0.0, 1.0)
            p = u + alpha * edge
            d = wp.length_sq(p - center)
            if d < distance:
                distance = d
                best = p
        if inside and count >= 3:
            best = projection
        return wp.vec4(best[0], best[1], best[2], 1.0)
    return wp.vec4(0.0)


@wp.kernel
def cell_kernel(mesh: wp.uint64, centers: wp.array(dtype=wp.vec3), half: float,
                labels: wp.array(dtype=int), points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    q = centers[i]
    query = wp.mesh_query_aabb(mesh, q - wp.vec3(half), q + wp.vec3(half))
    face = int(0)
    closest = float(1.e20)
    while wp.mesh_query_aabb_next(query, face):
        a = wp.mesh_eval_position(mesh, face, 1.0, 0.0)
        b = wp.mesh_eval_position(mesh, face, 0.0, 1.0)
        c = wp.mesh_eval_position(mesh, face, 0.0, 0.0)
        result = clipped_point(a, b, c, q, half)
        if result[3] > 0.0:
            p = wp.vec3(result[0], result[1], result[2])
            d = wp.length_sq(p - q)
            if d < closest:
                closest = d
                labels[i] = 1
                points[i] = p


@wp.kernel
def ray_kernel(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
               max_range: float, distance: wp.array(dtype=float), face: wp.array(dtype=int)):
    i = wp.tid()
    hit = wp.mesh_query_ray(mesh, origins[i], directions[i], max_range)
    face[i] = -1
    if hit.result:
        distance[i] = hit.t
        face[i] = hit.face


@wp.kernel
def distance_kernel(mesh: wp.uint64, xyz: wp.array(dtype=wp.vec3), output: wp.array(dtype=float)):
    i = wp.tid()
    query = wp.mesh_query_point(mesh, xyz[i], 1000.0)
    p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    output[i] = wp.length(xyz[i] - p)


class MeshQueries:
    def __init__(self, mesh, device='cuda:0'):
        self.device = device
        self.vertices = wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device=device)
        self.faces = wp.array(np.asarray(mesh.faces, np.int32).ravel(), dtype=int, device=device)
        self.mesh = wp.Mesh(self.vertices, self.faces)

    def cells(self, grid, size):
        centers = (np.asarray(grid, np.float64) + .5) * size
        labels = wp.zeros(len(grid), dtype=int, device=self.device)
        points = wp.zeros(len(grid), dtype=wp.vec3, device=self.device)
        wp.launch(cell_kernel, len(grid), [self.mesh.id, wp.array(centers, dtype=wp.vec3, device=self.device),
                  float(size / 2), labels, points], device=self.device)
        return labels.numpy().astype(bool), points.numpy()

    def rays(self, origins, directions, max_range=30.):
        distances = wp.zeros(len(origins), dtype=float, device=self.device)
        faces = wp.empty(len(origins), dtype=int, device=self.device)
        wp.launch(ray_kernel, len(origins), [self.mesh.id, wp.array(origins, dtype=wp.vec3, device=self.device),
                  wp.array(directions, dtype=wp.vec3, device=self.device), max_range, distances, faces], device=self.device)
        return distances.numpy(), faces.numpy()

    def distances(self, xyz):
        out = wp.empty(len(xyz), dtype=float, device=self.device)
        if len(xyz):
            wp.launch(distance_kernel, len(xyz), [self.mesh.id, wp.array(xyz, dtype=wp.vec3, device=self.device), out], device=self.device)
        return out.numpy()


CHILDREN = np.array(list(__import__('itertools').product(range(2), repeat=3)), np.int64)


def grid_between(lower, upper, spacing):
    lo = np.floor(np.asarray(lower) / spacing).astype(np.int64)
    hi = np.ceil(np.asarray(upper) / spacing).astype(np.int64)
    return np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)), indexing='ij'), -1).reshape(-1, 3)


def surface_targets(query, lower, upper, levels):
    """All intersecting leaves, including completely unobserved target surfaces."""
    grid = grid_between(lower, upper, levels[0])
    result = {}
    for level, size in enumerate(levels):
        label, point = query.cells(grid, size)
        result[f'grid_{level}'] = grid.astype(np.int32)
        result[f'label_{level}'] = label
        result[f'point_{level}'] = point
        grid = (grid[label, None] * 2 + CHILDREN).reshape(-1, 3)
    return result
