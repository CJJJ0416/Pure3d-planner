"""CUDA representative sampling and exact nearest representatives through a Triton hash."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _slot(key, mask: tl.constexpr):
    value = key.to(tl.uint64)
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ^ (value >> 27)) * 0x94d049bb133111eb
    return ((value ^ (value >> 31)) & mask).to(tl.int64)


@tr.jit
def _insert(Keys, Table, Rows, N: tl.constexpr, Mask: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    active = row < N
    key = tl.load(Keys + row, active, other=-1)
    slot = _slot(key, Mask)
    while tl.sum(active.to(tl.int32), 0) > 0:
        previous = tl.atomic_cas(Table + slot, tl.full((B,), -1, tl.int64), key, sem="relaxed")
        placed = active & ((previous == -1) | (previous == key))
        tl.store(Rows + slot, row, placed)
        active = active & ~placed
        slot = (slot + active.to(tl.int64)) & Mask


@tr.jit
def _nearest(Points, Original, Queries, Cells, Table, Rows, Result, Distance, Radius,
             X0: tl.constexpr, Y0: tl.constexpr, Z0: tl.constexpr,
             NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
             H: tl.constexpr, Mask: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0)
    qx = tl.load(Queries + q * 3).to(tl.float64)
    qy = tl.load(Queries + q * 3 + 1).to(tl.float64)
    qz = tl.load(Queries + q * 3 + 2).to(tl.float64)
    cx = tl.load(Cells + q * 3)
    cy = tl.load(Cells + q * 3 + 1)
    cz = tl.load(Cells + q * 3 + 2)
    lane = tl.arange(0, B)
    best = tl.full((), float('inf'), tl.float64)
    best_original = tl.full((), 0x7fffffffffffffff, tl.int64)
    best_row = tl.full((), -1, tl.int64)
    radius = 1
    finished = False
    while ~finished:
        width = 2 * radius + 1
        offset = 0
        while offset < width * width * width:
            i = offset + lane
            dx = i // (width * width) - radius
            dy = (i // width) % width - radius
            dz = i % width - radius
            shell = (radius == 1) | (tl.maximum(tl.abs(dx), tl.maximum(tl.abs(dy), tl.abs(dz))) == radius)
            x, y, z = cx + dx - X0, cy + dy - Y0, cz + dz - Z0
            active = (i < width * width * width) & shell
            active = active & (x >= 0) & (x < NX) & (y >= 0) & (y < NY) & (z >= 0) & (z < NZ)
            key = (x * NY + y) * NZ + z
            slot = _slot(key, Mask)
            found = tl.full((B,), False, tl.int1)
            pending = active
            while tl.sum(pending.to(tl.int32), 0) > 0:
                stored = tl.load(Table + slot, pending, other=-1)
                equal = pending & (stored == key)
                found = found | equal
                pending = pending & (stored != -1) & ~equal
                slot = (slot + pending.to(tl.int64)) & Mask
            row = tl.load(Rows + slot, found, other=0)
            px = tl.load(Points + row * 3, found, other=0).to(tl.float64)
            py = tl.load(Points + row * 3 + 1, found, other=0).to(tl.float64)
            pz = tl.load(Points + row * 3 + 2, found, other=0).to(tl.float64)
            distance = (px-qx)*(px-qx) + (py-qy)*(py-qy) + (pz-qz)*(pz-qz)
            distance = tl.where(found, distance, float('inf'))
            original = tl.load(Original + row, found, other=0x7fffffffffffffff)
            minimum = tl.min(distance, 0)
            original_min = tl.min(tl.where(found & (distance == minimum), original, 0x7fffffffffffffff), 0)
            row_min = tl.min(tl.where(found & (distance == minimum) & (original == original_min), row, 0x7fffffffffffffff), 0)
            replace = (minimum < best) | ((minimum == best) & (original_min < best_original))
            best_row = tl.where(replace, row_min, best_row)
            best_original = tl.where(replace, original_min, best_original)
            best = tl.minimum(best, minimum)
            offset += B
        # Every unvisited cell is outside this cube. Strict inequality also resolves ties.
        lower = tl.minimum(tl.minimum(qx-(cx-radius)*H, (cx+radius+1)*H-qx),
                           tl.minimum(tl.minimum(qy-(cy-radius)*H, (cy+radius+1)*H-qy),
                                      tl.minimum(qz-(cz-radius)*H, (cz+radius+1)*H-qz)))
        lower = tl.maximum(lower - 1.e-9, 0.)
        finished = best < lower * lower
        radius += (~finished).to(tl.int32)
    tl.store(Result + q, best_row)
    tl.store(Distance + q, tl.sqrt(best))
    tl.store(Radius + q, radius)


def representatives(points, voxel_size=.05):
    """Lowest original row per world-aligned voxel, with no centroid displacement."""
    cells = torch.floor(points.double() / voxel_size).long()
    low = cells.amin(0)
    shape = cells.amax(0) - low + 1
    relative = cells - low
    keys = (relative[:, 0] * shape[1] + relative[:, 1]) * shape[2] + relative[:, 2]
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    first = torch.cat((torch.ones(1, dtype=torch.bool, device=points.device), sorted_keys[1:] != sorted_keys[:-1]))
    index = order[first]
    return points[index], index, sorted_keys[first], low.tolist(), shape.tolist()


class VoxelHash:
    def __init__(self, points, original, keys, low, shape, voxel_size=.05):
        self.points, self.original = points, original
        self.low, self.shape, self.voxel_size = low, shape, voxel_size
        self.capacity = tr.next_power_of_2(2 * len(points))
        self.table = torch.full((self.capacity,), -1, dtype=torch.int64, device=points.device)
        self.rows = torch.empty_like(self.table)
        _insert[(tr.cdiv(len(points), 256),)](keys, self.table, self.rows, len(points), self.capacity-1, 256)

    def query(self, queries):
        cells = torch.floor(queries.double() / self.voxel_size).long()
        result = torch.empty(len(queries), dtype=torch.int64, device=queries.device)
        distance = torch.empty(len(queries), dtype=torch.float32, device=queries.device)
        radius = torch.empty(len(queries), dtype=torch.int32, device=queries.device)
        _nearest[(len(queries),)](self.points, self.original, queries, cells, self.table, self.rows,
            result, distance, radius, *self.low, *self.shape, self.voxel_size, self.capacity-1, 128,
            num_warps=4, enable_fp_fusion=False)
        return result, distance, radius


def verify_nearest(points, original, queries, result, samples=128):
    """One real-data audit against exhaustive GPU distances, outside the training path."""
    chosen = torch.linspace(0, len(queries)-1, min(samples, len(queries)), device=points.device).long()
    q = queries[chosen].double()
    best = torch.full((len(chosen),), float('inf'), dtype=torch.float64, device=points.device)
    row = torch.full((len(chosen),), -1, dtype=torch.int64, device=points.device)
    origin = torch.full_like(row, torch.iinfo(torch.int64).max)
    for first in range(0, len(points), 8192):
        delta = q[:, None] - points[None, first:first+8192].double()
        d = (delta * delta).sum(-1)
        minimum = d.min(1).values
        ids = original[first:first+8192]
        tie = torch.where(d == minimum[:, None], ids[None], torch.iinfo(torch.int64).max)
        local_origin, local_row = tie.min(1)
        take = (minimum < best) | ((minimum == best) & (local_origin < origin))
        best = torch.minimum(best, minimum)
        row = torch.where(take, first + local_row, row)
        origin = torch.where(take, local_origin, origin)
    equal = torch.equal(row, result[chosen])
    assert equal, 'GPU hash correspondence differs from exhaustive nearest representatives.'
    return dict(samples=len(chosen), exhaustive_nearest_identical=equal,
                query_index=chosen.cpu().tolist(), original_point_index=origin.cpu().tolist())
