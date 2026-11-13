"""原点固定的均值体素与实测占用壳几何场。"""
import time

import numpy as np
from scipy import ndimage


def voxel_mean(xyz, cost, resolution=0.1):
    """每个占用体素输出一个质心、平均 cost、整数索引和原始点数。"""
    grid = np.floor(np.asarray(xyz, dtype=np.float64) / resolution).astype(np.int64)
    lower = grid.min(axis=0)
    shape = grid.max(axis=0) - lower + 1
    shifted = grid - lower
    keys = (shifted[:, 0] * shape[1] + shifted[:, 1]) * shape[2] + shifted[:, 2]
    order = np.argsort(keys)
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    coord = np.add.reduceat(np.asarray(xyz, dtype=np.float64)[order], starts, axis=0) / counts[:, None]
    values = np.add.reduceat(np.asarray(cost, dtype=np.float64)[order], starts) / counts
    return dict(xyz=coord.astype(np.float32), cost=values.astype(np.float32),
                ijk=grid[order[starts]].astype(np.int32), counts=counts.astype(np.uint32))


def observed_voxel_field(xyz, resolution=.1):
    """观测占用壳的有符号EDT，用于提取其边界。

    负值仅表示含实测点的占用体素。正值是这个离散集合的外部，
    不代表由射线确认的空闲，也不推断建筑实体内部或补全未知空间。
    输入可以是原点固定、同分辨率体素的质心，不改变其占用索引。
    """
    began=time.perf_counter()
    grid=np.floor(np.asarray(xyz,dtype=float)/resolution).astype(np.int32)
    lower=grid.min(0)-3;upper=grid.max(0)+3;shape=upper-lower+1
    occupied=np.zeros(tuple(shape),dtype=bool)
    occupied[tuple((grid-lower).T)]=True
    outside=ndimage.distance_transform_edt(~occupied).astype(np.float32)
    inside=ndimage.distance_transform_edt(occupied).astype(np.float32)
    field=(outside-inside)*resolution
    origin=(lower+.5)*resolution
    metadata=dict(source='observed PCD voxel membership only',resolution_m=resolution,
        shape=shape.tolist(),sample_origin_xyz=origin.tolist(),observed_voxels=int(occupied.sum()),
        cavity_filling=False,dilation=False,mesh_read=False,
        meaning='Signed distance to the observed occupied-voxel set, not a visibility TSDF or true building solid SDF.',
        field_prepare_s=time.perf_counter()-began)
    return field,origin,metadata


def support_clearance(xyz, field, origin, resolution, gravity_up, interval_free):
    """沿反重力射线遍历已有占用体素，排除净空不足的支撑来源。

    跳过来源自身的占用体素。只按实测点的占用判定阻挡，未观测体素
    不产生阻挡。DDA在地图坐标中沿相对方向前进，不使用地面Z零点。
    """
    points=np.asarray(xyz,dtype=float)
    up=np.asarray(gravity_up,dtype=float);up=up/np.linalg.norm(up)
    grid=np.floor(points/resolution).astype(np.int32)
    lower=np.rint(origin/resolution-.5).astype(np.int32)
    occupied=field<0;shape=np.asarray(field.shape)
    cell=grid.copy();direction=np.sign(up).astype(np.int32)
    boundary=(grid+(up>0))*resolution
    next_t=np.full(points.shape,np.inf)
    np.divide(boundary-points,up,out=next_t,where=up!=0)
    stride=np.full(3,np.inf)
    np.divide(resolution,np.abs(up),out=stride,where=up!=0)
    clear=np.ones(len(points),dtype=bool);first_hit=np.full(len(points),np.inf)
    rows=np.arange(len(points))
    crossings=int(np.ceil(interval_free/resolution*np.abs(up).sum()))+3
    for _ in range(crossings):
        axis=np.argmin(next_t,axis=1);distance=next_t[rows,axis].copy()
        cell[rows,axis]+=direction[axis];next_t[rows,axis]+=stride[axis]
        local=cell-lower
        active=(distance<interval_free)&clear&((local>=0)&(local<shape)).all(1)
        ids=np.flatnonzero(active);ids=ids[occupied[tuple(local[ids].T)]]
        first_hit[ids]=distance[ids];clear[ids]=False
    return clear,first_hit
