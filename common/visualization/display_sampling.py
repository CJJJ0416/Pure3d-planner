"""Bounded-memory voxel representatives for display only; original rows retained."""
import numpy as np


def display_rows(xyz,cost=None,voxel_m=.1,max_points=200000,chunk_size=262144):
    if len(xyz)>=65536:
        if voxel_m<=0 or max_points<=0 or chunk_size<=0:raise ValueError('Display sampling settings must be positive')
        from common.native.arrays import native_display_rows
        return native_display_rows(xyz,cost,voxel_m,max_points)
    return numpy_display_rows(xyz,cost,voxel_m,max_points,chunk_size)


def numpy_display_rows(xyz,cost=None,voxel_m=.1,max_points=200000,chunk_size=262144):
    xyz=np.asarray(xyz)
    if voxel_m<=0 or max_points<=0 or chunk_size<=0:raise ValueError('Display sampling settings must be positive')
    if xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('Expected XYZ')
    if cost is not None and np.shape(cost)!=(len(xyz),):raise ValueError('Cost count differs from XYZ')
    if not len(xyz):return np.empty(0,np.int64),0
    lower=np.floor(xyz.min(0).astype(np.float64)/voxel_m).astype(np.int64)
    upper=np.floor(xyz.max(0).astype(np.float64)/voxel_m).astype(np.int64)
    shape=upper-lower+1
    if np.prod(shape.astype(object))>np.iinfo(np.int64).max:raise ValueError('Display grid exceeds int64 range')
    ids=np.empty(0,np.int64);rows=np.empty(0,np.int64)
    for first in range(0,len(xyz),chunk_size):
        last=min(first+chunk_size,len(xyz))
        grid=np.floor(xyz[first:last].astype(np.float64)/voxel_m).astype(np.int64)-lower
        keys=(grid[:,0]*shape[1]+grid[:,1])*shape[2]+grid[:,2]
        if cost is None:
            new_ids,index=np.unique(keys,return_index=True)
        else:
            # The highest-cost original point represents each voxel. No cost
            # averaging that could wash a red obstacle into surrounding green.
            order=np.lexsort((np.arange(last-first),-np.asarray(cost[first:last]),keys))
            ordered=keys[order];starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
            new_ids=ordered[starts];index=order[starts]
        new_rows=index+first
        if not len(ids):ids,rows=new_ids,new_rows;continue
        position=np.searchsorted(ids,new_ids);inside=position<len(ids)
        existing=np.zeros(len(new_ids),bool);existing[inside]=ids[position[inside]]==new_ids[inside]
        if cost is not None:
            update=np.flatnonzero(existing)
            update=update[np.asarray(cost)[new_rows[update]]>np.asarray(cost)[rows[position[update]]]]
            rows[position[update]]=new_rows[update]
        added=~existing
        if added.any():
            merged_ids=np.r_[ids,new_ids[added]];merged_rows=np.r_[rows,new_rows[added]]
            order=np.argsort(merged_ids);ids,rows=merged_ids[order],merged_rows[order]
    voxel_points=len(rows)
    if len(rows)>max_points:rows=rows[np.linspace(0,len(rows)-1,max_points,dtype=np.int64)]
    return rows,voxel_points
