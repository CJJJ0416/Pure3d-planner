"""ROS/CUDA-independent exact native array indexing; no model changes."""
from common.paths import ROOT, NATIVE_DIR
import ctypes
import threading
import numpy as np


_library=None
_lock=threading.Lock()


def library():
    global _library
    with _lock:
        if _library is None:
            from planner.light_planner.native import compile_cached
            path,_,_=compile_cached((NATIVE_DIR / 'fast_arrays.cpp'),
                ROOT/'.cache/native/array_requests','libentry_arrays')
            lib=ctypes.CDLL(str(path))
            ptr=ctypes.c_void_p
            lib.fast_arrays.argtypes=[ptr,ctypes.c_int,ctypes.c_uint64,ptr,ctypes.c_int,ctypes.c_uint64,ctypes.c_uint64,ptr,ctypes.c_int]
            lib.fast_arrays.restype=ptr
            lib.fast_arrays_error.restype=ctypes.c_char_p
            lib.fast_arrays_sizes.argtypes=[ptr,ptr]
            lib.fast_arrays_copy.argtypes=[ptr,ptr,ptr,ptr]
            lib.fast_arrays_free.argtypes=[ptr]
            _library=lib
    return _library


def scan(xyz,size,cost=None,partition=False):
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('Expected XYZ')
    if xyz.dtype not in (np.dtype('float32'),np.dtype('float64')):xyz=xyz.astype(np.float64)
    if xyz.strides[1]!=xyz.dtype.itemsize or xyz.strides[0]<3*xyz.dtype.itemsize or xyz.strides[0]%xyz.dtype.itemsize:
        xyz=np.ascontiguousarray(xyz)
    size=np.ascontiguousarray(size,dtype=np.float64)
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError('Invalid grid size')
    if cost is not None:
        cost=np.asarray(cost)
        if cost.shape!=(len(xyz),):raise ValueError('Cost count differs from XYZ')
        if cost.dtype not in (np.dtype('float32'),np.dtype('float64')):cost=cost.astype(np.float64)
        if cost.strides[0]<=0 or cost.strides[0]%cost.dtype.itemsize:cost=np.ascontiguousarray(cost)
    lib=library();handle=lib.fast_arrays(xyz.ctypes.data,xyz.dtype.itemsize,xyz.strides[0]//xyz.dtype.itemsize,
        None if cost is None else cost.ctypes.data,0 if cost is None else cost.dtype.itemsize,
        0 if cost is None else cost.strides[0]//cost.dtype.itemsize,
        len(xyz),size.ctypes.data,int(partition))
    if not handle:raise ValueError(lib.fast_arrays_error().decode())
    try:
        counts=np.zeros(2,np.uint64);lib.fast_arrays_sizes(handle,counts.ctypes.data)
        groups,nrows=map(int,counts);rows=np.empty(nrows,np.int64)
        keys=np.empty((groups,3),np.int64) if partition else None
        offsets=np.empty(groups+1,np.uint64) if partition else None
        lib.fast_arrays_copy(handle,rows.ctypes.data,None if keys is None else keys.ctypes.data,
            None if offsets is None else offsets.ctypes.data)
        return rows,groups,keys,offsets
    finally:lib.fast_arrays_free(handle)


def native_partition(xyz,size):
    rows,_,keys,offsets=scan(xyz,size,partition=True)
    return [(tuple(map(int,key)),rows[int(offsets[i]):int(offsets[i+1])]) for i,key in enumerate(keys)]


def native_display_rows(xyz,cost=None,voxel_m=.1,max_points=200000):
    if not np.isfinite(voxel_m) or voxel_m<=0 or max_points<=0:raise ValueError('Invalid display sampling settings')
    rows,count,_,_=scan(xyz,[voxel_m]*3,cost=cost)
    if len(rows)>max_points:rows=rows[np.linspace(0,len(rows)-1,max_points,dtype=np.int64)]
    return rows,count
