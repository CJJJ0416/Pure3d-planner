"""Conservative count filter on occupied world voxels. No geometry is generated."""
from common.paths import ROOT, NATIVE_DIR
import ctypes
from functools import lru_cache
import time
import numpy as np


@lru_cache(maxsize=1)
def native():
    from planner.light_planner.native import compile_cached
    root=ROOT
    path,initialization_s,cache_hit=compile_cached((NATIVE_DIR / 'voxel_filter.cpp'),root/'.cache/native/voxel_filter','libvoxel_filter')
    lib=ctypes.CDLL(str(path))
    f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    u64=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.entry_voxel_filter.argtypes=[f32,ctypes.c_uint64,ctypes.c_double,ctypes.c_uint64,u8,u64,u64,f64]
    lib.entry_voxel_filter.restype=ctypes.c_int
    lib.entry_voxel_filter_error.restype=ctypes.c_char_p
    return lib,dict(initialization_s=initialization_s,compiled_library_cache_hit=cache_hit,library=str(path))


def filter_points(xyz,voxel_m=.1,min_points=3):
    lib,initialization=native()
    start=time.perf_counter()
    xyz=np.ascontiguousarray(xyz,dtype=np.float32)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite N x 3 XYZ')
    if not np.isfinite(voxel_m) or voxel_m<=0 or not isinstance(min_points,(int,np.integer)) or min_points<1:
        raise ValueError('Expected positive voxel size and integer minimum')
    keep=np.empty(len(xyz),np.uint8);support=np.empty(len(xyz),np.uint64)
    cells=np.zeros(1,np.uint64);seconds=np.zeros(2,np.float64)
    if lib.entry_voxel_filter(xyz,len(xyz),voxel_m,min_points,keep,support,cells,seconds):
        raise ValueError(lib.entry_voxel_filter_error().decode())
    mask=keep.astype(bool)
    return mask,support,dict(voxel_m=voxel_m,min_points=min_points,neighborhood='3x3x3 cells, includes self',
        original_points=len(xyz),retained_points=int(mask.sum()),removed_points=int((~mask).sum()),
        occupied_voxels=int(cells[0]),hash_s=float(seconds[0]),count_s=float(seconds[1]),
        filter_s=time.perf_counter()-start,**initialization)
