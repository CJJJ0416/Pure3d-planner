"""编译并调用正式DC单向A*，不包含其它搜索实现。"""
import ctypes
from pathlib import Path
import subprocess
import time

import numpy as np


class NativeAStar:
    def __init__(self, build_folder):
        build_folder=Path(build_folder)
        build_folder.mkdir(parents=True,exist_ok=True)
        library=build_folder/'libdc_astar.so'
        started=time.perf_counter()
        subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off','-shared','-fPIC',
            str(Path(__file__).with_suffix('.cpp')),'-o',str(library)],check=True)
        self.compile_s=time.perf_counter()-started
        self.library=ctypes.CDLL(str(library));self.function=self.library.dc_astar
        array=np.ctypeslib.ndpointer
        self.function.argtypes=[ctypes.c_uint32,array(dtype=np.uint64,ndim=1,flags='C_CONTIGUOUS'),
            array(dtype=np.uint32,ndim=1,flags='C_CONTIGUOUS'),array(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'),
            array(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'),ctypes.c_uint32,ctypes.c_uint32,ctypes.c_double,
            array(dtype=np.uint32,ndim=1,flags='C_CONTIGUOUS'),ctypes.POINTER(ctypes.c_uint32),
            array(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')]
        self.function.restype=ctypes.c_int
        self.trace_function=self.library.dc_astar_trace
        self.trace_function.argtypes=self.function.argtypes+[
            array(dtype=np.uint32,ndim=1,flags='C_CONTIGUOUS'),
            array(dtype=np.uint32,ndim=1,flags='C_CONTIGUOUS'),
            array(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'),
            array(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'),
            array(dtype=np.uint32,ndim=2,flags='C_CONTIGUOUS')]
        self.trace_function.restype=ctypes.c_int

    def search(self, graph, xyz, cost, endpoints, weight, trace=False):
        path=np.empty(len(xyz),dtype=np.uint32);size=ctypes.c_uint32();stats=np.empty(6,dtype=np.float64)
        started=time.perf_counter()
        args=[len(xyz),graph.indptr.astype(np.uint64),graph.indices.astype(np.uint32),
            np.ascontiguousarray(xyz,dtype=np.float64),np.ascontiguousarray(cost,dtype=np.float64),
            int(endpoints[0]),int(endpoints[1]),weight,path,ctypes.byref(size),stats]
        if trace:
            order=np.empty(len(xyz),dtype=np.uint32);parent=np.empty(len(xyz),dtype=np.uint32)
            g=np.empty(len(xyz),dtype=np.float64);priority=np.empty(len(xyz),dtype=np.float64)
            decisions=np.zeros((len(xyz),3),dtype=np.uint32)
            found=self.trace_function(*args,order,parent,g,priority,decisions)
            count=int(stats[2])
            self.last_trace=dict(expansion_order=order[:count].copy(),parent=parent,g=g,
                expansion_priority=priority[:count].copy(),decisions=decisions[:count].copy(),
                termination='goal_reached' if found else 'open_set_exhausted')
        else:
            found=self.function(*args)
        metrics=dict(found=bool(found),search_s=float(stats[5]),call_wall_s=time.perf_counter()-started,
            objective=float(stats[1]) if found else None,expanded_nodes=int(stats[2]),examined_edges=int(stats[3]),
            length_m=float(stats[4]) if found else None,path_points=int(size.value))
        return metrics,path[:size.value].copy()
