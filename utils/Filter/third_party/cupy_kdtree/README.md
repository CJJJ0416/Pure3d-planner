# CuPy GPU KDTree kernels

`_kdtree_utils.py` is copied from CuPy v14.0.1:
https://github.com/cupy/cupy/blob/v14.0.1/cupyx/scipy/spatial/_kdtree_utils.py

The upstream MIT license is retained in LICENSE. This module uses the upstream
GPU index construction and exact KNN kernels with CuPy 13.6, preserving the
workspace's NumPy 1.26 environment. CuPy 14 itself requires NumPy 2. The only
source adaptation is omitting the two `cupy/float16.cuh` includes, because CuPy
13 already defines half in carray.cuh and does not ship that separate header.

The upstream `find_nodes_in_radius` API allocates a queries-by-points index
matrix even for count-only requests. Filter does not call that API. Its two
radius counts use the bounded-memory CUDA implementation in data/gpu_knn.py.
