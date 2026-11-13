"""DC三角边、代价剪枝与机器人条件化跨阶连边。"""
import numpy as np
from scipy import sparse


def surface_graph(vertices, triangles, cost, threshold):
    """cost 剪枝后，仅保留原三角网格边，绝不跨空隙建立半径连边。"""
    retained = np.flatnonzero(cost <= threshold)
    mapping = np.full(len(vertices), -1, dtype=np.int32)
    mapping[retained] = np.arange(len(retained), dtype=np.int32)
    edges = np.concatenate((triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]))
    edges = mapping[edges]
    edges = edges[(edges >= 0).all(axis=1)]
    edges = np.concatenate((edges, edges[:, ::-1]))
    graph = sparse.coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                              shape=(len(retained),) * 2).tocsr()
    graph.sum_duplicates()
    return (np.ascontiguousarray(vertices[retained], dtype=np.float64),
            np.ascontiguousarray(cost[retained], dtype=np.float64),
            graph.indptr.astype(np.uint64), graph.indices.astype(np.uint32), retained, graph)


def connect_step_surfaces(vertices, normals, retained, graph, step_max, slope_max,
                          horizontal_radius=.3, gravity_up=(0., 0., 1.), minimum_height=.05,
                          support_mask=None):
    """给已按 cost 剪枝的表面图补入机器人允许的跨阶连接。

    三角形边只描述同一表面。踏面之间即使高差可跨越，也可能要绕经高代价
    立面才能连接。这里连接朝上的两端踏面，保持原节点 cost 和搜索边权。
    support_mask仅限定新增边的来源支撑面，原三角网格节点和边不变。
    不恢复26邻域或任意三维半径邻接。
    """
    from scipy.spatial import cKDTree

    up=np.asarray(gravity_up,dtype=float);up/=np.linalg.norm(up)
    allowed=normals[retained]@up >= np.cos(slope_max)
    if support_mask is not None:
        allowed &= support_mask[retained]
    eligible=np.flatnonzero(allowed)
    points=vertices[retained[eligible]]
    pairs=cKDTree(points).query_pairs(np.hypot(horizontal_radius,step_max),output_type='ndarray')
    delta=points[pairs[:,1]]-points[pairs[:,0]]
    height=delta@up
    horizontal=delta-height[:,None]*up
    take=(np.abs(height)>minimum_height)&(np.abs(height)<=step_max)&(np.linalg.norm(horizontal,axis=1)<=horizontal_radius)
    pairs=eligible[pairs[take]]
    edges=np.concatenate((pairs,pairs[:,::-1]))
    additions=sparse.csr_matrix((np.ones(len(edges),dtype=np.uint8),(edges[:,0],edges[:,1])),shape=graph.shape)
    linked=(graph+additions).tocsr()
    return linked,dict(eligible_surface_nodes=len(eligible),step_connections=len(pairs),
        original_directed_edges=graph.nnz,linked_directed_edges=linked.nnz,
        step_max_m=step_max,slope_max_rad=slope_max,horizontal_radius_m=horizontal_radius,
        minimum_height_difference_m=minimum_height,gravity_up=up.tolist())
