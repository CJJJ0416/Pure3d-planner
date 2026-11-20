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


def step_surface_pairs(points, step_max, horizontal_radius, gravity_up, minimum_height, xy_tree=None):
    """Enumerate bounded height-changing pairs before expensive support checks."""
    from scipy.spatial import cKDTree

    up=np.asarray(gravity_up,dtype=float)
    up=up/np.linalg.norm(up)
    tree=cKDTree(points) if xy_tree is None else xy_tree
    radius=np.hypot(horizontal_radius,step_max) if xy_tree is None else horizontal_radius
    pairs=tree.query_pairs(radius,output_type='ndarray')
    delta=points[pairs[:,1]]-points[pairs[:,0]]
    height=delta@up
    horizontal=delta-height[:,None]*up
    keep=(np.abs(height)>minimum_height)&(np.abs(height)<=step_max)
    keep &= np.linalg.norm(horizontal,axis=1)<=horizontal_radius
    return pairs[keep]


def connect_step_surfaces(vertices, normals, retained, graph, step_max, slope_max,
                          horizontal_radius=.3, gravity_up=(0., 0., 1.), minimum_height=.05,
                          support_mask=None, candidate_pairs=None, gateway_spacing=None):
    """Connect checked tread pairs, optionally retaining sparse spatial gateways.

    The default preserves DC's complete set of valid step edges. Sparse gateways
    keep at least one edge for every connected-component pair represented by
    valid candidates, preserving reachability while reducing parallel links.
    """
    from scipy.sparse.csgraph import connected_components

    up=np.asarray(gravity_up,dtype=float)
    up=up/np.linalg.norm(up)
    allowed=normals[retained]@up>=np.cos(slope_max)
    if support_mask is not None:
        allowed &= support_mask[retained]
    eligible=np.flatnonzero(allowed)
    if candidate_pairs is None:
        pairs=step_surface_pairs(vertices[retained[eligible]],step_max,horizontal_radius,
                                 up,minimum_height)
        pairs=eligible[pairs]
    else:
        pairs=np.asarray(candidate_pairs,dtype=np.int64).reshape(-1,2)
        pairs=pairs[allowed[pairs].all(axis=1)]
    checked_count=len(pairs)
    if gateway_spacing is not None and len(pairs):
        if not np.isfinite(gateway_spacing) or gateway_spacing<=0:
            raise ValueError('Step gateway spacing must be positive')
        _,labels=connected_components(graph,directed=False)
        pairs=pairs[labels[pairs[:,0]]!=labels[pairs[:,1]]]
        if len(pairs):
            components=np.sort(labels[pairs],axis=1)
            midpoint=vertices[retained[pairs]].mean(axis=1)
            axis=np.eye(3)[np.argmin(np.abs(up))]
            horizontal=axis-up*np.dot(axis,up)
            horizontal/=np.linalg.norm(horizontal)
            frame=np.column_stack((horizontal,np.cross(up,horizontal)))
            cells=np.floor((midpoint@frame)/gateway_spacing).astype(np.int64)
            lengths=np.sum(np.diff(vertices[retained[pairs]],axis=1)[:,0]**2,axis=1)
            order=np.lexsort((pairs[:,1],pairs[:,0],lengths,cells[:,1],cells[:,0],
                              components[:,1],components[:,0]))
            keys=np.column_stack((components,cells))[order]
            first=np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)]
            pairs=pairs[order[first]]
    edges=np.concatenate((pairs,pairs[:,::-1]))
    additions=sparse.csr_matrix((np.ones(len(edges),dtype=np.uint8),(edges[:,0],edges[:,1])),shape=graph.shape)
    linked=(graph+additions).tocsr()
    return linked,dict(eligible_surface_nodes=len(eligible),step_connections=len(pairs),
        checked_step_pairs=checked_count,gateway_spacing_m=gateway_spacing,
        original_directed_edges=graph.nnz,linked_directed_edges=linked.nnz,
        step_max_m=step_max,slope_max_rad=slope_max,horizontal_radius_m=horizontal_radius,
        minimum_height_difference_m=minimum_height,gravity_up=up.tolist())


def sparse_step_support(points, full_points, frame, xy_resolution, z_resolution, query_indices=None, xy_tree=None):
    """Estimate boundary tread normals and query overhead with sorted column keys."""
    from scipy.spatial import cKDTree

    local=np.asarray(points)@frame
    full=np.asarray(full_points)@frame
    selected=np.arange(len(local)) if query_indices is None else np.asarray(query_indices,dtype=np.int64)
    normals=np.zeros_like(local)
    supported=np.zeros(len(local),dtype=bool)
    overhead=np.full(len(local),np.inf)
    if not len(selected):
        return normals,supported,overhead
    query=local[selected]
    tree=cKDTree(local[:,:2]) if xy_tree is None else xy_tree
    _,ids=tree.query(query[:,:2],k=min(32,len(local)),distance_upper_bound=.25)
    if ids.ndim==1:ids=ids[:,None]
    valid=ids<len(local)
    neighbors=local[np.minimum(ids,len(local)-1)]
    valid &= np.abs(neighbors[:,:,2]-query[:,None,2])<=z_resolution+1e-6
    count=valid.sum(axis=1)
    center=(neighbors*valid[:,:,None]).sum(axis=1)/np.maximum(count[:,None],1)
    delta=(neighbors-center[:,None,:])*valid[:,:,None]
    covariance=np.einsum('nki,nkj->nij',delta,delta)/np.maximum(count[:,None,None],1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    normal=eigenvectors[:,:,0]
    normal*=np.where(normal[:,2]<0.,-1.,1.)[:,None]
    support=(count>=3)&(eigenvalues[:,1]>1e-5)
    normal[~support]=0.
    normals[selected]=normal@frame.T
    supported[selected]=support

    # The first lexicographic key above each query is its nearest same-column
    # observed overhead point. This preserves the previous threshold exactly.
    dtype=np.dtype([('x','<i8'),('y','<i8'),('z','<f8')])
    columns=np.empty(len(full),dtype=dtype)
    keys=np.floor(full[:,:2]/xy_resolution).astype(np.int64)
    columns['x'],columns['y'],columns['z']=keys[:,0],keys[:,1],full[:,2]
    columns.sort(order=['x','y','z'])
    targets=np.empty(len(query),dtype=dtype)
    keys=np.floor(query[:,:2]/xy_resolution).astype(np.int64)
    targets['x'],targets['y']=keys[:,0],keys[:,1]
    targets['z']=query[:,2]+z_resolution+1e-6
    position=np.searchsorted(columns,targets,side='right')
    found=position<len(columns)
    nearest=columns[np.minimum(position,len(columns)-1)]
    found &= (nearest['x']==targets['x'])&(nearest['y']==targets['y'])
    overhead[selected[found]]=nearest['z'][found]-query[found,2]
    return normals,supported,overhead
