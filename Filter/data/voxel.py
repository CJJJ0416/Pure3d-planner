"""World-aligned voxel representatives with exact original-row mappings."""
import numpy as np


def representatives(coord,size):
    grid=np.floor(coord.astype(np.float64)/size).astype(np.int64)
    _,first,inverse=np.unique(grid,axis=0,return_index=True,return_inverse=True)
    order=np.argsort(first)
    remap=np.empty(len(order),np.int64);remap[order]=np.arange(len(order))
    return first[order],remap[inverse]
