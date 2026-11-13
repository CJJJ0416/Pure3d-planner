"""Map-aligned spatial grids, shared by complete and corrupted inputs."""
import numpy as np


def bounds_grid(lower,upper,margin=1.,spacing=.24):
    origin=np.floor((np.asarray(lower,dtype=np.float64)-margin)/spacing)*spacing
    end=(np.floor((np.asarray(upper,dtype=np.float64)+margin)/spacing)+1)*spacing
    shape=np.rint((end-origin)/spacing).astype(int)
    return origin.astype(np.float32),shape


def input_grid(xyz,margin=1.):
    return bounds_grid(xyz.min(0),xyz.max(0),margin)
