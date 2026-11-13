"""Apply irregular Subtraction to every part of one complete terrain PCD."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from common.pointcloud_io import write_pcd
from utils.pointcloud import read_json,write_json,workspace_path
from Reconstruction.data.random_subtraction import SubtractionGenerator


def surface_index(xyz,spacing=.1,chunk=1000000):
    """Uniform surface-cell representatives without sorting an N-by-3 matrix."""
    lower=np.floor(xyz.min(0)/spacing).astype(np.int64)
    upper=np.floor(xyz.max(0)/spacing).astype(np.int64)
    shape=upper-lower+1;keys=[];indices=[]
    for start in range(0,len(xyz),chunk):
        cells=np.floor(xyz[start:start+chunk]/spacing).astype(np.int64)-lower
        key=(cells[:,0]*shape[1]+cells[:,1])*shape[2]+cells[:,2]
        unique,first=np.unique(key,return_index=True)
        keys.append(unique);indices.append(first+start)
    _,first=np.unique(np.concatenate(keys),return_index=True)
    representatives=np.concatenate(indices)[first]
    return cKDTree(xyz),representatives


def export_input(path,xyz,keep,chunk=1000000):
    """Stream original rows in order, with no coordinate conversion."""
    header=('VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
            f'WIDTH {len(keep)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(keep)}\nDATA binary\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        for start in range(0,len(keep),chunk):
            np.ascontiguousarray(xyz[keep[start:start+chunk]],dtype='<f4').tofile(stream)
    with path.open('rb') as stream:
        while stream.readline().strip()!=b'DATA binary':pass
        exact=True
        for start in range(0,len(keep),chunk):
            ids=keep[start:start+chunk]
            actual=np.fromfile(stream,dtype='<f4',count=len(ids)*3).reshape(-1,3)
            exact &= np.array_equal(actual,xyz[ids])
    return bool(exact)


def prepare(config,mask_seed,region_multiplier=1.):
    root=workspace_path(config['root']);terrain=root/f"terrain_{config['terrain_seed']}"
    name=f'global_subtraction_{mask_seed}' if region_multiplier==1. else f'global_subtraction_{mask_seed}_density_{region_multiplier:g}'
    out=terrain/name;out.mkdir(exist_ok=True)
    xyz=np.load(terrain/'terrain.npy',mmap_mode='r');blocks=read_json(terrain/'blocks.json')['blocks']
    settings=dict(config['subtraction'])
    settings['heavy_regions']=[max(1,int(round(v*len(blocks)*region_multiplier))) for v in settings['heavy_regions']]
    started=perf_counter();index=surface_index(xyz)
    print(f'Global index: {len(xyz):,} points, {len(index[1]):,} surface cells, {perf_counter()-started:.2f}s',flush=True)
    tick=perf_counter()
    keep,removed,metadata=SubtractionGenerator(mask_seed,settings).sample(xyz,'heavy',sample_seed=mask_seed,spatial_index=index)
    mask_s=perf_counter()-tick;del index
    print(f'Whole terrain: {metadata["region_count"]:,} regions, removed {len(removed):,} ({metadata["removed_fraction"]:.2%})',flush=True)
    np.savez(out/'indices.npz',keep_indices=keep,removed_indices=removed)
    write_json(out/'degradation.json',metadata)
    exact=export_input(out/'input.pcd',xyz,keep)
    write_pcd(out/'removed.pcd',xyz[removed])
    # Both global views select from the same original rows. Deleted display
    # rows are excluded from the input, so downsampling cannot fill a real gap.
    stride=max(1,int(np.ceil(len(xyz)/800000)))
    display_ids=np.arange(0,len(xyz),stride)
    mask=np.zeros(len(xyz),bool);mask[removed]=True
    write_pcd(out/'input_display.pcd',xyz[display_ids[~mask[display_ids]]])
    write_pcd(out/'gt_display.pcd',xyz[display_ids])
    red_stride=max(1,int(np.ceil(len(removed)/400000)))
    write_pcd(out/'removed_display.pcd',xyz[removed[::red_stride]])
    positions=np.asarray([r['center'] for r in metadata['regions']])
    size=np.array(config['block_size_m']);origin=np.floor(xyz.min(0)/size)*size
    visited=np.unique(np.floor((positions-origin)/size).astype(int),axis=0)
    report=dict(terrain=str(terrain),source='terrain.npy',input_points=len(xyz),retained_points=len(keep),
                removed_points=len(removed),removed_fraction=len(removed)/len(xyz),
                region_count=metadata['region_count'],region_types=dict(Counter(r['kind'] for r in metadata['regions'])),
                occupied_core_blocks=len(blocks),blocks_with_mask_centers=len(visited),mask_seed=mask_seed,region_multiplier=region_multiplier,
                serialized_input_exact_subset=exact,whole_terrain=True,training=False,
                display_stride=stride,removed_display_stride=red_stride,mask_s=mask_s,total_s=perf_counter()-started)
    write_json(out/'report.json',report);print(report,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='Reconstruction/configs/terrain.json')
    p.add_argument('--mask-seed',type=int,default=251113);p.add_argument('--region-multiplier',type=float,default=1.)
    a=p.parse_args();prepare(read_json(workspace_path(a.config)),a.mask_seed,a.region_multiplier)

if __name__=='__main__':main()
