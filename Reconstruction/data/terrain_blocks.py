"""WFC large terrain -> complete PCD -> fixed core/context block geometry.

Preview only. No neural feature caching, network updates, or trained output.
"""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh
from common.geneator.filter_wfc import build_wfc_terrain
from common.pointcloud_io import write_pcd
from utils.mesh_sampling import sample_surface
from utils.pointcloud import read_json,write_json,workspace_path
from Reconstruction.data.storage import npy_save,json_save


def chunks(xyz,n=500000):
    for start in range(0,len(xyz),n):yield start,xyz[start:start+n]


def block_index(xyz,size):
    origin=np.floor(xyz.min(0)/size)*size
    shape=np.floor((xyz.max(0)-origin)/size).astype(int)+1
    counts=Counter()
    for _,points in chunks(xyz):
        cells=np.floor((points.astype(np.float64)-origin)/size).astype(np.int64)
        keys=np.ravel_multi_index(cells.T,shape)
        unique,number=np.unique(keys,return_counts=True)
        counts.update(dict(zip(map(int,unique),map(int,number))))
    return [dict(block_id=f'block_{key:05d}',cell=list(map(int,np.unravel_index(key,shape))),
                 core_lower=(origin+np.array(np.unravel_index(key,shape))*size).tolist(),
                 core_upper=(origin+(np.array(np.unravel_index(key,shape))+1)*size).tolist(),core_points=number)
            for key,number in sorted(counts.items())]


def extract(xyz,lower,upper):
    indices=[]
    for start,points in chunks(xyz):
        local=np.flatnonzero(((points>=lower)&(points<upper)).all(1))
        indices.append(local+start)
    ids=np.concatenate(indices)
    return ids,np.asarray(xyz[ids])


def representative_blocks(blocks,scene,cfg):
    """Metadata anchors, containing core first, then nearest unused core and ID."""
    transform=np.asarray(scene['transform']);anchors=[]
    for role,kind,railings in [('straight_railed','straight',True),('straight_open','straight',False),
                              ('switchback','switchback',None),('spiral','spiral',None),('ramp','ramp',None)]:
        row=next(r for r in scene['stairs'] if r['kind']==kind and (railings is None or bool(r['railings'])==railings))
        anchors.append((role,[*row['center_local_xy'],cfg['floor_z_m'][row['from_floor']]+row['rise_m']/2]))
    corridor=scene['corridors'][0]
    door=np.asarray(corridor['local_to_scene'])@np.array([0.,corridor['doorway_offset_m'],1.,1.])
    anchors.append(('corridor_door',door[:3]))
    row=scene['stairs'][0]
    r,c=row['upper_landing_cell'];rows,columns=cfg['shape'];dx,dy=cfg['tile_dim_m'][:2]
    x,y=(c-(columns-1)/2)*dx,((rows-1)/2-r)*dy
    anchors.append(('platform_edge',[x,y,cfg['floor_z_m'][row['to_floor']]]))
    anchors.append(('floor',[0.,0.,cfg['floor_z_m'][0]]))
    chosen=[]
    for role,p in anchors:
        point=(transform@np.r_[p,1.])[:3]
        # Zero distance to a containing core. Resolve other candidates by box distance.
        ranked=sorted(blocks,key=lambda b:(not ((point>=b['core_lower'])&(point<b['core_upper'])).all(),
                      np.linalg.norm(np.maximum(np.maximum(np.asarray(b['core_lower'])-point,point-np.asarray(b['core_upper'])),0)),b['block_id']))
        block=dict(next(b for b in ranked if b['block_id'] not in {c['block_id'] for c in chosen}))
        block.update(role=role,anchor=point.tolist());chosen.append(block)
    return chosen


def prepare(config):
    root=workspace_path(config['root']);terrain=root/f"terrain_{config['terrain_seed']}";terrain.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    if not (terrain/'scene.json').exists():
        _,mesh,_,_,_,scene=build_wfc_terrain(config['terrain_seed'],config['generation'],terrain)
        temporary=terrain/'terrain.obj.tmp';mesh.export(temporary,file_type='obj');temporary.replace(terrain/'terrain.obj')
        json_save(terrain/'scene.json',scene)
    else:
        mesh=trimesh.load(terrain/'terrain.obj',force='mesh',process=False);scene=read_json(terrain/'scene.json')
    print(f'Terrain Mesh ready: {len(mesh.faces):,} faces',flush=True)
    if not (terrain/'terrain.npy').exists():
        points,_=sample_surface(mesh,config['surface_spacing_m'])
        npy_save(terrain/'terrain.npy',points)
        del points
    xyz=np.load(terrain/'terrain.npy',mmap_mode='r')
    if not (terrain/'terrain.pcd').exists():
        temporary=terrain/'terrain.pcd.tmp';write_pcd(temporary,xyz);temporary.replace(terrain/'terrain.pcd')
    print(f'Complete terrain PCD: {len(xyz):,} points, spacing {config["surface_spacing_m"]} m',flush=True)
    blocks=block_index(xyz,np.array(config['block_size_m']))
    write_json(terrain/'blocks.json',dict(block_size_m=config['block_size_m'],context_m=config['context_m'],
                                        scene_split='preview_only',blocks=blocks))
    # Preview covers stair connections and other spatially sampled blocks.
    # This selects views only. Mask centers and shapes never use structure IDs.
    if config.get('representative_blocks')==8:
        chosen=representative_blocks(blocks,scene,read_json(terrain/'wfc_config.resolved.json'))
        stride=max(1,int(np.ceil(len(xyz)/400000)))
        write_pcd(terrain/'terrain_display.pcd',xyz[::stride])
        render_overview(terrain,xyz,chosen)
        report=dict(terrain=terrain.name,complete_points=len(xyz),blocks_total=len(blocks),representatives=chosen,
                    original_spacing_m=config['surface_spacing_m'],wall_s=perf_counter()-started)
        write_json(terrain/'base_report.json',report)
        return report
    transform=np.array(scene['transform']);anchors=[]
    cfg=read_json(terrain/'wfc_config.resolved.json')
    for kind in ['straight','switchback','spiral','ramp']:
        row=next(r for r in scene['stairs'] if r['kind']==kind)
        p=np.array([*row['center_local_xy'],cfg['floor_z_m'][row['from_floor']]+row['rise_m']/2,1.])
        anchors.append((transform@p)[:3])
    chosen=[]
    for point in anchors:
        ranked=sorted(blocks,key=lambda b:np.linalg.norm((np.array(b['core_lower'])+b['core_upper'])/2-point))
        chosen.append(next(b for b in ranked if b not in chosen))
    rng=np.random.default_rng(config['terrain_seed'])
    remaining=[b for b in blocks if b not in chosen and b['core_points']>1000]
    chosen += [remaining[i] for i in rng.choice(len(remaining),config['preview_blocks']-len(chosen),replace=False)]
    # This entry only builds terrain and block geometry. Corruptions are made
    # globally by terrain_subtraction before prepare.py extracts model inputs.
    samples=[]
    for block in chosen:
        folder=terrain/block['block_id'];folder.mkdir(exist_ok=True)
        lower=np.asarray(block['core_lower'])-config['context_m'];upper=np.asarray(block['core_upper'])+config['context_m']
        global_ids,complete=extract(xyz,lower,upper)
        core=((complete>=block['core_lower'])&(complete<block['core_upper'])).all(1)
        np.save(folder/'complete.npy',complete);np.save(folder/'terrain_indices.npy',global_ids)
        np.save(folder/'core_mask.npy',core);write_pcd(folder/'complete.pcd',complete)
        print(f'{block["block_id"]}: {len(complete):,} complete context points',flush=True)
    # Only the large-terrain context display is decimated. Selected blocks are full resolution.
    preview_stride=max(1,int(np.ceil(len(xyz)/400000)))
    write_pcd(terrain/'terrain_display.pcd',xyz[::preview_stride])
    render_overview(terrain,xyz,chosen)
    manifest=dict(version=config['version'],config=config,terrain=terrain.name,complete_points=len(xyz),
                  blocks_total=len(blocks),preview_blocks=chosen,samples=samples,
                  display_stride=preview_stride,training=False,wall_s=perf_counter()-started,
                  audit=dict(all_terrain_points_assigned_once=sum(b['core_points'] for b in blocks)==len(xyz),
                             corruption_scope='global_before_block_extraction'))
    write_json(root/'manifest.json',manifest)
    return manifest


def render_overview(folder,xyz,blocks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import product
    fig=plt.figure(figsize=(12,10));ax=fig.add_subplot(111,projection='3d')
    points=xyz[np.random.default_rng(0).choice(len(xyz),min(240000,len(xyz)),replace=False)]
    ax.scatter(*points.T,s=.5,c=points[:,2],cmap='viridis',alpha=.75,rasterized=True)
    for block in blocks:
        lo,hi=np.array(block['core_lower']),np.array(block['core_upper'])
        corners=np.array(list(product(*zip(lo,hi))))
        for i,a in enumerate(corners):
            for b in corners[i+1:]:
                if np.count_nonzero(a!=b)==1:ax.plot(*np.stack([a,b]).T,c='red',lw=1)
        ax.text(*((lo+hi)/2),block['block_id'],color='black',fontsize=8)
    ax.set(xlabel='Map X (m)',ylabel='Map Y (m)',zlabel='Z (m)',title='Continuous WFC terrain and selected block cores')
    ax.set_box_aspect(np.maximum(xyz.max(0)-xyz.min(0),.1));ax.view_init(28,-55)
    fig.tight_layout();fig.savefig(folder/'overview.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='Reconstruction/configs/terrain.json')
    p.add_argument('--terrain-seed',type=int)
    a=p.parse_args();config=read_json(workspace_path(a.config))
    if a.terrain_seed is not None:config['terrain_seed']=a.terrain_seed
    prepare(config)

if __name__=='__main__':main()
