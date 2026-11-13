"""Native teacher supervision for independent stair/ramp geometry and two 40-profile banks."""
import argparse
from itertools import combinations
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter

from utils.pointcloud import ROOT, read_json, write_json, workspace_path
os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'datasets/cache/cupy'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'datasets/cache/matplotlib'))
sys.path.insert(0,str(ROOT/'common/vst_global_planner/src'))
from vst_global_planner.conditioned_teacher import ConditionedTeacher
from vst_global_planner.tomogram_cloud import tomogram_points
from common.geneator.stair_families import generate_augmented_stair_scene
from common.geneator.wfc_connections import wedge, box
from common.pointcloud_io import read_pcd_xyz, write_pcd
from utils.mesh_sampling import sample_surface
from utils.pointcloud import voxel_downsample
from Traversability.experiments.stair_data import geometry_cache, empty_metadata, legacy_stair_mask
from Traversability.data.teacher_conditions import FIXED

NAMES=['clearance_min','slope_max','step_max','stand_ratio']


def specifications(config):
    seed=config['seed']
    rows=[]
    for split,count in config['splits_per_kind'].items():
        for kind in config['kinds']:
            rng=np.random.default_rng(seed)
            unit=np.column_stack([(rng.permutation(count)+rng.random(count))/count for _ in range(7)])
            for u in unit:
                rows.append(dict(seed=seed,split=split,kind=kind,railings=bool(rng.integers(2)),
                    mirror=bool(rng.integers(2)),three_floors=bool(rng.integers(2)),wall=True,
                    riser_m=.12+.32*u[0],tread_depth_m=.18+.67*u[1],width_m=.55+1.35*u[2],
                    side_gap_m=.2+1.3*u[3],inner_radius_m=.18+.77*u[4],turns=.65+.95*u[5],
                    rise_m=2.3+1.2*u[6],slope_rad=float(rng.uniform(.12,.85))))
                seed+=1
    return rows


def ramp_scene(folder,spec):
    import trimesh
    from common.geneator.stair_families import render_overview
    rng=np.random.default_rng(spec['seed'])
    rise=spec['rise_m'];run=rise/np.tan(spec['slope_rad']);w=spec['width_m'];margin=1.25
    base=float(rng.uniform(2.5,3.8)) if spec['three_floors'] else 0.
    width=w+2*spec['side_gap_m'];length=run+2*margin
    parts=[wedge(w,run,rise,base),box([width,length,.14],[0,0,base-.07]),
           box([width,margin,.14],[0,(run+margin)/2,base+rise-.07])]
    for sign in [-1,1]:
        parts.append(box([.12,length,rise+1.1],[sign*(width/2+.06),0,base+(rise+1.1)/2]))
    if spec['railings']:
        from common.geneator.wfc_connections import flight_rails
        parts+=flight_rails(w,run,rise,base,0,-run/2,1,.95)
    if base:
        parts.append(box([width,length,.14],[0,0,-.07]))
    mesh=trimesh.util.concatenate(parts)
    yaw=float(rng.uniform(-np.pi,np.pi));transform=trimesh.transformations.rotation_matrix(yaw,[0,0,1])
    transform[:2,3]=rng.uniform(-.15,.15,2)
    if spec['mirror']:transform[:3,:3]=transform[:3,:3]@np.diag([-1.,1.,1.])
    mesh.apply_transform(transform)
    folder.mkdir(parents=True,exist_ok=True);mesh.export(folder/'terrain.ply')
    points,sampling=sample_surface(mesh,.03);down=voxel_downsample(points,.05)
    write_pcd(folder/'terrain.pcd',points);write_pcd(folder/'terrain_downsampled.pcd',down)
    record=dict(id='target_ramp',kind='ramp',width_m=w,rise_m=rise,slope_rad=spec['slope_rad'],run_m=run,
                yaw_rad=yaw,translation_xy=transform[:2,3].tolist(),base_z_m=base,
                unrotated_footprint_m=[width,length],railings=spec['railings'],mirrored=spec['mirror'])
    scene=dict(seed=spec['seed'],split=spec['split'],family_id=f"family_{spec['seed']}",variant=0,kind='ramp',
               stairs=[record],input_points=len(down),transform=transform.tolist(),sampling=sampling,
               bounds=mesh.bounds.tolist(),augmentation=spec)
    write_json(folder/'scene.json',scene)
    # Shared overview helper assumes discrete stairs. Ramp gets its own geometric caption.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,5))
    chosen=down[np.random.default_rng(0).choice(len(down),min(len(down),60000),replace=False)]
    for i,(elev,azim) in enumerate([(30,-55),(5,-85)],1):
        ax=fig.add_subplot(1,2,i,projection='3d');ax.scatter(*chosen.T,c=chosen[:,2],cmap='terrain',s=.3,lw=0)
        ax.set_box_aspect(np.ptp(down,axis=0));ax.view_init(elev,azim)
    fig.suptitle(f"{folder.name} | ramp {spec['slope_rad']:.4f}rad | width {w:.3f}m")
    fig.savefig(folder/'overview.png',dpi=130);plt.close(fig)
    return scene


def facility_mask(xyz,scene):
    r=scene['stairs'][0]
    a=r['yaw_rad'];rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    local=(xyz[:,:2]-r['translation_xy'])@rotation
    width,length=r['unrotated_footprint_m'];base=r.get('base_z_m',0.)
    return ((abs(local[:,0])<=width/2+.3)&(abs(local[:,1])<=length/2+.6)
            &(xyz[:,2]>=base-.05)&(xyz[:,2]<=base+r['rise_m']+.15))


def nine_values(lo,hi,center,offsets):
    candidates=list(np.clip(np.r_[lo,hi,center+np.asarray(offsets)],lo,hi)) if center is not None else list(np.linspace(lo,hi,9))
    values=list(dict.fromkeys(float(np.float32(v)) for v in candidates))
    for value in np.linspace(lo,hi,65):
        if len(values)>=9:break
        value=float(np.float32(value))
        if value not in values:values.append(value)
    return sorted(values[:9])


def dominant(values,lo,hi):
    v=values[np.isfinite(values)&(values>lo)&(values<hi)]
    if not len(v):return None
    unique,count=np.unique(np.round(v,4),return_counts=True)
    return float(unique[np.argmax(count)])


def condition_banks(teacher,seed,ranges,region,training):
    rng=np.random.default_rng(np.random.SeedSequence([seed,44401]))
    profiles=[dict(id=name,condition=c) for name,c in zip(['nominal','low_stand','strict_stand','relaxed'],FIXED)]
    banks=[]
    grad=teacher.grad_sq.get();maximum=teacher.grad_max.get()
    valid=teacher.valid_ground.get()&(teacher.interval.get()>=.5)&region
    pairs=[list(combinations(range(4+9*p,13+9*p),2)) for p in range(4)]
    for bank in range(2):
        anchor=np.array([.5,.6,.4,.05]) if bank==0 else rng.uniform(ranges[:,0],ranges[:,1])
        flat=(1.2*teacher.resolution*np.tan(anchor[1]))**2
        step_values=np.sqrt(maximum[valid&(grad>flat)])
        slope_values=np.arctan(np.sqrt(grad[valid])/(1.2*teacher.resolution))
        counts=np.rint(uniform_filter((grad<flat).astype(np.float32),size=(1,7,7),mode='constant')*49)
        step_region=valid&(grad>flat)&(maximum<=anchor[2]**2)
        thresholds=(counts[step_region]+2)/49
        centers=[None,dominant(slope_values,*ranges[1]),dominant(step_values,*ranges[2]),dominant(thresholds,*ranges[3])]
        offsets=[None,[-.08,-.04,-.01,0,.01,.04,.08] if training else [-.10,-.05,-.02,0,.02,.05,.10],
                 [-.04,-.02,-.005,0,.005,.02,.04] if training else [-.06,-.03,-.01,0,.01,.03,.06],
                 np.array([-1.25,-.75,-.25,0,.25,.75,1.25])/49 if training else np.array([-1.1,-.6,-.2,0,.2,.6,1.1])/49]
        columns=list(range(4))
        for p,column in enumerate([0,1,2,4]):
            lo,hi=ranges[p]
            values=np.linspace(lo,hi,9) if p==0 else nine_values(lo,hi,centers[p],offsets[p])
            for i,value in enumerate(values):
                c=np.array([anchor[0],anchor[1],anchor[2],.5,anchor[3]])
                c[column]=float(value)
                columns.append(len(profiles));profiles.append(dict(id=f'bank{bank}_{NAMES[p]}_{i}',condition=c.tolist(),bank=bank,
                    parameter=p,value=float(value),critical_value=centers[p]))
        banks.append(columns)
    return profiles,banks,pairs


def teacher_region(teacher,scene):
    nx,ny=teacher.shape_xy
    ix,iy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
    xy=np.c_[((ix-nx//2)*teacher.resolution+teacher.center[0]).ravel(),
             ((iy-ny//2)*teacher.resolution+teacher.center[1]).ravel()]
    xyz=np.c_[np.tile(xy,(len(teacher.ground),1)),teacher.ground.ravel()]
    return facility_mask(xyz,scene).reshape(teacher.ground.shape)


def label_scene(folder,teacher,profiles,data_root,scene_id,done,total):
    folder.mkdir(parents=True,exist_ok=True)
    costs=[]
    for i,p in enumerate(profiles):
        raw=teacher.cost(p['condition']);payload=teacher.payload(p['condition'],raw)
        cloud=tomogram_points(payload);costs.append(cloud[:,3])
        if i==0:
            xyz=cloud[:,:3];write_pcd(folder/'nominal.pcd',xyz,cloud[:,3])
        if i==3:
            with (folder/'vst_relaxed.pickle').open('wb') as stream:pickle.dump(payload,stream,pickle.HIGHEST_PROTOCOL)
        write_json(data_root/'preparation.json',dict(complete=False,scenes_done=done,scenes_total=total,
            active_scene=scene_id,profile=i+1,profiles=len(profiles)))
    np.save(folder/'cost.npy',np.column_stack(costs))
    np.save(folder/'conditions.npy',np.asarray([p['condition'] for p in profiles],np.float32))
    write_json(folder/'profiles.json',profiles)
    return xyz


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    config=read_json(parser.parse_args().config);root=workspace_path(config['data']);root.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();plan=specifications(config);write_json(root/'geometry_plan.json',dict(scenes=plan,config=config))
    metadata=empty_metadata();metadata['version']='parameter_decoupling_v4'
    previous=read_json(workspace_path(config['replay_data'])/'blocks.json')
    replay=[s for s in previous['scenes'] if s['split'] in ['train','val']]
    jobs=[('replay',s) for s in replay]+[('new',s) for s in plan]
    manifest=[]
    for index,(source,spec) in enumerate(jobs):
        scene_id=spec['id'] if source=='replay' else f"scene_{spec['seed']}"
        folder=root/source/scene_id;ready=folder/'ready.json'
        if ready.exists():
            stored=read_json(ready);row,blocks=stored['row'],stored['blocks']
        else:
            if source=='new':
                scene=(read_json(folder/'scene.json') if (folder/'scene.json').exists() else
                       ramp_scene(folder,spec) if spec['kind']=='ramp' else generate_augmented_stair_scene(folder,spec))
                pcd=folder/'terrain_downsampled.pcd'
            else:
                pcd=workspace_path(config['replay_data'])/spec['input_pcd']
                scene=read_json(workspace_path(config['replay_data'])/scene_id/'scene.json')
            points=read_pcd_xyz(pcd);teacher=ConditionedTeacher(points)
            region=teacher_region(teacher,scene) if source=='new' else np.ones_like(teacher.ground,dtype=bool)
            profiles,banks,pairs=condition_banks(teacher,spec['seed'],np.asarray(config['condition_ranges']),region,spec['split']=='train')
            xyz=label_scene(folder/'conditioned',teacher,profiles,root,scene_id,index,len(jobs))
            membership=facility_mask(xyz,scene) if source=='new' else legacy_stair_mask(xyz,scene)
            base_scene=scene if source=='new' else dict(seed=spec['seed'],split=spec['split'],family_id=scene_id)
            arrays,blocks=geometry_cache(folder,xyz,base_scene,input_pcd=pcd,membership=membership)
            mask_file=folder/'geometry/facility_mask.npy';np.save(mask_file,membership)
            kind=spec['kind'] if source=='new' else 'replay'
            blocks=[dict(b,replay=source=='replay',augmentation_target=kind) for b in blocks]
            row=dict(id=scene_id,seed=spec['seed'],split=spec['split'],source=source,kind=kind,family=kind,
                input_pcd=str(pcd),geometry_scene=str(folder/'scene.json') if source=='new' else str(workspace_path(config['replay_data'])/scene_id/'scene.json'),
                arrays=dict(arrays,facility_mask=str(mask_file),cost=str(folder/'conditioned/cost.npy'),conditions=str(folder/'conditioned/conditions.npy')),
                query_points=len(xyz),input_points=len(points),profiles=profiles,profile_banks=banks,blocks=len(blocks))
            write_json(ready,dict(row=row,blocks=blocks));del teacher
        metadata['scenes'].append(row);metadata['blocks'].extend(blocks)
        metadata['condition_pairs']=[list(combinations(range(4+9*p,13+9*p),2)) for p in range(4)]
        for name in ['train','evaluation']:
            write_json(root/name/'blocks.json',metadata)
        manifest.append(dict(id=scene_id,source=source,kind=row['kind'],split=row['split'],pcd=row['input_pcd'],geometry_scene=row['geometry_scene']))
        write_json(root/'manifest.json',dict(scenes=manifest,config=config))
        print('READY',index+1,len(jobs),scene_id,row['kind'],spec['split'],flush=True)
    result=dict(complete=True,scenes_done=len(jobs),scenes_total=len(jobs),new_scenes=144,replay_scenes=5,
                splits=dict(train=96,val=24,test=24),banks=2,profiles_per_bank=40,seconds=time.perf_counter()-began)
    write_json(root/'data_ready.json',result);write_json(root/'preparation.json',result)
    print('DATA_COMPLETE',result,flush=True)


if __name__=='__main__':main()
