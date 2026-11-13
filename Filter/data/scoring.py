"""Full Mesh surfaces plus sparse residuals, with continuous pointwise GT."""
import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from Filter.data.local import build
import numpy as np

from common.pointcloud_io import write_pcd
from common.geneator.filter_components import SEMANTICS
from Filter.data.augment import augment_cloud
from Filter.data.boxes import export_boxes,collect_boxes
from Filter.data.groups import loss_groups
from Filter.data.targets import StaticSurfaceTargets
from Filter.visualization.visualize import render_overview,render_mesh_gt_demo
from utils.mesh_sampling import sample_surface
from utils.pointcloud import read_json,write_json,workspace_path


def generate(config):
    root=workspace_path(config['dataset_root']);root.mkdir(parents=True,exist_ok=True)
    base=read_json(workspace_path(config['base_config']))
    manifest=dict(version=config['version'],coordinate_frame='map',test_suites=['nominal'],scenes=[])
    started=perf_counter()
    for case in config['scenes']:
        seed=case['seed'];folder=root/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        mesh,meta,roles,scene=build(seed,0,case['kind'])
        scene.update(scene_id=folder.name,split=case['split'],geometry_group_id=f'seed_{seed}')
        mesh.export(folder/'clean.obj',include_color=False,digits=9)
        np.savez(folder/'face_metadata.npz',**meta);write_json(folder/'scene.json',scene)
        write_json(folder/'roles.json',dict(semantic_classes=SEMANTICS,instances=roles))
        clean,face,_=sample_surface(mesh,config['surface_spacing_m'],return_face_ids=True)
        attrs={name:value[face] for name,value in meta.items()}
        attrs['surface_normal']=mesh.face_normals[face].astype(np.float32)
        write_pcd(folder/'clean.pcd',clean);np.savez(folder/'clean_attributes.npz',**attrs)
        render_overview(folder,mesh,meta,scene)
        teacher=StaticSurfaceTargets(mesh,base['target']);variants=[]
        for v,profile in enumerate(config['variants']):
            tick=perf_counter();rng=np.random.default_rng(np.random.SeedSequence([seed,v,5901]))
            aug=deepcopy(base['augmentations'])
            aug.update(retention_probability_range=[1.,1.],hole_count=0,stripe_width_m_range=[0.,0.],
                jitter_std_m_range=[0.,.002],jitter_fraction=.4,suite_seed=v)
            aug['disconnected_controls']['semantic_ids']=[]
            aug['dynamic_trails'].update(positions_range=[3,5],surface_fraction_range=[.003,.012])
            count=int(len(clean)*profile['floating_fraction'])
            low=mesh.bounds[0]+[-.9,-.9,.3];high=mesh.bounds[1]+[.9,.9,1.]
            aug['floating_ghosts']=[]
            if count:
                aug['floating_ghosts'].append(dict(kind='volume_scatter',semantic='wall',zone='whole_input',
                    air_regions=[dict(lower=low.tolist(),upper=high.tolist(),points=count)]))
                # A few near-surface and short-trail observations retain their
                # continuous surface-distance targets rather than source labels.
                wall=np.flatnonzero(attrs['semantic_id']==1)
                pivot=clean[rng.choice(wall)]
                aug['floating_ghosts'].append(dict(kind='near_surface_control',semantic='wall',zone='whole_input',
                    radius_m=.2,destination_xyz=(pivot+[.02,0.,0.]).tolist(),points_per_position=20,jitter_std_m=.006))
            if profile['short_trail']:
                aug['floating_ghosts'].append(dict(kind='trail',semantic='stair',zone='whole_input',radius_m=.45,
                    destination_xyz=(mesh.bounds.mean(0)+[.8,-.7,.7]).tolist(),drift_xyz=[.25,.2,.4],
                    rotation_deg=[2.,4.,6.],copies=5,points_per_position=4,jitter_std_m=.015))
            rough,record,sources=augment_cloud(clean,attrs,aug,seed*100+v,
                dict(name=profile['name'],registration=False,dynamic=profile['dynamic']),scene)
            targets,summary=teacher.score(rough);rough.update(targets)
            rough['loss_group']=loss_groups(rough)
            arrays=folder/f'arrays_{v:03d}';arrays.mkdir(exist_ok=True)
            for name in ['coord','keep_target','loss_group','semantic_id']: np.save(arrays/f'{name}.npy',rough[name])
            write_pcd(folder/f'rough_{v:03d}.pcd',rough['coord'])
            write_pcd(folder/f'gt_{v:03d}.pcd',rough['coord'],rough['keep_target'],field_name='keep_target')
            np.savez(folder/f'rough_attributes_{v:03d}.npz',**rough)
            np.savez(folder/f'augmentation_sources_{v:03d}.npz',**sources)
            windows=export_boxes(folder,v,rough,clean,attrs,config['window'],seed*100+v)
            record.update(input_downsampling=False,background='complete Mesh samples',target_report=summary)
            write_json(folder/f'augmentation_{v:03d}.json',record)
            variants.append(dict(id=v,name=profile['name'],points=len(rough['coord']),
                boxes=len(windows['windows']),wall_s=perf_counter()-tick))
            if v==2:
                preview=folder/'preview';preview.mkdir(exist_ok=True)
                render_mesh_gt_demo(preview,mesh,clean,rough)
            print(f"data seed={seed} split={case['split']} variant={v} points={len(rough['coord'])} boxes={len(windows['windows'])}",flush=True)
        boxes=collect_boxes(folder)
        manifest['scenes'].append(dict(scene_id=folder.name,geometry_seed=seed,kind=case['kind'],
            split=case['split'],geometry_group_id=scene['geometry_group_id'],clean_points=len(clean),
            variants=variants,box_count=boxes['count']))
        write_json(root/'manifest.json',manifest)
    write_json(root/'config.resolved.json',config)
    write_json(root/'splits.json',{split:[s['scene_id'] for s in manifest['scenes'] if s['split']==split]
        for split in ['train','val','test']})
    write_json(root/'preparation.json',dict(seeds=len(manifest['scenes']),samples=sum(len(s['variants']) for s in manifest['scenes']),
        boxes=sum(s['box_count'] for s in manifest['scenes']),input_downsampling=False,wall_s=perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();generate(read_json(workspace_path(a.config)))

if __name__=='__main__': main()
