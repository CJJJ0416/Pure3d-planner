"""Permanent geometry/heldout data and resumable replacement of active training data.

initialize prepares only training stage 1. prepare-stage is the explicit promotion
entry. READY plus the atomic manifest is the commit point. Interrupted preparation
keeps the published version and resumes pending files on the next invocation.
"""
import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh
from utils.pointcloud import read_json,workspace_path
from utils.mesh_sampling import sample_surface
from Reconstruction.data.storage import json_save,array_save,npy_save
from Reconstruction.data.terrain_blocks import prepare as prepare_terrain,extract
from Reconstruction.data.terrain_subtraction import surface_index,export_input
from Reconstruction.data.random_subtraction import SubtractionGenerator
from Reconstruction.data.grid import bounds_grid
from Reconstruction.data.prepare import openings_for_scene


def identity(config):
    digest=hashlib.sha256()
    with workspace_path(config['filter_checkpoint']).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    digest.update(str(config['filter_threshold']).encode())
    return digest.hexdigest()[:16]


def version_id(config,stage_id):
    digest=hashlib.sha256(json.dumps(dict(config=config,stage_id=stage_id,frontend=identity(config)),sort_keys=True).encode())
    return f'stage_{stage_id}_{digest.hexdigest()[:12]}'


def mask_seed(config,terrain_seed,version):
    return int(np.random.SeedSequence([config['experiment_seed'],terrain_seed,version]).generate_state(1)[0])


def global_mask(config,spec,out,target,seed):
    report=out/'report.json'
    if report.exists():return read_json(report)
    out.mkdir(parents=True,exist_ok=True);started=perf_counter()
    xyz=np.load(workspace_path(spec['folder'])/'terrain.npy',mmap_mode='r')
    terrain_cfg=read_json(workspace_path(config['terrain_config']))
    index=surface_index(xyz)
    keep,removed,metadata=SubtractionGenerator(seed,terrain_cfg['subtraction']).sample(
        xyz,'heavy',sample_seed=seed,spatial_index=index,target_removed_fraction=target)
    del index;gc.collect()
    array_save(out/'indices.npz',keep_indices=keep,removed_indices=removed)
    json_save(out/'degradation.json',metadata)
    temporary=out/'input.pcd.tmp';exact=export_input(temporary,xyz,keep);temporary.replace(out/'input.pcd')
    result=dict(seed=seed,target_removed_fraction=target,removed_fraction=len(removed)/len(xyz),
                removed_points=len(removed),input_points=len(keep),region_count=metadata['region_count'],
                serialized_input_exact_subset=exact,wall_s=perf_counter()-started)
    json_save(report,result)
    print(f"MASK {spec['seed']} {out.name}: {result['removed_fraction']:.3%}, {result['region_count']} regions",flush=True)
    return result


def cache_features(frontend,path,complete,indices,local=None,filtered=True):
    if path.exists():return
    selected=np.arange(len(complete)) if local is None else local
    tick=perf_counter();values=frontend.encode(np.asarray(complete[selected]),filtered)
    values['global_index']=indices[selected[values['keep_index']]]
    array_save(path,**values)
    print(f'CACHE {path.name}: {len(selected):,} -> {len(values["xyz"]):,} points {perf_counter()-tick:.1f}s',flush=True)


def base(config,device,frontend):
    root=workspace_path(config['root']);front=identity(config);permanent=root/'permanent.json'
    if permanent.exists():return read_json(permanent)
    rows=[]
    for spec in config['terrains']:
        terrain=workspace_path(spec['folder']);base_report=terrain/'base_report.json'
        if not base_report.exists():
            cfg=read_json(workspace_path(config['terrain_config']))
            cfg.update(root=str(terrain.parent),terrain_seed=spec['seed'],representative_blocks=8)
            prepare_terrain(cfg)
        xyz=np.load(terrain/'terrain.npy',mmap_mode='r');scene=read_json(terrain/'scene.json')
        blocks=read_json(base_report)['representatives'];mesh=query=None
        for block in blocks:
            folder=root/'blocks'/terrain.name/block['block_id'];folder.mkdir(parents=True,exist_ok=True)
            lo,hi=np.asarray(block['core_lower']),np.asarray(block['core_upper'])
            if not (folder/'complete.npy').exists():
                indices,complete=extract(xyz,lo-config['context_m'],hi+config['context_m'])
                npy_save(folder/'complete_indices.npy',indices);npy_save(folder/'complete.npy',complete)
            else:
                indices=np.load(folder/'complete_indices.npy',mmap_mode='r');complete=np.load(folder/'complete.npy',mmap_mode='r')
            origin,shape=bounds_grid(lo,hi,config['context_m'])
            json_save(folder/'scene.json',dict(scene,frame=scene['transform'],core_lower=lo.tolist(),core_upper=hi.tolist(),origin=origin.tolist(),shape=shape.tolist(),openings=openings_for_scene(scene)))
            if not (folder/'targets.npz').exists():
                from Reconstruction.data.geometry import MeshQueries
                from Reconstruction.data.supervision import build_targets
                if query is None:
                    mesh=trimesh.load(terrain/'terrain.obj',force='mesh',process=False);query=MeshQueries(mesh,device)
                print(f'TARGET {terrain.name}/{block["block_id"]}',flush=True)
                targets=build_targets(query,complete,indices,origin,shape,config['levels_m'],spec['seed'],config['near_surface_m'])
                array_save(folder/'targets.npz',**targets);del targets
            if not (folder/'reference.npy').exists():
                if mesh is None:mesh=trimesh.load(terrain/'terrain.obj',force='mesh',process=False)
                faces=((mesh.triangles.max(1)>=lo)&(mesh.triangles.min(1)<hi)).all(1)
                local=mesh.submesh([np.flatnonzero(faces)],append=True,repair=False)
                reference,_=sample_surface(local,.0075)
                npy_save(folder/'reference.npy',reference[((reference>=lo)&(reference<hi)).all(1)])
            grid=f'{terrain.name}_{block["block_id"]}'
            full=root/'features'/front/grid/'complete.npz';filtered=full.with_name('complete_filtered.npz')
            cache_features(frontend,full,complete,indices,filtered=False)
            cache_features(frontend,filtered,complete,indices)
            core=((complete>=lo)&(complete<hi)).all(1)
            rows.append(dict(scene=terrain.name,kind=block['role'],terrain_seed=spec['seed'],split=spec['split'],
                block_id=block['block_id'],grid_id=grid,folder=str(folder.relative_to(root)),terrain_folder=spec['folder'],
                complete_indices=str((folder/'complete_indices.npy').relative_to(workspace_path('.'))),mask_id='complete',
                sample_id=grid+'_complete',input_prefix='complete',feature_file=str(filtered.relative_to(root)),
                complete_feature_file=str(full.relative_to(root)),frontend_id=front,origin=origin.tolist(),shape=shape.tolist(),
                core_lower=lo.tolist(),core_upper=hi.tolist(),core_points=int(core.sum()),core_removed_fraction=0.,
                difficulty='complete',input_points=len(complete),feature_cached=True,targets_cached=True))
            print(f'BASE {grid} ready ({len(rows)}/48)',flush=True)
            del complete,indices;gc.collect()
    result=dict(config=config,frontend_id=front,samples=rows)
    json_save(permanent,result);return result


def corrupted_rows(config,base_rows,spec,mask_folder,public_folder,mask_id,difficulty,frontend):
    root=workspace_path(config['root']);rows=[]
    with np.load(mask_folder/'indices.npz') as data:removed=data['removed_indices']
    for full in base_rows:
        if full['terrain_seed']!=spec['seed']:continue
        folder=root/full['folder'];complete=np.load(folder/'complete.npy',mmap_mode='r')
        indices=np.load(folder/'complete_indices.npy',mmap_mode='r')
        keep=~np.isin(indices,removed);local=np.flatnonzero(keep)
        cache=mask_folder/'features'/f"{full['grid_id']}.npz"
        cache_features(frontend,cache,complete,indices,local)
        core=((complete>=full['core_lower'])&(complete<full['core_upper'])).all(1)
        row=dict(full,mask_id=mask_id,sample_id=full['grid_id']+'_'+mask_id,input_prefix=mask_id,
                 mask_file=str((public_folder/'indices.npz').relative_to(workspace_path('.'))),
                 feature_file=str((public_folder/'features'/cache.name).relative_to(root)),
                 core_removed_fraction=float((core&~keep).sum()/core.sum()),difficulty=difficulty,input_points=len(local))
        rows.append(row)
    return rows


def heldout(config,permanent,frontend):
    root=workspace_path(config['root']);file=root/'heldout.json'
    if file.exists():return read_json(file)
    rows=[]
    for spec in config['terrains']:
        if spec['split']=='train':continue
        for i,(name,target) in enumerate(config['heldout_targets'].items(),101):
            folder=root/'heldout'/f"terrain_{spec['seed']}"/name
            global_mask(config,spec,folder,target,mask_seed(config,spec['seed'],i))
            rows+=corrupted_rows(config,permanent['samples'],spec,folder,folder,name,name,frontend)
    result=dict(samples=rows);json_save(file,result);return result


def publish(root,config):
    permanent=read_json(root/'permanent.json');fixed=read_json(root/'heldout.json')
    active=read_json(root/'train_current'/'ready.json')
    manifest=dict(version=config['version'],config=config,frontend_id=permanent['frontend_id'],
                  stage_id=active['stage_id'],data_version=active['data_version'],
                  samples=permanent['samples']+fixed['samples']+active['samples'],
                  masks=active['masks'],curriculum=read_json(workspace_path(config['curriculum_config'])))
    json_save(root/'manifest.json',manifest)
    json_save(root/'stage_state.json',dict(stage_id=active['stage_id'],data_version=active['data_version'],pending_stage=None))
    return manifest


def recover_switch(root,config):
    """Finish or roll back a directory rename interrupted before publication."""
    current,previous,pending=[root/n for n in ['train_current','train_previous','train_pending']]
    if previous.exists():
        if not current.exists():
            if (pending/'ready.json').exists():pending.replace(current)
            else:previous.replace(current)
        publish(root,config)
        if previous.exists():shutil.rmtree(previous)


def prepare_stage(config,stage_id,device='cuda:0',frontend=None):
    root=workspace_path(config['root']);root.mkdir(parents=True,exist_ok=True)
    recover_switch(root,config)
    current=root/'train_current';pending=root/'train_pending';version=version_id(config,stage_id)
    if (current/'ready.json').exists() and read_json(current/'ready.json')['data_version']==version:
        return publish(root,config)
    own_frontend=frontend is None
    if own_frontend:
        from Reconstruction.data.features import FrozenFrontend
        frontend=FrozenFrontend(workspace_path(config['filter_checkpoint']),device,config['filter_threshold'])
    if pending.exists() and (pending/'preparing.json').exists() and read_json(pending/'preparing.json')['data_version']!=version:
        shutil.rmtree(pending)
    pending.mkdir(exist_ok=True)
    json_save(pending/'preparing.json',dict(stage_id=stage_id,data_version=version))
    json_save(root/'stage_state.json',dict(stage_id=read_json(current/'ready.json')['stage_id'] if current.exists() else None,
              data_version=read_json(current/'ready.json')['data_version'] if current.exists() else None,pending_stage=stage_id))
    permanent=read_json(root/'permanent.json');curriculum=read_json(workspace_path(config['curriculum_config']))
    stage=next(s for s in curriculum['stages'] if s['stage_id']==stage_id)
    rows=[];masks=[]
    for spec in config['terrains']:
        if spec['split']!='train':continue
        out=pending/f"terrain_{spec['seed']}";public=current/out.name
        report=global_mask(config,spec,out,stage['target_removed_fraction'],mask_seed(config,spec['seed'],stage_id))
        masks.append(dict(terrain_seed=spec['seed'],**report))
        rows+=corrupted_rows(config,permanent['samples'],spec,out,public,f'stage_{stage_id}',f'stage_{stage_id}',frontend)
    json_save(pending/'ready.json',dict(stage_id=stage_id,data_version=version,samples=rows,masks=masks,
              feature_caches=len(rows),complete=True))
    if current.exists():current.replace(root/'train_previous')
    pending.replace(current)
    manifest=publish(root,config)
    if (root/'train_previous').exists():shutil.rmtree(root/'train_previous')
    print(f'PUBLISHED {version}: {len(rows)} corrupted blocks. Previous training files removed.',flush=True)
    return manifest


def initialize(config,device='cuda:0'):
    root=workspace_path(config['root']);root.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4)
    from Reconstruction.data.features import FrozenFrontend
    frontend=FrozenFrontend(workspace_path(config['filter_checkpoint']),device,config['filter_threshold'])
    permanent=base(config,device,frontend)
    heldout(config,permanent,frontend)
    return prepare_stage(config,1,device,frontend)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['initialize','prepare-stage'])
    p.add_argument('--config',default='Reconstruction/configs/dataset.json');p.add_argument('--stage-id',type=int,choices=[1,2,3,4])
    p.add_argument('--device',default='cuda:0');a=p.parse_args();config=read_json(workspace_path(a.config))
    if a.command=='initialize':initialize(config,a.device)
    else:
        if a.stage_id is None:p.error('prepare-stage requires --stage-id')
        import torch
        torch.set_num_threads(4)
        prepare_stage(config,a.stage_id,a.device)

if __name__=='__main__':main()
