"""Direct target-rich local components. No building-sized WFC generation."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'common/geneator/terrain-generator'))
import numpy as np
import trimesh
from common.geneator.filter_components import SEMANTICS,COLORS,surface_union
from common.geneator.filter_wfc import object_proxies
from common.geneator.wfc_connections import box,rod,straight_segment,flight_rails,annular_tread
from common.pointcloud_io import write_pcd
from Filter.data.augment import augment_cloud
from Filter.data.groups import loss_groups
from Filter.data.voxel import representatives
from Filter.data.targets import StaticSurfaceTargets
from Filter.visualization.visualize import render_overview,render_comparison,render_details
from utils.mesh_sampling import sample_surface
from utils.pointcloud import read_json,write_json,workspace_path

KINDS=['straight_railed','straight_open','switchback','spiral','wall_corner','railing','corridor','landing']


def build(seed,ordinal,kind):
    rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal,1101]))
    example=kind=='block_ghost_example'
    parts,roles=[],[]
    def add(semantic,mesh):
        instance=len(roles)+1
        roles.append(dict(instance_id=instance,name=f'{semantic}_{instance}',semantic=semantic,
            semantic_id=SEMANTICS[semantic],target_role=1,color_rgb=COLORS[SEMANTICS[semantic]].tolist()))
        parts.append(dict(mesh=mesh,instance_id=instance,semantic_id=SEMANTICS[semantic],target_role=1))
        return instance
    add('floor',box([4.8,4.8,.1] if example else [3.8,3.8,.1],[0,0,-.05]))
    add('wall',box([.1,4.6,2.3] if example else [.1,3.7,2.3],
                   [-2.3,0,1.15] if example else [-1.85,0,1.15]))
    if kind in ['wall_corner','corridor']:
        add('wall',box([3.7,.1,2.3],[0,1.85,1.15]))
    if kind=='corridor':
        add('ceiling',box([3.8,3.8,.1],[0,0,2.35]))
    rise=float(rng.uniform(.65,1.15));width=float(rng.uniform(.65,1.05));run=float(rng.uniform(1.5,2.3))
    if example: rise,width,run=.9,.9,1.8
    if kind=='spiral':
        for j in range(8):
            add('stair',annular_tread(.2,1.05,j*np.pi/5,(j+1)*np.pi/5,(j+1)*rise/8,.08))
        add('column_beam',rod([0,0,0],[0,0,rise],.18))
    elif kind=='switchback':
        for x,start,direction,base in [(-.55,-.9,1,0),(.55,.9,-1,rise/2)]:
            stairs,_=straight_segment(.7,1.8,rise/2,base,x,start,direction,4,.08)
            add('stair',stairs)
        add('floor',box([1.8,.4,.1],[0,1.1,rise/2-.05]))
    else:
        if kind in ['wall_corner','railing','corridor']:
            width,run,rise=.65,1.1,.55
        stairs,_=straight_segment(width,run,rise,0,0,-run/2,1,6,.08)
        add('stair',stairs)
        if kind in ['straight_railed','landing','block_ghost_example']:
            for rail in flight_rails(width,run,rise,0,0,-run/2,1,.85): add('railing',rail)
    # A thin-structure context exists in each block, including open stair cases.
    rail_x=2.2 if example else 1.55
    for y in [-1.5,-.5,.5,1.5]: add('railing',rod([rail_x,y,0],[rail_x,y,.95],.025))
    add('railing',rod([rail_x,-1.55,.95],[rail_x,1.55,.95],.035))
    if kind=='landing': add('floor',box([2.1,.8,.14],[0,1.3,rise-.07]))
    if example:
        add('floor',box([1.5,.9,.14],[0,1.35,rise-.07]))
        add('large_block_obstacle',box([.65,.7,.65],[1.25,-.9,.325]))
        for x in [-.65,.65]:
            for y in [1.,1.7]: add('column_beam',rod([x,y,0],[x,y,rise-.14],.055))
    frame=trimesh.transformations.rotation_matrix(rng.uniform(0,2*np.pi),[0,0,1])
    frame[:3,3]=[rng.uniform(-8,8),rng.uniform(-8,8),rng.uniform(-.2,.2)]
    if example: frame=np.eye(4)
    instances={}
    for semantic,mesh in ([] if example else object_proxies([0,0,0])):
        origin=np.r_[mesh.bounds[:,:2].mean(0),mesh.bounds[0,2]]
        mesh.apply_translation(-origin);mesh.apply_scale(rng.uniform(.8,1.05))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(rng.uniform(-np.pi,np.pi),[0,0,1]))
        angle=rng.uniform(-np.pi,np.pi)
        mesh.apply_translation([1.15*np.cos(angle),1.15*np.sin(angle),0])
        instances[str(SEMANTICS[semantic])]=add(semantic,mesh)
    for part in parts: part['mesh'].apply_transform(frame)
    mesh,meta=surface_union(parts);meta['face_id']=np.arange(len(mesh.faces),dtype=np.int64)
    for role in roles: role['faces']=int((meta['instance_id']==role['instance_id']).sum())
    axis=float(rng.uniform(0,2*np.pi))
    corridor_frame=trimesh.transformations.rotation_matrix(axis,[0,0,1])
    corridor=dict(id='local_motion',floor=0,width_m=3.6,length_m=3.6,height_m=2.4,
        local_to_scene=corridor_frame.tolist(),motion_interval_m=[-1.7,1.7],object_instances=instances)
    scene=dict(seed=seed,case_seed=seed*100+ordinal,family='local_component',kind=kind,
        coordinate_frame='map',frame_id='map',units='m',z_axis='up',transform=frame.tolist(),corridors=[corridor],
        floors=[dict(floor=0,z_m=frame[2,3])],bounds_world=mesh.bounds.tolist(),
        generator='direct local components, native terrain-generator stair parts',local_extent_m=[6,6,5])
    return mesh,meta,roles,scene


def export_local_window(folder,variant,rough,clean,scene,window):
    """Crop actual context rows after global labelling, retaining global indices."""
    center=np.asarray(scene['transform'])[:3,3]+[0,0,1]
    half=np.asarray(window['core_size_m'])/2
    lower,upper=center-half,center+half
    context_lower,context_upper=lower-window['context_margin_m'],upper+window['context_margin_m']
    xyz=rough['coord']
    ids=np.flatnonzero(((xyz>=context_lower)&(xyz<context_upper)).all(1))
    core=((xyz[ids]>=lower)&(xyz[ids]<upper)).all(1)
    for name,value in [('index',ids),('core',core),('offsets',np.array([0,len(ids)],np.int64))]:
        np.save(folder/f'windows_{variant:03d}_{name}.npy',value)
    np.savez(folder/f'windows_{variant:03d}.npz',point_index=ids,core_mask=core,
             offsets=np.array([0,len(ids)],np.int64))
    counts=np.bincount(rough['loss_group'][ids[core]],minlength=5).tolist()
    boxfolder=folder/f'boxes/variant_{variant:03d}/box_00000';boxfolder.mkdir(parents=True,exist_ok=True)
    clean_context=((clean>=context_lower)&(clean<context_upper)).all(1)
    row=dict(id=0,variant=variant,box_path=str(boxfolder.relative_to(folder)),
        core_lower=lower.tolist(),core_upper=upper.tolist(),context_lower=context_lower.tolist(),
        context_upper=context_upper.tolist(),context_points=len(ids),core_points=int(core.sum()),
        clean_context_points=int(clean_context.sum()),
        clean_core_points=int(((clean>=lower)&(clean<upper)).all(1).sum()),
        group_counts=counts,dynamic_points=counts[4],registration_points=counts[3],scene_id=scene['scene_id'],
        geometry_group_id=scene['geometry_group_id'],split=scene['split'],geometry_seed=scene['seed'],
        coordinate_frame='map',model_input_fields=['x','y','z'])
    write_json(boxfolder/'box.json',row)
    write_json(folder/f'windows_{variant:03d}.json',dict(core_size_m=window['core_size_m'],
        context_margin_m=window['context_margin_m'],selected_window=0,windows=[row]))
    if window['export_individual_pcd_boxes']:
        attributes={name:value[ids] for name,value in rough.items()}
        attributes.update(point_index=ids,core_mask=core)
        write_pcd(boxfolder/'rough.pcd',xyz[ids])
        write_pcd(boxfolder/'clean.pcd',clean[clean_context])
        np.savez(boxfolder/'rough_attributes.npz',**attributes)
        np.savez(boxfolder/'targets.npz',keep_target=rough['keep_target'][ids],
            surface_distance_m=rough['surface_distance_m'][ids],point_index=ids,core_mask=core)
    return row,ids,core


def variants(folder,clean,attrs,mesh,scene,config,suite,suite_id):
    teacher=StaticSurfaceTargets(mesh,config['target']);records=[]
    aug=deepcopy(config['augmentations']);aug['suite_seed']=suite_id
    if suite=='sparse': aug.update(retention_probability_range=[.25,.45],hole_size_m_range=[.7,1.2])
    if suite=='few_positions': aug['dynamic_trails']['positions_range']=[3,7]
    for v,profile in enumerate(config['variants']):
        started=perf_counter();seed=scene['case_seed']*100+suite_id*10000+v
        rough,record,sources=augment_cloud(clean,attrs,aug,seed,profile,scene)
        raw_points=len(rough['coord'])
        selected,_=representatives(rough['coord'],config['input_voxel_m'])
        rough={key:value[selected] for key,value in rough.items()}
        rough['input_point_index_before_downsample']=selected
        record['input_downsampling']=dict(voxel_m=config['input_voxel_m'],before=raw_points,after=len(selected),selection='original_surface_point')
        target,summary=teacher.score(rough);rough.update(target);rough['loss_group']=loss_groups(rough)
        arrays=folder/f'arrays_{v:03d}';arrays.mkdir(parents=True,exist_ok=True)
        for name in ['coord','keep_target','loss_group','semantic_id']: np.save(arrays/f'{name}.npy',rough[name])
        write_pcd(folder/f'rough_{v:03d}.pcd',rough['coord'])
        np.savez(folder/f'rough_attributes_{v:03d}.npz',**rough)
        np.savez(folder/f'augmentation_sources_{v:03d}.npz',**sources)
        write_json(folder/f'augmentation_{v:03d}.json',dict(**record,target_report=summary))
        write_json(folder/f'targets_{v:03d}.json',summary)
        row,ids,core=export_local_window(folder,v,rough,clean,scene,config['window'])
        xyz=rough['coord'];counts=row['group_counts']
        records.append(dict(id=f'{scene["scene_id"]}_{suite}_v{v}',path=row['box_path'],variant=v,
            rough_points=len(ids),core_points=int(core.sum()),clean_points=row['clean_context_points'],
            group_counts=counts,dynamic_points=counts[4],registration_points=counts[3],
            split=scene['split'],geometry_group_id=scene['geometry_group_id']))
        if scene['kind']=='straight_railed' and suite=='nominal':
            render_comparison(folder,v,xyz,rough['semantic_id'],rough['source_role'],rough['keep_target'],
                              np.array([xyz.min(0),xyz.max(0)]),'comparison')
            render_details(folder,v,clean,rough,record)
        print(f"local {scene['scene_id']} {suite} v{v} | points={len(xyz)} core={int(core.sum())} groups={counts} | {perf_counter()-started:.1f}s",flush=True)
    write_json(folder/'boxes.json',dict(scene_id=scene['scene_id'],samples=records,count=4))


def generate(config):
    root=workspace_path(config['dataset_root']);root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    manifest=(read_json(manifest_path) if manifest_path.exists() else
              dict(version=config['version'],coordinate_frame='map',units='m',z_axis='up',scenes=[]))
    started=perf_counter()
    for batch in config['batches']:
        for seed in batch['seeds']:
            for ordinal,kind in enumerate(KINDS):
                folder=root/f'seed_{seed}_{kind}';folder.mkdir(parents=True,exist_ok=True)
                mesh,meta,roles,scene=build(seed,ordinal,kind)
                scene.update(scene_id=folder.name,split=batch['split'],geometry_group_id=f'local_seed_{seed}')
                mesh.export(folder/'clean.obj',include_color=False,digits=9)
                np.savez(folder/'face_metadata.npz',**meta);write_json(folder/'scene.json',scene)
                write_json(folder/'roles.json',dict(semantic_classes=SEMANTICS,instances=roles))
                clean,face,_=sample_surface(mesh,config['sampling']['candidate_spacing_m'],return_face_ids=True)
                grid=np.floor(clean.astype(np.float64)/config['sampling']['voxel_m']).astype(np.int64)
                _,chosen=np.unique(grid,axis=0,return_index=True);chosen.sort();clean=clean[chosen];face=face[chosen]
                attrs={k:v[face] for k,v in meta.items()};attrs['surface_normal']=mesh.face_normals[face].astype(np.float32)
                write_pcd(folder/'clean.pcd',clean);np.savez(folder/'clean_attributes.npz',**attrs)
                render_overview(folder,mesh,meta,scene)
                variants(folder,clean,attrs,mesh,scene,config,'nominal',0)
                if scene['split']=='test':
                    for name,k in [('sparse',1),('few_positions',2)]:
                        suite=folder/'suites'/name;suite.mkdir(parents=True,exist_ok=True)
                        variants(suite,clean,attrs,mesh,scene,config,name,k)
                entry=dict(scene_id=folder.name,geometry_seed=seed,kind=kind,split=scene['split'],
                    geometry_group_id=scene['geometry_group_id'],batch=batch['id'],clean_points=len(clean))
                manifest['scenes']=[s for s in manifest['scenes'] if s['scene_id']!=folder.name]+[entry]
                write_json(root/'manifest.json',manifest)
    write_json(root/'splits.json',{split:[s['scene_id'] for s in manifest['scenes'] if s['split']==split] for split in ['train','val','test']})
    write_json(root/'config.resolved.json',config)
    write_json(root/'preparation.json',dict(local_cases=len(manifest['scenes']),
        seed_groups=len({s['geometry_seed'] for s in manifest['scenes']}),wall_s=perf_counter()-started))
    print('Local dataset complete',flush=True)


def generate_example(spec,seed=None,output=None):
    """One full-point case. Global teacher queries precede the context crop."""
    from Filter.visualization.visualize import render_block_example
    seed=spec['seed'] if seed is None else seed
    folder=workspace_path(spec['output'] if output is None else output)
    folder.mkdir(parents=True,exist_ok=True)
    config=read_json(workspace_path(spec['base_config']))
    config['augmentations']['floating_ghosts']=spec['floating_ghosts']
    timing={};started=perf_counter();tick=perf_counter()
    mesh,meta,roles,scene=build(seed,0,'block_ghost_example')
    scene.update(scene_id=folder.name,split='example',geometry_group_id=f'example_seed_{seed}',
        version=spec['version'],input_downsampling='none',candidate_spacing_m=spec['candidate_spacing_m'])
    mesh.export(folder/'clean.obj',include_color=False,digits=9)
    np.savez(folder/'face_metadata.npz',**meta)
    write_json(folder/'scene.json',scene)
    write_json(folder/'roles.json',dict(semantic_classes=SEMANTICS,instances=roles))
    timing['mesh_s']=perf_counter()-tick;tick=perf_counter()
    clean,face,sampling=sample_surface(mesh,spec['candidate_spacing_m'],return_face_ids=True)
    attrs={name:value[face] for name,value in meta.items()}
    attrs['surface_normal']=mesh.face_normals[face].astype(np.float32)
    write_pcd(folder/'clean.pcd',clean)
    np.savez(folder/'clean_attributes.npz',**attrs)
    timing['sample_and_save_clean_s']=perf_counter()-tick;tick=perf_counter()
    rough,record,sources=augment_cloud(clean,attrs,config['augmentations'],seed,spec['variant'],scene)
    record['input_downsampling']=dict(enabled=False,voxel_m=None,before=len(rough['coord']),after=len(rough['coord']))
    timing['augmentation_s']=perf_counter()-tick;tick=perf_counter()
    teacher=StaticSurfaceTargets(mesh,config['target'])
    target,summary=teacher.score(rough)
    rough.update(target);rough['loss_group']=loss_groups(rough)
    timing['global_surface_labels_s']=perf_counter()-tick;tick=perf_counter()
    xyz=rough['coord'];center=np.array([0.,0.,1.]);half=np.asarray(spec['window']['core_size_m'])/2
    lower,upper=center-half,center+half; margin=spec['window']['context_margin_m']
    core=((xyz>=lower)&(xyz<upper)).all(1)
    context=((xyz>=lower-margin)&(xyz<upper+margin)).all(1)
    rough.update(point_index=np.arange(len(xyz),dtype=np.int64),core_mask=core,context_mask=context,
                 region_id=np.where(core,0,np.where(context,1,2)).astype(np.uint8))
    row,ids,local_core=export_local_window(folder,0,rough,clean,scene,spec['window'])
    write_pcd(folder/'rough.pcd',xyz)
    write_pcd(folder/'rough_000.pcd',xyz)
    write_pcd(folder/'labels.pcd',xyz,rough['keep_target'],field_name='keep_target')
    write_pcd(folder/'retained_gt.pcd',xyz[rough['keep_target']>=.5])
    write_pcd(folder/'context.pcd',xyz[ids])
    write_pcd(folder/'core.pcd',xyz[ids[local_core]])
    write_pcd(folder/'context_labels.pcd',xyz[ids],rough['keep_target'][ids],field_name='keep_target')
    write_pcd(folder/'core_labels.pcd',xyz[core],rough['keep_target'][core],field_name='keep_target')
    np.savez(folder/'rough_attributes_000.npz',**rough)
    np.savez(folder/'targets.npz',keep_target=rough['keep_target'],surface_distance_m=rough['surface_distance_m'],
        point_index=rough['point_index'],core_mask=core,context_mask=context,region_id=rough['region_id'])
    np.savez(folder/'augmentation_sources_000.npz',**sources)
    record['labels_status']='D2 complete, full-point reference before context crop'
    write_json(folder/'augmentation_000.json',record)
    write_json(folder/'targets_000.json',summary)
    write_json(folder/'boxes.json',dict(scene_id=scene['scene_id'],count=1,samples=[dict(
        id=f'{scene["scene_id"]}_v0',variant=0,path=row['box_path'],rough_points=len(ids),
        core_points=int(core.sum()),split='example',geometry_group_id=scene['geometry_group_id'])]))
    write_json(folder/'config.resolved.json',dict(spec,seed=seed,output=str(folder),
        augmentations=config['augmentations'],target=config['target']))
    timing['crop_and_save_s']=perf_counter()-tick;tick=perf_counter()
    render_overview(folder,mesh,meta,scene)
    render_block_example(folder,rough,row,record)
    timing['plots_s']=perf_counter()-tick
    ghost=np.isin(rough['source_role'],[2,6]);score=rough['keep_target']
    floating_ids=[entry['id'] for entry in record['floating_ghosts'] if entry['kind']!='near_surface_control']
    floating=np.isin(rough['ghost_placement_id'],floating_ids)
    scatter_ids=[entry['id'] for entry in record['floating_ghosts'] if entry['kind'] in ['volume_scatter','vertical_scatter']]
    core_ghosts=core&ghost;core_scatter=core&np.isin(rough['ghost_placement_id'],scatter_ids)
    def statistics(mask):
        values=score[mask]
        return dict(points=int(mask.sum()),mean_keep_target=float(values.mean()) if len(values) else None,
            retained_at_0_5=int((values>=.5).sum()),low_at_0_1=int((values<.1).sum()))
    regions={}
    for code,name in enumerate(['core','context_ring','outside_context']):
        mask=rough['region_id']==code
        regions[name]=dict(all=statistics(mask),ghosts=statistics(mask&ghost),
            accurate_structure=statistics(mask&(rough['source_role']==0)))
    report=dict(seed=seed,output=str(folder),clean_points=len(clean),rough_points=len(xyz),
        input_downsampling=False,input_voxel_m=None,candidate_spacing_m=spec['candidate_spacing_m'],
        ghost_points=int(ghost.sum()),ghost_fraction=float(ghost.mean()),
        floating_points=int(floating.sum()),floating_fraction=float(floating.mean()),
        core_ghost_points=int(core_ghosts.sum()),core_scattered_points=int(core_scatter.sum()),
        core_scatter_fraction=float(core_scatter.sum()/core_ghosts.sum()),
        sampling=sampling,regions=regions,floating_ghosts=statistics(floating),
        disconnected_controls=statistics(rough['is_disconnected_control']),
        near_surface_copies=statistics(ghost&(rough['surface_distance_m']<.04)),
        label_histogram_edges=np.linspace(0,1,11).tolist(),label_histogram_counts=np.histogram(score,np.linspace(0,1,11))[0].tolist(),
        context_points=len(ids),core_points=int(core.sum()),outside_context_points=int((~context).sum()),
        model_input_fields=['x','y','z'],training_started=False,timing=timing,total_s=perf_counter()-started)
    write_json(folder/'report.json',report)
    print(report,flush=True)
    return report


def export_mesh_gt_demo(spec):
    """Complete Mesh -> full GT PCD, plus intact geometry with sparse ghosts."""
    from Filter.visualization.visualize import render_mesh_gt_demo
    source=workspace_path(spec['output']);folder=source/'mesh_gt_demo'
    folder.mkdir(parents=True,exist_ok=True)
    tick=perf_counter()
    mesh=trimesh.load(source/'clean.obj',force='mesh',process=False)
    metadata=dict(np.load(source/'face_metadata.npz'))
    clean,face,sampling=sample_surface(mesh,spec['candidate_spacing_m'],return_face_ids=True)
    clean_attributes={name:value[face] for name,value in metadata.items()}
    clean_attributes.update(coord=clean,keep_target=np.ones(len(clean),np.float32),
        point_index=np.arange(len(clean),dtype=np.int64),surface_normal=mesh.face_normals[face].astype(np.float32))
    parent=dict(np.load(source/'rough_attributes_000.npz'))
    ghost_ids=np.flatnonzero(np.isin(parent['source_role'],[2,6]))
    coord=np.concatenate([clean,parent['coord'][ghost_ids]])
    rough=dict(coord=coord,
        source_role=np.r_[np.zeros(len(clean),np.uint8),parent['source_role'][ghost_ids]],
        source_face_id=np.r_[face,parent['source_face_id'][ghost_ids]],
        semantic_id=np.r_[clean_attributes['semantic_id'],parent['semantic_id'][ghost_ids]],
        instance_id=np.r_[clean_attributes['instance_id'],parent['instance_id'][ghost_ids]],
        ghost_placement_id=np.r_[np.full(len(clean),-1,np.int32),parent['ghost_placement_id'][ghost_ids]],
        point_index=np.arange(len(coord),dtype=np.int64),
        gt_surface_point_index=np.r_[np.arange(len(clean)),np.full(len(ghost_ids),-1,np.int64)],
        parent_rough_point_index=np.r_[np.full(len(clean),-1,np.int64),ghost_ids],
        is_disconnected_control=np.zeros(len(coord),bool))
    teacher=StaticSurfaceTargets(mesh,read_json(workspace_path(spec['base_config']))['target'])
    labels,summary=teacher.score(rough);rough.update(labels)
    write_pcd(folder/'mesh_gt.pcd',clean)
    write_pcd(folder/'mesh_gt_labels.pcd',clean,clean_attributes['keep_target'],field_name='keep_target')
    write_pcd(folder/'input.pcd',coord)
    write_pcd(folder/'input_gt_scores.pcd',coord,rough['keep_target'],field_name='keep_target')
    np.savez(folder/'mesh_gt_attributes.npz',**clean_attributes)
    np.savez(folder/'input_attributes.npz',**rough)
    data_s=perf_counter()-tick;tick=perf_counter()
    render_mesh_gt_demo(folder,mesh,clean,rough)
    floating=(rough['ghost_placement_id']>=0)&(rough['ghost_placement_id']<9)
    report=dict(seed=spec['seed'],reference_mesh=str((source/'clean.obj').resolve()),
        gt_points=len(clean),input_points=len(coord),ghost_source_points=len(ghost_ids),
        floating_points=int(floating.sum()),gt_surface_score_min=float(rough['keep_target'][:len(clean)].min()),
        floating_score_max=float(rough['keep_target'][floating].max()),
        clean_surface_distance_max_m=float(rough['surface_distance_m'][:len(clean)].max()),
        input_background='complete Mesh surface samples, no holes or stripe deletion',
        input_downsampling=False,plot_downsampling=False,candidate_spacing_m=spec['candidate_spacing_m'],
        sampling=sampling,model_input_fields=['x','y','z'],output='same XYZ and point order plus continuous keep_target',
        training_started=False,hard_filtering=False,target_report=summary,
        data_generation_s=data_s,plot_s=perf_counter()-tick)
    write_json(folder/'report.json',report)
    print(report,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('Filter/configs/dataset.json'))
    p.add_argument('--example',action='store_true',help='Generate only the complete full-point block example')
    p.add_argument('--mesh-gt-demo',action='store_true',help='Export full Mesh GT and intact input with sparse ghosts')
    p.add_argument('--seed',type=int)
    p.add_argument('--output',type=Path)
    a=p.parse_args();config=read_json(workspace_path(a.config))
    if a.mesh_gt_demo: export_mesh_gt_demo(config)
    elif a.example: generate_example(config,a.seed,a.output)
    else: generate(config)

if __name__=='__main__': main()
