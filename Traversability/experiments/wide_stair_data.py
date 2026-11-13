"""独立宽深高直梯数据增强，复用旧数据回放，不改变原生 VST 监督规则。"""
import argparse
import copy
from pathlib import Path
import time

import numpy as np

from common.geneator.stair_families import generate_augmented_stair_scene, stair_membership
from common.pointcloud_io import read_pcd_xyz
from Traversability.data.teacher_conditions import condition_banks
from Traversability.experiments.stair_data import ConditionedTeacher, label_scene, geometry_cache, empty_metadata, legacy_stair_mask
from utils.pointcloud import ROOT, read_json, write_json, workspace_path


def specifications(config):
    """每个 seed 的高度、深度、宽度独立分层采样，不复用族内固定参数。"""
    result = []
    seed = config['seed']
    for split, count in config['splits'].items():
        rng = np.random.default_rng(seed)
        focused = int(count*config['focused_fraction'])
        for group, n in [('focused', focused), ('broad', count-focused)]:
            unit = np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(3)])
            ranges = config[group+'_geometry']
            for j in range(n):
                row = dict(seed=seed, split=split, distribution=group,
                           railings=bool(rng.random()<.75), three_floors=seed % 3 != 0,
                           wall=bool(rng.integers(0, 2)),
                           side_gap_m=float(rng.uniform(*config['side_gap_m'])))
                for k, name in enumerate(['riser_m','tread_depth_m','width_m']):
                    lo, hi = ranges[name]
                    row[name] = float(lo+unit[j,k]*(hi-lo))
                result.append(row)
                seed += 1
    # 只借用已知失败设施的尺寸，场景 seed、朝向、井口与上下文均重新生成。
    result[0].update(config['target_dimensions'])
    result[0].update(distribution='dimension_anchor', railings=True, three_floors=True)
    return result


def teacher_region(teacher, scene):
    """在教师 ground 几何上选取临界参数的来源区域，不筛除任何 cost 标签。"""
    nx,ny = teacher.shape_xy
    ix,iy = np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
    xy = np.column_stack(((ix.ravel()-nx//2)*teacher.resolution+teacher.center[0],
                          (iy.ravel()-ny//2)*teacher.resolution+teacher.center[1]))
    ground = teacher.ground
    xyz = np.column_stack((np.tile(xy,(len(ground),1)),ground.ravel()))
    return stair_membership(xyz,scene).reshape(ground.shape)


def main(specification_builder=specifications):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'Traversability/configs/wide_stair_v2.json')
    args=parser.parse_args();config=read_json(args.config);output=workspace_path(config['data']);output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();plan=specification_builder(config)
    write_json(output/'geometry_plan.json',dict(config=config,scenes=plan))
    training=empty_metadata();training['version']=config.get('dataset_version','wide_stair_v2')
    evaluation=empty_metadata();evaluation['version']=training['version']
    previous=read_json(workspace_path(config['replay_data'])/'blocks.json')
    retention=empty_metadata();retention['version']='wide_stair_v2_retention'
    replay_root=workspace_path(config['replay_data'])
    for old in [s for s in previous['scenes'] if s['split'] in ['train','val']]:
        folder=output/'replay'/old['id']
        ready=workspace_path(config.get('replay_cache',config['data']))/'replay'/old['id']/'ready.json'
        if ready.exists():
            stored=read_json(ready);row,blocks,pairs=stored['row'],stored['blocks'],stored['condition_pairs']
        else:
            metadata=read_json(replay_root/old['id']/'scene.json');pcd=replay_root/old['input_pcd']
            teacher=ConditionedTeacher(read_pcd_xyz(pcd))
            profiles,banks,pairs=condition_banks(teacher,old['seed'],
                banks=config['training_banks'] if old['split']=='train' else 1,
                critical=old['split']=='train',evaluation=old['split']=='val',ranges=config['condition_ranges'])
            xyz=label_scene(folder/'conditioned',teacher,profiles)
            family=f"replay_family_{old['seed']}"
            geometry=dict(seed=old['seed'],split=old['split'],family_id=family)
            arrays,blocks=geometry_cache(folder,xyz,geometry,input_pcd=pcd,membership=legacy_stair_mask(xyz,metadata))
            blocks=[dict(b,replay=True) for b in blocks]
            row=dict(id=old['id'],seed=old['seed'],split=old['split'],batch=family,family=family,input_pcd=str(pcd),
                arrays=dict(arrays,cost=str(folder/'conditioned/cost.npy'),conditions=str(folder/'conditioned/conditions.npy')),
                query_points=len(xyz),input_points=old['input_points'],profiles=profiles,profile_banks=banks,
                blocks=len(blocks),geometry_scene=str(replay_root/old['id']/'scene.json'))
            write_json(ready,dict(row=row,blocks=blocks,condition_pairs=pairs));del teacher
        meta=training if old['split']=='train' else evaluation
        meta['scenes'].append(row);meta['blocks'].extend(blocks);meta['condition_pairs']=pairs
        if old['split']=='val':
            retention['scenes'].append(row);retention['blocks'].extend(blocks);retention['condition_pairs']=pairs
        print('REPLAY READY',old['id'],flush=True)
    write_json(output/'retention'/'blocks.json',retention)
    manifest=[]
    for index,spec in enumerate(plan):
        folder=output/'scenes'/f"scene_{spec['seed']}";ready=folder/'ready.json'
        if ready.exists():
            stored=read_json(ready);row,blocks,scene=stored['row'],stored['blocks'],stored['scene']
            pairs=stored['condition_pairs']
        else:
            scene=(read_json(folder/'scene.json') if (folder/'scene.json').exists() else
                   generate_augmented_stair_scene(folder,spec))
            teacher=ConditionedTeacher(read_pcd_xyz(folder/'terrain_downsampled.pcd'))
            target=scene['stairs'][0]
            anchor=[.5,.6,max(.4,target['riser_m']+.03),.05]
            profiles,banks,pairs=condition_banks(teacher,spec['seed'],
                banks=config['training_banks'] if spec['split']=='train' else 1,
                critical=spec['split']=='train',evaluation=spec['split']!='train',
                ranges=config['condition_ranges'],focus_anchor=anchor,region_mask=teacher_region(teacher,scene))
            xyz=label_scene(folder/'conditioned',teacher,profiles)
            arrays,blocks=geometry_cache(folder,xyz,scene)
            target_id=spec.get('template_id','wide_stair')
            blocks=[dict(b,augmentation_target=target_id) for b in blocks]
            row=dict(id=folder.name,seed=spec['seed'],split=spec['split'],batch=scene['family_id'],family=scene['family_id'],
                input_pcd=str(folder/'terrain_downsampled.pcd'),arrays=dict(arrays,cost=str(folder/'conditioned/cost.npy'),
                conditions=str(folder/'conditioned/conditions.npy')),query_points=len(xyz),input_points=scene['input_points'],
                profiles=profiles,profile_banks=banks,blocks=len(blocks),geometry_scene=str(folder/'scene.json'))
            write_json(ready,dict(row=row,blocks=blocks,scene=scene,condition_pairs=pairs))
            del teacher
        meta=training if spec['split']=='train' else evaluation
        meta['scenes'].append(row);meta['blocks'].extend(blocks);meta['condition_pairs']=pairs
        manifest.append(dict(seed=spec['seed'],split=spec['split'],family=scene['family_id'],
            actual=scene['stairs'][0],template_id=spec.get('template_id'),overview=str(folder/'overview.png'),
            points=row['input_points'],queries=row['query_points']))
        for name,m in [('train',training),('evaluation',evaluation)]:
            m['splits']={s:dict(scenes=sum(x['split']==s for x in m['scenes']),blocks=sum(x['split']==s for x in m['blocks'])) for s in ['train','val','test']}
            write_json(output/name/'blocks.json',m)
        write_json(output/'manifest.json',dict(scenes=manifest,config=config))
        write_json(output/'preparation.json',dict(complete=False,scenes_done=index+1,scenes_total=len(plan),
                   active_scene=row['id'],seconds=time.perf_counter()-started))
        print(f"READY {index+1}/{len(plan)} {row['id']} {spec['split']} points={row['input_points']} queries={row['query_points']}",flush=True)
    def actual_value(scene,name):
        actual=scene['actual']
        return actual.get('tread_depth_m',actual.get('tread_depth_at_mid_radius_m')) if name=='tread_depth_m' else actual[name]
    stats={name:[min(actual_value(s,name) for s in manifest if s['split']=='train'),
                 max(actual_value(s,name) for s in manifest if s['split']=='train')]
           for name in ['riser_m','tread_depth_m','width_m']}
    complete=dict(complete=True,new_scenes=len(plan),independent_train_geometries=config['splits']['train'],
                  actual_train_geometry_ranges=stats,seconds=time.perf_counter()-started,
                  scenes_done=len(plan),scenes_total=len(plan))
    write_json(output/'data_ready.json',complete)
    write_json(output/'preparation.json',complete)
    print('DATA COMPLETE',stats,flush=True)


if __name__=='__main__':main()
