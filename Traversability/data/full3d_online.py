"""Geometry/index caching for trainable Tiny, without encoder-feature caches."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np

from Traversability.data.full3d import prepare_fixed
from Traversability.data.rotating import prepare_generation
from Traversability.data.sliced import progress
from utils.pointcloud import read_json, workspace_path, write_json


def verify_generation(config):
    root=workspace_path(config['data']);output=workspace_path(config['output'])
    reference=workspace_path(config['frozen_reference_run'])/'scene_archive'
    meta=read_json(root/'manifest.json');checks=[]
    for scene in [s for s in meta['scenes'] if s['split']=='train']:
        old=read_json(reference/scene['id']/'supervision.json')
        prior=read_json(reference/scene['id']/'data_summary.json')
        xyz=np.load(root/scene['id']/'coord.npy',mmap_mode='r')
        current=dict(scene=scene['id'],source_pcd_sha256_equal=scene['source_sha256']==old['source_sha256'],
            query_xyz_sha256_equal=hashlib.sha256(xyz.tobytes()).hexdigest()==prior['xyz_sha256'],
            conditions_equal=scene['profiles']==read_json(reference/scene['id']/'conditions.json'),
            banks_equal=scene['banks']==old['banks'],pairs_equal=scene['pairs']==old['pairs'])
        assert all(value for key,value in current.items() if key!='scene'), current
        write_json(output/'scene_archive'/scene['id']/'frozen_data_comparison.json',current)
        checks.append(current)
    return checks


def prepare_fixed_maps(config):
    """Only reuse immutable geometry and point indices, never features.npy."""
    root=prepare_fixed(config)
    source=workspace_path(config['frozen_reference_data'])/'fixed'
    meta=read_json(root/'manifest.json')
    for scene in meta['scenes']:
        folder=root/scene['id'];original=source/scene['id']
        if (folder/'online_geometry.json').exists():continue
        for name in ['geometry_coord.npy','geometry_source_index.npy','query_representative_index.npy',
                     'query_original_index.npy','query_match_distance.npy']:
            (folder/name).symlink_to((original/name).resolve())
        maps=folder/'maps';maps.mkdir(exist_ok=True)
        for row in [b for b in meta['blocks'] if b['scene']==scene['id']]:
            old=original/'features'/row['id']
            context=np.load(old/'context_source_index.npy').astype(np.int64)
            with np.load(old/'correspondence.npz') as a:
                source_index=a['representative_index'];point_index=a['point_index']
            np.savez(maps/(row['id']+'.npz'),context_index=context,
                query_local=np.searchsorted(context,source_index),point_index=point_index)
        write_json(folder/'online_geometry.json',dict(complete=True,geometry_and_indices_only=True,
            source=str(original),frozen_features_used=False,blocks=sum(b['scene']==scene['id'] for b in meta['blocks'])))
    return root


def prepare_maps(config):
    import torch
    from common.pointcloud_io import read_pcd_xyz
    from Traversability.data.voxel_hash import representatives,VoxelHash
    root=workspace_path(config['data']);meta=read_json(root/'manifest.json')
    reports=[]
    for scene in meta['scenes']:
        folder=root/scene['id'];ready=folder/'online_geometry.json'
        if ready.exists():
            reports.append(read_json(ready));continue
        torch.cuda.synchronize();started=time.perf_counter()
        raw=read_pcd_xyz(scene['source_pcd'])
        points,original,keys,low,shape=representatives(torch.from_numpy(raw).to(config['device']),config['geometry_voxel_size_m'])
        query=torch.from_numpy(np.array(np.load(folder/'coord.npy',mmap_mode='r'))).to(config['device'])
        table=VoxelHash(points,original,keys,low,shape,config['geometry_voxel_size_m'])
        source,distance,radius=table.query(query)
        np.save(folder/'geometry_coord.npy',points.cpu().numpy())
        np.save(folder/'geometry_source_index.npy',original.cpu().numpy())
        np.save(folder/'query_representative_index.npy',source.cpu().numpy())
        np.save(folder/'query_original_index.npy',original[source].cpu().numpy())
        np.save(folder/'query_match_distance.npy',distance.cpu().numpy())
        del table,keys
        size=torch.tensor(config['window']['core_size_m'],device=config['device']);context=config['window']['context_m']
        maps=folder/'maps';maps.mkdir(exist_ok=True)
        rows=[b for b in meta['blocks'] if b['scene']==scene['id']]
        count=0
        for row in rows:
            lower=torch.tensor(row['lower'],device=config['device'])
            index=torch.nonzero(((points>=lower-context)&(points<lower+size+context)).all(1)).flatten()
            with np.load(folder/(row['id']+'.npz')) as a:point_index=a['point_index'].copy()
            selected=source[torch.from_numpy(point_index).to(config['device'])]
            local=torch.searchsorted(index,selected)
            assert bool((local<len(index)).all()) and torch.equal(index[local],selected)
            np.savez(maps/(row['id']+'.npz'),context_index=index.int().cpu().numpy(),
                query_local=local.int().cpu().numpy(),point_index=point_index)
            count+=len(point_index)
        assert count==len(query)
        torch.cuda.synchronize()
        report=dict(complete=True,scene=scene['id'],raw_points=len(raw),geometry_points=len(points),
            query_points=len(query),blocks=len(rows),geometry_and_indices_only=True,frozen_features_used=False,
            maximum_match_m=float(distance.max()),maximum_cell_radius=int(radius.max()),
            preparation_s=time.perf_counter()-started)
        write_json(ready,report);reports.append(report)
        write_json(workspace_path(config['output'])/'scene_archive'/scene['id']/'online_geometry.json',report)
        progress(config,'geometry_mapping',scene=scene['id'],blocks_completed=len(rows),blocks_total=len(rows))
        del raw,points,original,source,query,distance,radius
        torch.cuda.empty_cache()
    write_json(root/'online_geometry.json',dict(complete=True,scenes=reports,frozen_features_used=False))
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--generation',type=int,required=True)
    args=parser.parse_args();config=read_json(args.config)
    local=deepcopy(config);local['fixed_data']=str(prepare_fixed(config))
    prepare_generation(local,args.generation)
    checks=verify_generation(config)
    write_json(workspace_path(config['output'])/'data_verification'/f'{args.generation:02d}.json',checks)


if __name__=='__main__':main()
