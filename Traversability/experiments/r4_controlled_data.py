"""The fixed R4 geometry, 33 fitting conditions and 12 held-out conditions."""
import argparse
import hashlib
from itertools import combinations
from pathlib import Path
import sys
import time

import numpy as np

from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import ROOT,read_json,write_json as _write_json,workspace_path

NAMES=['clearance_min','slope_max','step_max','stand_ratio']
TREAD_PARTS=[1,2,3,4,6,7,8,9]
LANDING_PARTS=[0,5,10]


def point_metrics(gt,pred):
    error=pred-gt;result=dict(points=len(gt),MAE=float(abs(error).mean()),MAE_raw50=float(abs(error).mean()*50),
                           bias=float(error.mean()),bias_raw50=float(error.mean()*50),thresholds={})
    for threshold in [.3,.5,.6]:
        low=gt<=threshold;high=~low;blocked=low&(pred>threshold);passed=high&(pred<=threshold)
        result['thresholds'][str(threshold)]=dict(false_block_count=int(blocked.sum()),GT_low_count=int(low.sum()),
            false_block_pct=float(100*blocked.sum()/low.sum()) if low.any() else None,
            false_pass_count=int(passed.sum()),GT_high_count=int(high.sum()),
            false_pass_pct=float(100*passed.sum()/high.sum()) if high.any() else None)
    return result


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    _write_json(temporary,value);temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions(height,baseline):
    scans=[np.arange(.2,1.01,.1),[.18,.25,.30,.40,.50,.60,.63,.67,.70],
           [.08,.20,.30,.34,height-.01,height+.01,.40,.48,.55],
           [.02,.05,.08,.10,.12,.20,.30,.40,.50]]
    holdouts=[[.35,.55,.75],[.22,.45,.65],[.32,height-.005,height+.005],[.06,.15,.35]]
    rows=[np.asarray(baseline,np.float32)];groups=[];held_groups=[]
    for parameter,values in enumerate(scans):
        columns=[]
        for value in values:
            c=np.array(baseline,np.float32);c[parameter]=value
            found=next((i for i,r in enumerate(rows) if np.array_equal(c,r)),None)
            if found is None:found=len(rows);rows.append(c)
            columns.append(found)
        groups.append(columns)
    train_count=len(rows)
    for parameter,values in enumerate(holdouts):
        columns=[]
        for value in values:
            c=np.array(baseline,np.float32);c[parameter]=value
            columns.append(len(rows));rows.append(c)
        held_groups.append(columns)
    return np.asarray(rows),groups,held_groups,train_count


def native_query(payload,xyz):
    data=np.asarray(payload['data'],np.float32).copy()
    for layer in range(data.shape[1]-1):
        take=data[3,layer+1]-data[3,layer]<payload['slice_dh']
        data[3,layer,take]=np.nan
        data[0,layer+1,take]=np.minimum(data[0,layer,take],data[0,layer+1,take])
    ij=np.rint((xyz[:,:2]-payload['center'])/payload['resolution']).astype(int)+np.array(data.shape[2:])//2
    ground=data[3,:,ij[:,0],ij[:,1]]
    d=np.where(np.isfinite(ground),np.abs(ground-xyz[:,2,None]),np.inf)
    layer=d.argmin(1)
    return data[0,layer,ij[:,0],ij[:,1]]/payload['cost_scale'],d[np.arange(len(xyz)),layer],ij


def main():
    from scipy.spatial import cKDTree
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    config=read_json(p.parse_args().config);out=workspace_path(config['output']);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'common/vst_global_planner/src'))
    from vst_global_planner.conditioned_teacher import ConditionedTeacher
    old=np.load(ROOT/'runs/planner/r4_step_slope_response_20261111/teacher.npz')
    layout=read_json(ROOT/'runs/planner/r4_cost_connectivity_audit_20261111/protocol.json')
    xyz=np.r_[old['query_xyz'],old['watch_xyz']];n=len(old['query_xyz'])
    values,groups,held,count=conditions(layout['stair']['riser_m'],config['baseline'])
    began=time.perf_counter();teacher=ConditionedTeacher(read_pcd_xyz(workspace_path(config['source_pcd'])))
    targets=[];profiles=[];errors=[]
    for i,c in enumerate(values):
        native=[float(c[0]),float(c[1]),float(c[2]),.5,float(c[3])]
        raw=teacher.cost(native);payload=teacher.payload(native,raw)
        target,dz,ij=native_query(payload,xyz);targets.append(target);errors.append(float(dz.max()))
        profiles.append(dict(column=i,condition=c.tolist(),split='train' if i<count else 'heldout',
                             teacher_parameters=teacher.parameters(native)))
        write_json(out/'preparation.json',dict(stage='teacher',done=i+1,total=len(values),complete=False))
        print('TEACHER',i+1,len(values),flush=True)
    target=np.column_stack(targets).astype(np.float32)
    pairs=np.array(sorted(cKDTree(xyz[:n]).query_pairs(.145)),np.int64)
    u,v=pairs.T;parts=old['part'].astype(np.int64)
    delta=np.abs(ij[u]-ij[v])
    take=(parts[u]==parts[v])&(np.abs(xyz[u,2]-xyz[v,2])<1e-5)&(delta.max(1)==1)
    pairs=pairs[take]
    query_cache=ROOT/'runs/planner/r4_stand_ratio_sweep_20261111/query_geometry'
    meta=read_json(ROOT/'runs/planner/large_terrain_six_vst_20261111/prediction/geometry/blocks.json')
    owner=np.full(n+4,-1,np.int64);active=[]
    for i,row in enumerate(meta['blocks']):
        with np.load(query_cache/f'block_{i:05d}.npz') as a:qi=a['query_indices']
        if len(qi):active.append(i);owner[qi]=i
    np.savez(out/'teacher.npz',query_xyz=xyz[:n],watch_xyz=xyz[n:],target=target[:n],watch_target=target[n:],
        conditions=values,part=parts,step=old['step'].astype(np.int64),flight=old['flight'].astype(np.int64),
        edges=pairs,block_owner=owner[:n],train_columns=np.arange(count),heldout_columns=np.arange(count,len(values)))
    previous=np.load(ROOT/'runs/traversability/cost_parameter_decoupling_v4/best_r4.npz')
    k=int(np.flatnonzero(np.all(np.isclose(previous['conditions'],config['baseline']),axis=1))[0])
    np.save(out/'reference_baseline.npy',np.r_[previous['prediction'][:,k],previous['watch_prediction'][:,k]])
    protocol=dict(configuration=config,source_checkpoint=str(workspace_path(config['source_checkpoint'])),
        source_checkpoint_sha256=sha(workspace_path(config['source_checkpoint'])),
        source_pcd_sha256=sha(workspace_path(config['source_pcd'])),conditions=values.tolist(),
        condition_groups=groups,condition_pairs=[list(combinations(g,2)) for g in groups],heldout_groups=held,
        train_condition_count=count,heldout_condition_count=len(values)-count,active_cores=active,
        query_points=n,watch_points=4,part_counts=np.bincount(parts,minlength=11).tolist(),
        spatial_edges=len(pairs),cross_core_edges=int((owner[pairs[:,0]]!=owner[pairs[:,1]]).sum()),
        maximum_native_ground_error_m=max(errors),teacher=teacher.parameters([.5,.6,.4,.5,.05]),
        spatial_pair_rule='Same actual horizontal support part, same height, native 8-neighbor XY indices. Across-core edges retained.',
        baseline_previous_GT_max_abs_difference=float(np.abs(target[:n,0]-old['target'][:,k]).max()),
        source_geometry_cache=str(ROOT/'runs/planner/large_terrain_six_vst_20261111/prediction/geometry'),
        source_query_cache=str(query_cache),teacher_seconds=time.perf_counter()-began,
        scope='R4 fitting diagnosis, not unseen-building generalization. No reliability mask or target filtering.')
    write_json(out/'protocol.json',protocol);write_json(out/'profiles.json',profiles)
    write_json(out/'preparation.json',dict(stage='complete',done=len(values),total=len(values),complete=True))
    print('DATA_COMPLETE',protocol['spatial_edges'],protocol['cross_core_edges'],flush=True)


if __name__=='__main__':main()
