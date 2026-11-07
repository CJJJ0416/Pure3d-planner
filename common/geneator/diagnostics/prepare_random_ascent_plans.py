#!/usr/bin/env python3
"""Fresh native VST/GPMP plans between seeded lower/upper floor endpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from stair_nav.planning.vst import endpoint_index,make_planner,optimized_world,save_path,check_sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--count',type=int,default=5);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
    cfg=json.loads((a.run.resolve().parent/'config.resolved.json').read_text());training=ROOT/'isaac_training'
    b=json.loads(Path(cfg['building_manifest']).read_text().splitlines()[0]);sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    surface=training/b['assets']['vst_surface.npz']['path'];assert sha(surface)==b['assets']['vst_surface.npz']['sha256']
    with np.load(surface,allow_pickle=False) as z:data=np.asarray(z['data'],np.float64);center=np.asarray(z['center']);resolution=float(z['resolution']);dh=float(z['slice_dh'])
    source=training/b['assets']['paths/ascent.gpmp.csv']['path'];points=np.loadtxt(source,delimiter=',',skiprows=1)[:,:3];points[:,2]-=.1
    delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1);arc=np.r_[0,np.cumsum(length)]
    slope=np.flatnonzero(abs(delta[:,2])/np.maximum(length,1e-9)>.1);first=float(arc[slope[0]]);last=float(arc[slope[-1]+1])
    at=lambda s:np.array([np.interp(s,arc,points[:,i]) for i in range(3)])
    rng=np.random.default_rng(a.seed);results=[];rejected=[];check_sources()
    cwd=Path.cwd();os.chdir(out)
    try:
        for attempt in range(100):
            start_arc=max(0.,first-float(rng.uniform(1.,4.)));goal_arc=min(arc[-1],last+float(rng.uniform(1.,4.)))
            endpoints=[]
            for s in [start_arc,goal_arc]:
                p=at(s);d=at(min(s+.2,arc[-1]))-at(max(s-.2,0));normal=np.array([-d[1],d[0]])/max(np.linalg.norm(d[:2]),1e-9)
                p[:2]+=normal*float(rng.uniform(-.25,.25));endpoints.append(p)
            start,goal=endpoints
            try:
                si=endpoint_index(start,data,center,resolution,dh,20.);gi=endpoint_index(goal,data,center,resolution,dh,20.)
                start[2]=data[3,si[0],si[2],si[1]];goal[2]=data[3,gi[0],gi[2],gi[1]]
                if goal[2]-start[2]<1.5:raise ValueError('Endpoints do not cover full ascent')
                planner=make_planner(data,resolution)
                if not planner.plan(si,gi,False):raise ValueError('No Astar path')
                if not planner.plan(si,gi,True):raise ValueError('GPMP failed')
                route=optimized_world(planner,data,center,resolution)
                route_length=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())
                if not 12.<=route_length<=24.:raise ValueError('Route length outside diagnostic bounds')
                ident=f'random_ascent_{len(results)+1}';stem=out/(ident+'.gpmp');save_path(route,stem)
                tangent=route[min(3,len(route)-1),:2]-route[0,:2];yaw=math.atan2(tangent[1],tangent[0])+float(rng.uniform(-math.pi/9,math.pi/9))
                results.append(dict(id=ident,start_support_xyz=start.tolist(),goal_support_xyz=goal.tolist(),start_yaw=yaw,
                    route_csv=str(stem)+'.csv',route_sha256=sha(Path(str(stem)+'.csv')),route_length_m=route_length,
                    source_sampling_arcs=[start_arc,goal_arc],seed=a.seed+len(results),attempt=attempt))
                (out/'plans.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results[-1]),flush=True)
                if len(results)==a.count:break
            except (ValueError,RuntimeError,IndexError) as e:rejected.append(dict(attempt=attempt,reason=str(e)))
        else:raise RuntimeError('Insufficient valid random ascent plans')
    finally:os.chdir(cwd)
    (out/'planning_protocol.json').write_text(json.dumps(dict(seed=a.seed,count=len(results),
        backend='same native VST Astar + quintic GPMP',surface_sha256=sha(surface),source_route_sha256=sha(source),
        endpoint_lateral_jitter_m=.25,initial_yaw_jitter_deg=20.,minimum_height_change_m=1.5,
        no_student_outcome_selection=True,rejected_planning_attempts=rejected),indent=2)+'\n')


if __name__=='__main__':main()
