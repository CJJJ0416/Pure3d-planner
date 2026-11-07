#!/usr/bin/env python3
"""Derive a ramp/no-railing variant without changing the source building.

Existing floor tiles, holes, walls, bridge, seed, flight footprint and endpoints
are retained. New mesh, analytic support, point cloud and provenance agree.
The generated scene needs fresh VST and planning before navigation training.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import trimesh
from stair_nav.core.io import read_json, write_json, write_pcd, sha256
from stair_nav.scenes.geometry import box, rotation, sample_surface
from stair_nav.scenes.stair_structure import add_prism,add_ramp

REMOVED = {'stair', 'landing', 'stair_stringer', 'stair_rail', 'stair_baluster',
           'floor_rail', 'floor_post'}


def derive(source):
    original = read_json(source/'scene.json')
    scene = deepcopy(original)
    scene['name'] = original['name']+'_ramp_no_rails'
    retained = [deepcopy(s) for s in original['solids'] if s['kind'] not in REMOVED]
    meshes = []
    scene['solids'] = []
    # Rebuild the exact saved solids rather than replaying random generation.
    for item in retained:
        if 'top_plane' in item:
            add_prism(scene, meshes, item['vertices_xy'], item['top_plane'],
                      item['bottom_plane'], item['kind'])
            scene['solids'][-1] = item
        elif item['kind'] == 'ramp':
            raise ValueError('Use an explicit prism for pre-existing relief ramps')
        else:
            meshes.append(box(item['lo'],item['hi']))
            scene['solids'].append(item)
    for st in scene['stair_flights']:
        st['source_construction']=st.get('construction')
        add_ramp(scene,meshes,st)
    mesh = trimesh.boolean.union(meshes,engine='manifold')
    transform=np.eye(4);transform[:3,:3]=rotation(scene['yaw']);mesh.apply_transform(transform)
    if not mesh.is_volume or not mesh.is_watertight:
        raise ValueError('Expected a watertight positive-volume building')
    scene['stair_structure']=dict(style='ramp',railings=False,transition_length_m=.12,
        source_style=original.get('stair_structure',{}).get('style'))
    scene['config'].update(name=scene['name'],stair_structure=deepcopy(scene['stair_structure']))
    scene['config']['floor_features']['railings']=False
    scene['floor_feature_options']['railings']=False
    for item in scene.get('floor_features',[]):
        item['railings']=False
    for q in scene['queries']:
        if q.get('required_transition')=='stairs':
            q['required_transition']='ramp'
    for edge in scene['floor_graph']['edges']:
        edge['type']='ramp'
    for edge in scene.get('room_graph',{}).get('edges',[]):
        if edge.get('kind')=='stairs':edge['kind']='ramp'
    scene['geometry_backend']='Saved building solids + continuous navigation ramp prisms; no railings'
    scene['source_scene']=str(source.resolve())
    scene['navigation_simplification']=dict(version=1,
        removed_counts=dict(Counter(s['kind'] for s in original['solids'] if s['kind'] in REMOVED)),
        source_scene_sha256=sha256(source/'scene.json'),source_mesh_sha256=sha256(source/'terrain.obj'),
        fixed=['seed','floors','walls','partitions','floor holes','bridge','flight widths','flight endpoints','world frame'],
        representation='C1 analytic profile tessellated at <=2.5cm, shared visual/collision geometry',
        scope='navigation-only simplified terrain; not a stair locomotion benchmark')
    # The new ramp is steeper than the old map's 0.4rad slope limit. Align only
    # this derived map's planner classification with the unchanged M0 0.6rad cap.
    scene['config']['vst']['slope_max']=.6
    scene['navigation_simplification']['map_slope_limit_rad']=[original['config']['vst']['slope_max'],.6]
    return scene,mesh,retained


def fixed_training_query(scene, run):
    """Keep the actual single-route experiment's world XY endpoints and yaw."""
    sys.path.insert(0,str(ROOT/'isaac_training/src'))
    from isaac_nav.data.registry import make_library
    from stair_nav.scenes.geometry import support_heights,to_local
    cfg=read_json(run/'config.resolved.json')
    lib=make_library(ROOT/'isaac_training',cfg,False)
    route=lib.route(lib.tasks[0]);a,b=cfg['overfit']['window_s_m']
    anchors=route.at([a,b]);resolved=anchors.copy()
    for i,p in enumerate(anchors):
        local=to_local(p,scene);heights=support_heights(*local[:2],scene)
        if not heights:raise ValueError('Source endpoint lost physical support')
        resolved[i,2]=min(heights,key=lambda h:abs(h-p[2]))
    q=deepcopy(next(q for q in scene['queries'] if q['id']=='ascent'))
    q.update(id='single_ascent',start_support_xyz=resolved[0].tolist(),goal_support_xyz=resolved[1].tolist(),
             start_yaw=cfg['overfit']['initial_yaw_rad'],required_transition='ramp')
    scene['queries'].append(q)
    scene['navigation_simplification'].update(fixed_training_query='single_ascent',
        source_run=str(run.resolve()),source_window_xyz=anchors.tolist(),
        fixed_initial_yaw=cfg['overfit']['initial_yaw_rad'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fixed-run',type=Path,help='Preserve this stopped run\'s active ascent endpoints')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    scene,mesh,retained=derive(args.source)
    if args.fixed_run:fixed_training_query(scene,args.fixed_run)
    args.output.mkdir(parents=True)
    mesh.export(args.output/'terrain.obj')
    p=sample_surface(mesh,scene['config']['sample_spacing']);p=p[p[:,2]>=-1e-6]
    write_pcd(args.output/'terrain.pcd',p)
    scene.update(point_count=len(p),pcd_sha256=sha256(args.output/'terrain.pcd'),
                 bounds_world=mesh.bounds.tolist(),mesh_watertight=True,pcd_ground_clip_z=0.)
    write_json(args.output/'scene.json',scene);write_json(args.output/'config.resolved.json',scene['config'])
    write_json(args.output/'simplification.json',dict(scene['navigation_simplification'],
        original_preserved=True,retained_solids=len(retained),new_solids=len(scene['solids']),
        rail_count=sum(s['kind'] in REMOVED-{'stair','landing','stair_stringer'} for s in scene['solids']),
        mesh_sha256=sha256(args.output/'terrain.obj'),vertices=len(mesh.vertices),triangles=len(mesh.faces),
        ramp_max_slopes_deg=[float(np.rad2deg(s['maximum_slope_rad'])) for s in scene['stair_flights']]))
    print(json.dumps(dict(output=str(args.output),vertices=len(mesh.vertices),triangles=len(mesh.faces),
        source_untouched=True,railings=0),ensure_ascii=False))


if __name__=='__main__':main()
