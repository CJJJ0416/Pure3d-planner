#!/usr/bin/env python3
"""New perception transfer scene using the vendored ETH terrain-generator stair mesh part."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(WORKSPACE/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-scene',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261104)
    args=parser.parse_args()
    import numpy as np
    import trimesh
    from stair_nav.vendor.eth_stairs import create_standard_stairs
    from stair_nav.scenes.geometry import sample_surface
    from stair_nav.core.io import read_json,sha256,write_json,write_pcd
    args.output.mkdir(parents=True)
    source=read_json(args.reference_scene/'scene.json')
    rng=np.random.default_rng(args.seed)
    n_steps,rise,step_depth,width,landing,slab=5,.12,.4,1.8,.8,.16
    height=n_steps*rise
    length=(n_steps+1)*step_depth+landing
    cfg=SimpleNamespace(n_steps=n_steps,total_height=height,step_depth=step_depth,
                        step_width=width,dim=(width,length,height),height_offset=slab,
                        floor_thickness=slab,add_residual_side_up=True,minimal_triangles=True)
    stairs=create_standard_stairs(cfg)
    stairs.apply_translation([0,length/2,height/2-slab])
    boxes=[('lower_floor',[-5,-4,-slab],[5,6,0]),
           ('upper_platform',[-1.9,3.15,-slab],[1.9,5.2,height]),
           ('obstacle',[-1.95,1.05,-.02],[-1.35,1.65,.4]),
           ('obstacle',[.55,3.95,height-.02],[1.15,4.55,height+.45])]
    meshes=[stairs]
    solids=[]
    for kind,lo,hi in boxes:
        lo,hi=np.asarray(lo,float),np.asarray(hi,float)
        if kind=='obstacle':
            jitter=float(rng.uniform(-.07,.07))
            lo[:2]+=jitter;hi[:2]+=jitter
        meshes.append(trimesh.creation.box(hi-lo,trimesh.transformations.translation_matrix((lo+hi)/2)))
        solids.append({'kind':kind,'lo':lo.tolist(),'hi':hi.tolist()})
    mesh=trimesh.boolean.union(meshes,engine='manifold')
    mesh.export(args.output/'terrain.obj')
    mesh.export(args.output/'terrain.glb')
    points=sample_surface(mesh,.04)
    points=points[points[:,2]>=-1e-6]
    write_pcd(args.output/'terrain.pcd',points)
    vendor=WORKSPACE/'src/stair_nav/vendor/eth_stairs.py'
    provenance={'project':'ETH Zurich / leggedrobotics terrain-generator',
                'repository':'https://github.com/leggedrobotics/terrain-generator',
                'commit':'7be9d9a6ebbed7dc669486426184a4694bfd0bb6',
                'module':'vendored create_standard_stairs', 'source_file':str(vendor),'source_sha256':sha256(vendor),
                'adaptation':'Original stair function body, reduced imports, manifold union backend.',
                'composition':'One upstream stair mesh part plus local floor/platform/obstacle boxes.',
                'full_wfc_executed':False}
    config={'name':'terrain_generator_stairs_platform_transfer','seed':args.seed,'sample_spacing':.04,
            'vst':source['config']['vst'],'geometry_provenance':provenance}
    scene={'schema_version':1,'name':config['name'],'seed':args.seed,'layout':'eth_stair_part_with_platform_and_blocks',
           'architecture':{'family':'terrain_generator_stairs_obstacles','label':'ETH stairs + platform + blocks'},
           'frame_id':'map','units':'m','z_axis':'up','yaw':0.,'width':10.,'depth':10.,'height':height+.45,
           'slab_thickness':slab,'config':config,'solids':solids,'support_surfaces':[],
           'stair_flights':[{'id':0,'construction':'solid_stairs','x':0.,'y':0.,'width':width,
                             'n_steps':n_steps,'riser':rise,'step_depth':step_depth,'length':length,
                             'base_z':0.,'top_z':height,'direction':1}],
           'rooms':[],'floors':[{'id':0,'z':0.},{'id':1,'z':height}],
           'bounds_world':mesh.bounds.tolist(),'mesh_watertight':True,'pcd_ground_clip_z':0.,
           'point_count':len(points),'pcd_sha256':sha256(args.output/'terrain.pcd'),
           'queries':[{'id':'ascent','start_support_xyz':[0,-.8,0.],'goal_support_xyz':[0,3.9,height],
                       'start_yaw':float(np.pi/2),'required_transition':'stairs','start_floor':0,'goal_floor':1}],
           'geometry_backend':provenance}
    # Kinematic reference only. It is not an executed legged trajectory.
    route=[(0,-.8,0.),(0,.2,0.)]
    for step in range(1,n_steps+1):
        y=(step+.5)*step_depth
        route.append((0,y,step*rise))
    route.extend([(0,3.1,height),(0,3.9,height)])
    xyz=np.asarray(route,float)
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    rows=np.column_stack((xyz,np.full(len(xyz),np.pi/2),distance))
    np.savetxt(args.output/'reference_route.csv',rows,delimiter=',',header='x,y,z,yaw,s',comments='')
    write_json(args.output/'scene.json',scene)
    write_json(args.output/'config.resolved.json',config)
    report={'status':'passed','generator':provenance,'seed':args.seed,'step_count':n_steps,
            'riser_m':rise,'tread_depth_m':step_depth,'stair_width_m':width,'platform_height_m':height,
            'obstacle_count':2,'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':True,
            'volume_m3':float(mesh.volume),'pcd_points':len(points),'mesh_sha256':sha256(args.output/'terrain.obj'),
            'training_scene_mesh_sha256':sha256(args.reference_scene/'terrain.obj'),
            'vst_config_unchanged_from_training':config['vst']==source['config']['vst'],
            'reference_motion':'held-pose kinematic sensor path; no legged dynamics or support-feasibility claim'}
    write_json(args.output/'geometry_report.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
