#!/usr/bin/env python3
"""Make a same-view synthetic RGB / GT cost / GT height image. No Isaac process is launched."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(WORKSPACE/'src'),str(WORKSPACE/'isaac_training/src')]


from ptv3_tomography.paths import relocated_path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--frame',type=int,default=52)
    parser.add_argument('--view',choices=['sensor','overview','all-points'],default='overview')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import trimesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image
    from isaac_nav.visualization.gpu_renderer import GPUCameraRenderer
    from ptv3_tomography.io import atomic_json,file_sha256,read_json
    manifest=read_json(args.maps/'manifest.json')
    cfg=manifest['map_config']
    row=next(row for row in manifest['frames'] if row['frame_index']==args.frame)
    path=args.maps/row['file']
    with np.load(path,allow_pickle=False) as a:
        data={key:a[key].copy() for key in a.files}
    dataset=relocated_path(manifest['source_dataset'])
    spec=read_json(dataset/'dataset_spec.json')
    scene_dir=relocated_path(spec['scene_directory'])
    if not scene_dir.is_absolute():
        scene_dir=dataset/scene_dir
    mesh_path=scene_dir/'terrain.obj'
    mesh_hash=file_sha256(mesh_path)
    expected=next(digest for name,digest in spec['source_fingerprints'].items() if Path(name).name=='terrain.obj')
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    scene=read_json(scene_dir/'scene.json')
    colors=np.asarray(mesh.visual.face_colors[:, :3],dtype=float)
    band=int(np.argmax(data['mask_cost'].sum(axis=(1,2))))
    tf=data['world_from_sensor']; up_sensor=data['gravity_up_sensor']

    def ground_points(mask):
        b,x,y=np.nonzero(mask)
        xx=(x-cfg['grid_size_xy'][0]//2)*cfg['resolution_m']
        yy=(y-cfg['grid_size_xy'][1]//2)*cfg['resolution_m']
        hh=data['gt_ground_height_gravity_m'][mask]
        zz=(hh-up_sensor[0]*xx-up_sensor[1]*yy)/up_sensor[2]
        return np.column_stack((xx,yy,zz))@tf[:3,:3].T+tf[:3,3]

    masks={name:data[f'mask_{name}'].copy() for name in ['cost','height']}
    for mask in masks.values():
        mask[np.arange(mask.shape[0])!=band]=False
    cost_points=ground_points(masks['cost'])
    if args.view=='sensor':
        origin=tf[:3,3]
        forward,right,up=tf[:3,0],-tf[:3,1],tf[:3,2]
    else:
        target=cost_points.mean(0)+[0,0,.08]
        origin=target+[2.3,-2.8,2.3]
        forward=target-origin;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
    width,height=960,640
    hfov,vfov=np.deg2rad([90,65])
    k={'fx':float(width/(2*np.tan(hfov/2))),'fy':float(height/(2*np.tan(vfov/2))),
       'cx':(width-1)/2,'cy':(height-1)/2}
    if args.view=='all-points':
        world_cloud=data['points_sensor_m']@tf[:3,:3].T+tf[:3,3]
        target=(world_cloud.min(0)+world_cloud.max(0))/2
        view_back=np.array([2.3,-2.8,2.3],dtype=float)
        view_back/=np.linalg.norm(view_back)
        forward=-view_back
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        # Fit both the real point cloud and reference cells, with a 10% image margin.
        fit_points=np.vstack([world_cloud,ground_points(masks['height']),cost_points])
        delta=fit_points-target
        along=delta@forward
        tan_h=width/(2*k['fx']);tan_v=height/(2*k['fy'])
        distance=max(1.,float(np.max(np.abs(delta@right)/(.9*tan_h)-along)),
                       float(np.max(np.abs(delta@up)/(.9*tan_v)-along)),float(-along.min()+.2))
        origin=target+view_back*distance
        check=world_cloud-origin
        z=check@forward
        uv=np.column_stack((k['fx']*(check@right)/z+k['cx'],k['cy']-k['fy']*(check@up)/z))
    stream={'width':width,'height':height,'near_m':.01,'far_m':30.,'intrinsics':k}
    renderer=GPUCameraRenderer(mesh,colors,{'rgb':stream,'depth':stream})
    try:
        rgb,depth=renderer.render(origin,forward,right,up)
        renderer_info=dict(renderer.info)
    finally:
        renderer.close()

    def project(points):
        delta=points-origin
        z=delta@forward
        safe=np.maximum(z,1e-8)
        uv=np.column_stack((k['fx']*(delta@right)/safe+k['cx'], k['cy']-k['fy']*(delta@up)/safe))
        visible=(z>.01)&(z<30)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        ids=np.flatnonzero(visible)
        pix=np.rint(uv[ids]).astype(int)
        pix[:,0]=np.clip(pix[:,0],0,width-1);pix[:,1]=np.clip(pix[:,1],0,height-1)
        scene_depth=depth[pix[:,1],pix[:,0]]
        visible[ids]&=(scene_depth>0)&(z[ids]<=scene_depth+.08)
        return uv,visible

    cost_cmap=LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(args.output.with_name(args.output.stem+'_rgb.png'))
    lo,hi=cfg['height_bin_edges_m'][band:band+2]
    visible_counts={}
    visible_indices={}
    with plt.rc_context({'font.size':12,'axes.titlesize':15,'figure.facecolor':'#f8fafc'}):
        fig,axes=plt.subplots(1,3,figsize=(17,6.2),constrained_layout=True)
        for ax in axes:
            ax.imshow(rgb);ax.set_xticks([]);ax.set_yticks([])
        axes[0].set_title('RGB: same source mesh')
        for ax,name,title,cmap,vmin,vmax in [
            (axes[1],'cost','GT cost on RGB',cost_cmap,0,60),
            (axes[2],'height','GT ground height on RGB','viridis',lo,hi)]:
            points=ground_points(masks[name]);uv,visible=project(points)
            values=data['gt_cost_raw' if name=='cost' else 'gt_ground_height_gravity_m'][masks[name]]
            visible_counts[name]=int(visible.sum())
            visible_indices[name]=np.flatnonzero(visible).tolist()
            plotted=ax.scatter(uv[visible,0],uv[visible,1],c=values[visible],s=30,
                               cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='#17202e',linewidths=.35)
            ax.set_title(title)
            fig.colorbar(plotted,ax=ax,fraction=.045,pad=.025,label='cost 0..60' if name=='cost' else 'gravity-relative height (m)')
        fig.suptitle(f'Frame {args.frame} | same camera view | GT gravity-height band [{lo:g}, {hi:g}) m\nSynthetic RGB for geometric context. Colored points are valid Tomography supervision only.',fontsize=16,fontweight='bold')
        fig.savefig(args.output,dpi=140)
        plt.close(fig)
    atomic_json(args.output.with_suffix('.json'),{'status':'passed','frame_index':args.frame,'band':band,
        'view':args.view,'rgb_is_synthetic_mesh_render':True,'real_rgb_capture_available':False,
        'renderer':renderer_info,'mesh_sha256':mesh_hash,'visible_gt_cells':visible_counts,
        'visible_gt_cell_indices':visible_indices,
        'camera':{'origin':origin.tolist(),'forward':forward.tolist(),'right':right.tolist(),'up':up.tolist(),'intrinsics':k},
        'same_view_rgb_gt_projection':True,'occlusion_tolerance_m':.08,'source_mesh_modified':False,'isaac_launched':False})
    print('RGB_GT',args.output,visible_counts,renderer_info,flush=True)


if __name__=='__main__':
    main()
