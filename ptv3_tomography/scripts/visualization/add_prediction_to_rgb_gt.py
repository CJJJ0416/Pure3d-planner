#!/usr/bin/env python3
"""Replot cached quantitative PTv3 outputs below an existing RGB/GT comparison."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image
    from ptv3_tomography.io import read_json,atomic_json,file_sha256
    reference=read_json(args.reference.with_suffix('.json'))
    frame,band=reference['frame_index'],reference['band']
    rgb_file=args.reference.with_name(args.reference.stem+'_rgb.png')
    rgb=np.asarray(Image.open(rgb_file).convert('RGB'))
    manifest=read_json(args.maps/'manifest.json')
    record=next(row for row in manifest['frames'] if row['frame_index']==frame)
    data_file=args.maps/record['file']
    with np.load(data_file,allow_pickle=False) as a:
        data={key:a[key].copy() for key in a.files}
    cfg=manifest['map_config'];lo,hi=cfg['height_bin_edges_m'][band:band+2]
    camera=reference['camera'];k=camera['intrinsics']
    origin,forward,right,up=[np.asarray(camera[name]) for name in ['origin','forward','right','up']]
    tf,gravity=data['world_from_sensor'],data['gravity_up_sensor']
    projections={}; masks={}; metrics={}
    for name in ['cost','height']:
        mask=data[f'mask_{name}'].copy()
        mask[np.arange(len(mask))!=band]=False
        masks[name]=mask
        b,x,y=np.nonzero(mask)
        xx=(x-cfg['grid_size_xy'][0]//2)*cfg['resolution_m']
        yy=(y-cfg['grid_size_xy'][1]//2)*cfg['resolution_m']
        hh=data['gt_ground_height_gravity_m'][mask]
        zz=(hh-gravity[0]*xx-gravity[1]*yy)/gravity[2]
        world=np.column_stack((xx,yy,zz))@tf[:3,:3].T+tf[:3,3]
        delta=world-origin;depth=delta@forward
        uv=np.column_stack((k['fx']*(delta@right)/depth+k['cx'],k['cy']-k['fy']*(delta@up)/depth))
        visible=(depth>.01)&(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])&(uv[:,1]>=0)&(uv[:,1]<rgb.shape[0])
        # Reuse the reference depth-buffer visibility decision at exactly the same anchors.
        if 'visible_gt_cell_indices' in reference:
            selected=np.asarray(reference['visible_gt_cell_indices'][name],dtype=int)
        else:
            selected=np.arange(len(uv))
        selected_mask=np.zeros(len(uv),dtype=bool)
        selected_mask[selected]=True
        mask[mask]=selected_mask
        projections[name]=uv[selected]
        pred=data['cost_raw' if name=='cost' else 'ground_height_gravity_m'][mask]
        gt=data['gt_cost_raw' if name=='cost' else 'gt_ground_height_gravity_m'][mask]
        metrics[name]={'cells':len(gt),'mae':float(np.abs(pred-gt).mean())}
    cost_cmap=LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':12,'axes.titlesize':15,'figure.facecolor':'#f8fafc'}):
        fig,axes=plt.subplots(2,3,figsize=(17,10.5),constrained_layout=True)
        axes[0,0].imshow(rgb);axes[0,0].set_title('RGB: original camera and source mesh')
        axes[0,0].set_xticks([]);axes[0,0].set_yticks([])
        for row in [0,1]:
            for col,name in [(1,'cost'),(2,'height')]:
                ax=axes[row,col];ax.imshow(rgb);ax.set_xticks([]);ax.set_yticks([])
                key=('gt_cost_raw' if name=='cost' else 'gt_ground_height_gravity_m') if row==0 else ('cost_raw' if name=='cost' else 'ground_height_gravity_m')
                values=data[key][masks[name]];uv=projections[name]
                im=ax.scatter(uv[:,0],uv[:,1],c=values,s=30,cmap=cost_cmap if name=='cost' else 'viridis',
                              vmin=0 if name=='cost' else lo,vmax=60 if name=='cost' else hi,
                              edgecolors='#17202e',linewidths=.35)
                ax.set_title(('GT ' if row==0 else 'PTv3 predicted ')+('cost on RGB' if name=='cost' else 'ground height on RGB'))
                fig.colorbar(im,ax=ax,fraction=.045,pad=.025,label='cost 0..60' if name=='cost' else 'gravity-relative height (m)')
        card=axes[1,0];card.set_axis_off()
        card.add_patch(FancyBboxPatch((.02,.03),.96,.91,boxstyle='round,pad=0.018',transform=card.transAxes,
                                     facecolor='#eaf0f8',edgecolor='#cbd5e1'))
        card.text(.07,.84,'Same-cell comparison',transform=card.transAxes,fontsize=19,fontweight='bold',color='#0f172a')
        card.text(.07,.64,f"Cost MAE  {metrics['cost']['mae']:.2f} / 60\nHeight MAE  {metrics['height']['mae']*100:.2f} cm",transform=card.transAxes,
                  fontsize=20,fontweight='bold',color='#1d4ed8',linespacing=1.6)
        card.text(.07,.40,f"Cost cells: {metrics['cost']['cells']}    Height cells: {metrics['height']['cells']}\nFull PTv3 checkpoint after one epoch",transform=card.transAxes,fontsize=13,color='#334155',linespacing=1.6)
        card.text(.07,.12,'Marker positions: the same GT surface anchors.\nMarker colors: GT or PTv3 predicted values.\nRGB: synthetic mesh render, not camera capture.',transform=card.transAxes,fontsize=12,color='#475569',linespacing=1.6)
        fig.suptitle(f'Frame {frame} | same view, same GT cells, same color scales | gravity height [{lo:g}, {hi:g}) m\nTop row: RGB and GT. Bottom row: trained PTv3 predictions.',fontsize=17,fontweight='bold')
        fig.savefig(args.output,dpi=140)
        plt.close(fig)
    atomic_json(args.output.with_suffix('.json'),{'status':'passed','frame_index':frame,'band':band,
                'reference_image':str(args.reference),'reference_sha256':file_sha256(args.reference),
                'rgb_sha256':file_sha256(rgb_file),'map_shard_sha256':record['sha256'],
                'checkpoint_sha256':manifest['checkpoint_sha256'],'camera':camera,
                'metrics_at_visible_gt_cells':metrics,'marker_positions':'shared GT surface anchors for scalar-value comparison',
                'prediction_values':'actual saved PTv3 cost/height, not redrawn or interpolated from GT',
                'same_color_scales':True,'isaac_launched':False,'source_images_modified':False})
    print('RGB_GT_PREDICTION',args.output,metrics)


if __name__=='__main__':
    main()
