#!/usr/bin/env python3
"""Same RGB/GT reference with inverse-restored features and all raw LiDAR point anchors."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


from ptv3_tomography.paths import relocated_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restoration',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gt-on-lidar',action='store_true',
                        help='Query original global VST at every raw LiDAR endpoint, without training masks')
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image
    from ptv3_tomography.io import read_json,atomic_json,file_sha256
    source = read_json(args.restoration/'report.json')
    reference = read_json(args.reference.with_suffix('.json'))
    index,band = reference['frame_index'],reference['band']
    row = next(r for r in source['frames_manifest'] if r['frame_index']==index)
    shard = args.restoration/row['file']
    with np.load(shard,allow_pickle=False) as a:
        restored = {k:a[k].copy() for k in a.files}
    maps = relocated_path(source['source_maps'])
    manifest = read_json(maps/'manifest.json')
    map_row = next(r for r in manifest['frames'] if r['frame_index']==index)
    map_file = maps/map_row['file']
    with np.load(map_file,allow_pickle=False) as a:
        data = {k:a[k].copy() for k in a.files}
    cfg = manifest['map_config']
    camera = reference['camera']; k = camera['intrinsics']
    origin,forward,right,up = [np.asarray(camera[key]) for key in ['origin','forward','right','up']]
    rgb_file = args.reference.with_name(args.reference.stem+'_rgb.png')
    rgb = np.asarray(Image.open(rgb_file).convert('RGB'))
    rgb_hash = file_sha256(rgb_file)
    tf = restored['world_from_sensor']

    def project(world):
        delta = world-origin
        z = delta@forward
        uv = np.column_stack((k['fx']*(delta@right)/np.maximum(z,1e-8)+k['cx'],
                              k['cy']-k['fy']*(delta@up)/np.maximum(z,1e-8)))
        visible = (z>.01)&(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])&(uv[:,1]>=0)&(uv[:,1]<rgb.shape[0])
        return uv,z,visible

    # All real return coordinates are retained. No teacher mask or camera-depth mask is used here.
    world = restored['points_sensor_m']@tf[:3,:3].T+tf[:3,3]
    point_uv,point_depth,in_view = project(world)
    inside = restored['within_map_domain']
    plot_order = np.flatnonzero(in_view)[np.argsort(point_depth[in_view])[::-1]]
    gt = {}
    metrics = {}
    for name in ['cost','height']:
        mask = data[f'mask_{name}'].copy()
        mask[np.arange(len(mask))!=band] = False
        b,x,y = np.nonzero(mask)
        xx = (x-cfg['grid_size_xy'][0]//2)*cfg['resolution_m']
        yy = (y-cfg['grid_size_xy'][1]//2)*cfg['resolution_m']
        hh = data['gt_ground_height_gravity_m'][mask]
        gravity = data['gravity_up_sensor']
        zz = (hh-gravity[0]*xx-gravity[1]*yy)/gravity[2]
        uv,_,visible = project(np.column_stack((xx,yy,zz))@tf[:3,:3].T+tf[:3,3])
        if 'visible_gt_cell_indices' in reference:
            selected = np.asarray(reference['visible_gt_cell_indices'][name],dtype=int)
        else:
            selected = np.arange(len(uv))
        key = 'cost_raw' if name=='cost' else 'ground_height_gravity_m'
        values = data['gt_'+key][mask][selected]
        predictions = data[key][mask][selected]
        gt[name] = {'uv':uv[selected],'values':values}
        metrics[name] = {'cells':len(selected),'mae':float(np.abs(values-predictions).mean())}
    full_gt = None
    point_metrics = None
    vst_hash = None
    if args.gt_on_lidar:
        from ptv3_tomography.data.tomography_pairs import load_tomography
        dataset = relocated_path(source['source_dataset'])
        spec = read_json(dataset/'dataset_spec.json')
        scene = relocated_path(spec['scene_directory'])
        if not scene.is_absolute():
            scene = dataset/scene
        payload,vst,_,vst_meta = load_tomography(scene)
        vst_hash = vst_meta['tomogram_sha256']
        n = len(world)
        native_xy = np.rint((world[:,:2]-payload['center'])/payload['resolution']).astype(np.int64)
        native_xy += np.asarray(vst.shape[2:])//2
        in_global = ((native_xy>=0)&(native_xy<np.asarray(vst.shape[2:]))).all(1)
        ground = np.full((vst.shape[1],n),np.nan,np.float64)
        costs = np.full_like(ground,np.nan)
        ids = np.flatnonzero(in_global)
        for layer in range(vst.shape[1]):
            ground[layer,ids] = vst[3,layer,native_xy[ids,0],native_xy[ids,1]]
            costs[layer,ids] = vst[0,layer,native_xy[ids,0],native_xy[ids,1]]
        # Choose by the point's gravity-height band, never collapse different floors by XY alone.
        point_band = np.searchsorted(cfg['height_bin_edges_m'],world[:,2]-tf[2,3],side='right')-1
        ground_band = np.searchsorted(cfg['height_bin_edges_m'],ground-tf[2,3],side='right')-1
        candidate = (np.isfinite(ground)&np.isfinite(costs)&(costs>=0)&(costs<=cfg['cost_scale'])
                     &(ground_band==point_band[None,:]))
        available = candidate.any(0)
        low = np.min(np.where(candidate,ground,np.inf),axis=0)
        high = np.max(np.where(candidate,ground,-np.inf),axis=0)
        ambiguous = available&((high-low)>cfg['duplicate_height_tolerance_m'])
        available &= ~ambiguous
        # Same-height duplicate slices use the existing canonical minimum-cost tie break.
        selected_layer = np.argmin(np.where(candidate,costs,np.inf),axis=0)
        gt_cost = np.where(available,costs[selected_layer,np.arange(n)],np.nan).astype(np.float32)
        gt_height = np.where(available,ground[selected_layer,np.arange(n)]-tf[2,3],np.nan).astype(np.float32)
        source_layer = np.where(available,selected_layer,-1).astype(np.int16)
        common = available&inside
        point_metrics = {}
        for name,values,pred_key in [('cost',gt_cost,'sampled_map_cost_raw'),
                                      ('height',gt_height,'sampled_map_height_gravity_m')]:
            point_metrics[name] = {'points':int(common.sum()),
                                  'mae':float(np.abs(values[common]-restored[pred_key][common]).mean()) if common.any() else None}
            gt[name] = {'uv':point_uv,'values':values,'available':available}
        full_gt = {'cost_raw':gt_cost,'ground_height_gravity_m':gt_height,'available':available,
                   'ambiguous':ambiguous,'source_layer':source_layer,'source_grid_xy':native_xy,
                   'point_height_band':point_band}
    heights = np.r_[restored['sampled_map_height_gravity_m'][inside],gt['height']['values']]
    heights = heights[np.isfinite(heights)]
    height_lo = float(np.floor(heights.min()/.5)*.5) if len(heights) else cfg['height_bin_edges_m'][0]
    height_hi = float(np.ceil(heights.max()/.5)*.5) if len(heights) else cfg['height_bin_edges_m'][-1]
    if height_hi<=height_lo:
        height_hi = height_lo+.5
    cost_cmap = LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    n,m,c = row['valid_returns'],row['voxel_tokens'],row['restored_feature_shape'][1]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':12,'axes.titlesize':14,'figure.facecolor':'#f8fafc'}):
        fig,axes = plt.subplots(2,3,figsize=(18,11.3),layout='constrained')
        for ax in axes.flat:
            ax.imshow(rgb);ax.set_xticks([]);ax.set_yticks([])
        axes[0,0].set_title(f'RGB: original camera | {n} valid LiDAR returns')
        axes[1,0].scatter(point_uv[plot_order,0],point_uv[plot_order,1],
                          c=restored['restored_feature_rgb'][plot_order],s=23,edgecolors='#17202e',linewidths=.3)
        axes[1,0].set_title(f'Restored PTv3 features: {m} x {c} -> {n} x {c}\nPCA feature colors')
        for col,name in [(1,'cost'),(2,'height')]:
            cmap = cost_cmap if name=='cost' else 'viridis'
            vmin,vmax = (0,60) if name=='cost' else (height_lo,height_hi)
            uv,values = gt[name]['uv'],gt[name]['values']
            if args.gt_on_lidar:
                available = gt[name]['available']
                unknown_gt = in_view&~available
                axes[0,col].scatter(point_uv[unknown_gt,0],point_uv[unknown_gt,1],c='#94a3b8',s=24,
                                   edgecolors='#17202e',linewidths=.3)
                selected = plot_order[available[plot_order]]
                uv,values = uv[selected],values[selected]
            im = axes[0,col].scatter(uv[:,0],uv[:,1],c=values,s=24 if args.gt_on_lidar else 30,cmap=cmap,vmin=vmin,vmax=vmax,
                                     edgecolors='#17202e',linewidths=.35)
            if args.gt_on_lidar:
                axes[0,col].set_title(f'VST GT {name} at original LiDAR points\n{len(values)} colored + {int(unknown_gt.sum())} gray | no training-mask filter')
            else:
                axes[0,col].set_title(f'GT {name}: {len(values)} supervised cells\nOriginal reference band')
            unknown = in_view&~inside
            axes[1,col].scatter(point_uv[unknown,0],point_uv[unknown,1],c='#94a3b8',s=24,
                               edgecolors='#17202e',linewidths=.3)
            order = plot_order[inside[plot_order]]
            key = 'sampled_map_cost_raw' if name=='cost' else 'sampled_map_height_gravity_m'
            axes[1,col].scatter(point_uv[order,0],point_uv[order,1],c=restored[key][order],s=24,
                               cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='#17202e',linewidths=.3)
            axes[1,col].set_title(f'Predicted map {name} at original LiDAR points\n{int((in_view&inside).sum())} colored + {int(unknown.sum())} gray | no GT filtering')
            for ax in [axes[0,col],axes[1,col]]:
                fig.colorbar(im,ax=ax,fraction=.045,pad=.02,
                             label='cost 0..60' if name=='cost' else 'gravity-relative ground height (m)')
        top_description = ('Top: RGB and VST GT at LiDAR positions. Bottom: restored features and predictions at the SAME positions.'
                           if args.gt_on_lidar else 'Top: original RGB and GT cells. Bottom: restored features and map predictions at LiDAR positions.')
        fig.suptitle(f'Frame {index} | point-restored visualization | {n} / {n} valid LiDAR returns retained\n'
                     +top_description,
                     fontsize=18,fontweight='bold')
        if args.gt_on_lidar and point_metrics['cost']['points']==0:
            metric_caption = 'No shared queryable GT/prediction points. Original anchors are retained.'
        else:
            metric_caption = (f"Endpoint-lookup comparison on {point_metrics['cost']['points']} shared points: "
                          f"cost MAE {point_metrics['cost']['mae']:.2f}/60, height MAE {point_metrics['height']['mae']*100:.2f} cm."
                          if args.gt_on_lidar else
                          f"Unchanged original-cell metrics: cost MAE {metrics['cost']['mae']:.2f}/60 ({metrics['cost']['cells']} cells), "
                      f"height MAE {metrics['height']['mae']*100:.2f} cm ({metrics['height']['cells']} cells).\n"
                              )
        gt_caption = ('GT is a direct global-VST lookup by world XY and height band. Strict training masks are unchanged.\n'
                      if args.gt_on_lidar else '')
        fig.supxlabel(metric_caption+'\n'+gt_caption+
                      'Feature PCA colors are not cost. Predicted colors sample the existing map, with original XYZ retained.\n'
                      'Gray: outside map domain. Point projections are shown through the RGB context without depth filtering. RGB is synthetic.',
                      fontsize=11,color='#334155')
        fig.savefig(args.output,dpi=150)
        plt.close(fig)
    if full_gt is not None:
        np.savez_compressed(args.output.with_name(args.output.stem+'_point_values.npz'),
                            points_sensor_m=restored['points_sensor_m'],points_world_m=world,
                            valid_ray_indices=restored['valid_ray_indices'],
                            gt_cost_raw=full_gt['cost_raw'],gt_ground_height_gravity_m=full_gt['ground_height_gravity_m'],
                            gt_available=full_gt['available'],gt_ambiguous=full_gt['ambiguous'],
                            gt_source_layer=full_gt['source_layer'],gt_source_grid_xy=full_gt['source_grid_xy'],
                            point_height_band=full_gt['point_height_band'],endpoint_near_ground=full_gt['endpoint_near_ground'],
                            prediction_available=inside,predicted_cost_raw=restored['sampled_map_cost_raw'],
                            predicted_ground_height_gravity_m=restored['sampled_map_height_gravity_m'])
    atomic_json(args.output.with_suffix('.json'),{'status':'passed','frame_index':index,
                'reference_camera_unchanged':True,'original_rgb_sha256':rgb_hash,
                'reference_image_sha256':file_sha256(args.reference),'restoration_shard_sha256':row['sha256'],
                'checkpoint_sha256':source['checkpoint_sha256'],'valid_lidar_points':n,'voxel_tokens':m,
                'restored_feature_shape':[n,c],'point_projections_inside_reference_camera':int(in_view.sum()),
                'prediction_colored_points':int((inside&in_view).sum()),'prediction_gray_points':int((~inside&in_view).sum()),
                'prediction_gt_mask_applied':False,'point_depth_occlusion_mask_applied':False,
                'point_xyz_unchanged':True,'pca_feature_colors_are_not_cost':True,
                'cost_height_values':'nearest fixed map cell lookup at each LiDAR endpoint; no separate point head',
                'gt_reference_band':band,'height_color_limits_m':[height_lo,height_hi],
                'original_gt_cell_metrics':metrics,'original_image_modified':False,'new_training_executed':False,
                'gt_on_all_lidar_anchors':args.gt_on_lidar,
                'gt_lookup_source':'global VST world XY + matching gravity height band' if args.gt_on_lidar else 'original masked grid cells',
                'gt_colored_points':int((full_gt['available']&in_view).sum()) if full_gt is not None else None,
                'gt_gray_points':int((~full_gt['available']&in_view).sum()) if full_gt is not None else None,
                'gt_ambiguous_points':int(full_gt['ambiguous'].sum()) if full_gt is not None else None,
                'endpoint_near_ground_points':int(full_gt['endpoint_near_ground'].sum()) if full_gt is not None else None,
                'gt_source_vst_sha256':vst_hash,'strict_training_masks_modified':False,
                'point_lookup_metrics':point_metrics,
                'output_sha256':file_sha256(args.output)})
    print('RGB_POINT_RESTORATION',args.output,{'points':n,'colored':int((inside&in_view).sum()),'gray':int((~inside&in_view).sum())},flush=True)


if __name__ == '__main__':
    main()
