"""Training curves and same-point GT / prediction / error / filtering plots."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils.pointcloud import read_json,workspace_path
from Filter.visualization.visualize import point_view,BACKGROUND


def training_curves(run):
    history=read_json(run/'history.json');epochs=[h['epoch'] for h in history]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    losses=np.array([h['loss'] for h in history])
    axes[0,0].plot(epochs,losses,color='#2463b5',alpha=.4,label='Epoch mean')
    if len(losses)>=5: axes[0,0].plot(epochs[4:],np.convolve(losses,np.ones(5)/5,mode='valid'),color='#2463b5',label='5-epoch mean')
    axes[0,0].set_title('Training SmoothL1 loss');axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel('Source / structure balanced loss');axes[0,0].set_yscale('log')
    val=[h for h in history if h['validation']]
    train=[h for h in history if h.get('training')]
    initial=read_json(run/'initial_metrics.json') if (run/'initial_metrics.json').exists() else None
    for axis,key,title in [(axes[0,1],'mae','All-point score MAE'),(axes[1,1],'macro_mae','Source-balanced macro MAE')]:
        for rows,field,label,color in [(train,'training','Train (eval mode)','#2463b5'),(val,'validation','Fixed validation','#e38828')]:
            x=[h['epoch'] for h in rows];y=[h[field][key] for h in rows]
            if initial is not None: x=[0]+x;y=[initial[field][key]]+y
            if x: axis.plot(x,y,label=label,color=color)
        axis.set_title(title);axis.set_ylabel('Error in keep_score [0, 1]');axis.set_yscale('log');axis.legend()
    names=sorted({name for h in val for name in h['validation']['groups']})
    for name in names:
        axes[1,0].plot([h['epoch'] for h in val],[h['validation']['groups'][name]['mae'] for h in val],label=name)
    axes[1,0].legend(fontsize=7);axes[1,0].set_title('Per-source fixed validation MAE')
    for ax in axes.flat: ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();out=run/'plots';out.mkdir(exist_ok=True);fig.savefig(out/'training.png',dpi=150);plt.close(fig)


def compare(folder,variant,prediction,output,threshold):
    folder=Path(folder);arrays=folder/f'arrays_{variant:03d}'
    coord=np.load(arrays/'coord.npy');target=np.load(arrays/'keep_target.npy');semantic=np.load(arrays/'semantic_id.npy')
    pred=np.load(prediction);valid=np.isfinite(pred)
    with np.load(folder/f'rough_attributes_{variant:03d}.npz') as attributes:
        roles=attributes['source_role'][valid]
    coord,target,pred,semantic=coord[valid],target[valid],pred[valid],semantic[valid]
    bounds=np.array([coord.min(0),coord.max(0)])
    fig=plt.figure(figsize=(16,12),facecolor=BACKGROUND)
    error=np.abs(pred-target)
    panels=[('Ground truth',target,np.ones(len(coord),bool),'RdYlGn'),
        ('Filter prediction',pred,np.ones(len(coord),bool),'RdYlGn'),
        ('Absolute error',error,np.ones(len(coord),bool),'magma'),
        (f'Predicted retained | threshold {threshold:g}',pred,pred>=threshold,'RdYlGn')]
    for j,(name,value,mask,cmap) in enumerate(panels):
        ax=fig.add_subplot(2,2,j+1,projection='3d')
        point_view(ax,coord[mask],semantic[mask],plt.get_cmap(cmap)(value[mask])[:,:3],bounds,
                   f'{name} | {mask.sum():,} points',groups=roles[mask],marker_size=3.)
    fig.suptitle(f'{folder.name} | variant {variant} | trained Filter\nSame XYZ, camera and core points. MAE {error.mean():.4f}',fontsize=15)
    fig.subplots_adjust(left=.01,right=.88,bottom=.04,top=.9,wspace=.05,hspace=.12)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='RdYlGn'),cax=fig.add_axes([.92,.57,.015,.28]),label='keep coefficient')
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='magma'),cax=fig.add_axes([.92,.15,.015,.28]),label='absolute error')
    fig.savefig(output,dpi=140,facecolor=BACKGROUND);plt.close(fig)


def real_prediction(folder,output):
    from common.pointcloud_io import read_pcd_fields,fields_xyz
    fields=read_pcd_fields(Path(folder)/'prediction.pcd')
    xyz,score=fields_xyz(fields),fields['keep_score']
    centered=xyz-xyz.mean(0)
    _,vectors=np.linalg.eigh(centered[:,:2].T@centered[:,:2])
    right=np.r_[vectors[:,-1],0.].astype(np.float32)
    if right[0]<0: right=-right
    side=np.array([-right[1],right[0],0.],np.float32)
    right=np.cos(np.deg2rad(35.))*right+np.sin(np.deg2rad(35.))*side
    horizontal=np.array([-right[1],right[0],0.],np.float32)
    elev=np.deg2rad(30.)
    up=-np.sin(elev)*horizontal+np.array([0,0,np.cos(elev)],np.float32)
    depth=np.cos(elev)*horizontal+np.array([0,0,np.sin(elev)],np.float32)
    cameras=[('Oblique',right,up,depth),('Top',np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]))]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for full_extent in [False,True]:
        for row,(name,r,u,d) in enumerate(cameras):
            fig,axes=plt.subplots(1,2,figsize=(16,7) if name=='Oblique' else (10,12),facecolor=BACKGROUND)
            projected=centered@np.stack([r,u,d],axis=1).astype(np.float32)
            q=[0.,1.] if full_extent else [.001,.999]
            bounds=np.quantile(projected[:,:2],q,axis=0)
            low,high=bounds
            visible=((projected[:,:2]>=low)&(projected[:,:2]<=high)).all(1)
            ids=np.flatnonzero(visible);uv=projected[ids,:2];z=projected[ids,2]
            width,height=1400,1000
            pixel=np.minimum(((uv-low)/(high-low)*[width-1,height-1]).astype(np.int64),[width-1,height-1])
            key=pixel[:,1]*width+pixel[:,0]
            order=np.lexsort((-z,key));sorted_key=key[order]
            first=np.r_[True,sorted_key[1:]!=sorted_key[:-1]]
            screen=sorted_key[first];original_ids=ids[order[first]]
            shade=.6+.4*(xyz[original_ids,2]-xyz[:,2].min())/np.ptp(xyz[:,2])
            raw=(shade[:,None]*np.array([[.25,.5,.82]])*255).astype(np.uint8)
            colored=(plt.get_cmap('RdYlGn')(score[original_ids])[:,:3]*255).astype(np.uint8)
            for column,(title,colors) in enumerate([('Original XYZ',raw),('Predicted keep_score',colored)]):
                raster=np.full((height*width,3),247,np.uint8);raster[screen]=colors
                ax=axes[column]
                ax.imshow(raster.reshape(height,width,3),origin='lower',extent=[low[0],high[0],low[1],high[1]])
                ax.set_title(f'{name}: {title} | {len(ids):,} points in view')
                ax.set_xlabel('Projected horizontal (m)');ax.set_ylabel('Projected vertical (m)')
            scope='full coordinate extent' if full_extent else 'main-cloud view (0.1%-99.9% bounds)'
            fig.suptitle(f'{Path(folder).name} | {len(xyz):,} original points scored | {scope}\n'
                'Same visible point IDs. Full point cloud retained. No voxel downsampling.',fontsize=14)
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='RdYlGn'),
                cax=fig.add_axes([.935,.23,.012,.52]),label='keep_score: low (red) to high (green)')
            fig.text(.5,.017,'All input points participate in depth projection. Real scan has no pointwise GT. No automatic point deletion.',
                ha='center',fontsize=9)
            fig.subplots_adjust(left=.055,right=.9,top=.86,bottom=.09,wspace=.18)
            suffix=('_top' if name=='Top' else '')+('_full_extent' if full_extent else '')
            path=output.with_name(output.stem+suffix+output.suffix)
            fig.savefig(path,dpi=150,facecolor=BACKGROUND);plt.close(fig)


def scoring_results(run,xyz_reference=None):
    """Full-point continuous score plots, without retained/removed subsets."""
    from common.pointcloud_io import write_pcd
    training_curves(run)
    rows=read_json(run/'evaluations/nominal_predictions/index.json')['samples']
    if xyz_reference is not None:
        from common.pointcloud_io import read_pcd_xyz
        cache=workspace_path(read_json(run/'config.json')['cache'])
        metadata=read_json(cache/'encoding.json');records=metadata['records']
        feature_cache=Path(metadata['feature_cache']) if 'feature_cache' in metadata else cache
    for row in rows:
        folder=Path(row['source_folder']);v=row['variant'];arrays=folder/f'arrays_{v:03d}'
        pred=np.load(row['prediction'])
        if xyz_reference is None:
            coord=np.load(arrays/'coord.npy');target=np.load(arrays/'keep_target.npy')
        else:
            coord=read_pcd_xyz(Path(xyz_reference)/row['scene']/f'prediction_{v:03d}.pcd')
            target=np.empty(len(pred),np.float32);group=np.empty(len(pred),np.uint8)
            for record in records:
                if (record['scene'],record['suite'],record['variant'])!=(row['scene'],row['suite'],v): continue
                source=feature_cache/record['cache_key'];ids=np.load(source/'point_index.npy')
                target[ids]=np.load(source/'target.npy');group[ids]=np.load(source/'group.npy')
        output=run/'predictions'/row['scene'];output.mkdir(parents=True,exist_ok=True)
        write_pcd(output/f'prediction_{v:03d}.pcd',coord,pred,field_name='keep_score')
        if v!=2: continue
        if xyz_reference is None:
            with np.load(folder/f'rough_attributes_{v:03d}.npz') as attrs:
                ghost=np.isin(attrs['source_role'],[2,6,7])
            scene=read_json(folder/'scene.json');frame=np.asarray(scene['transform'])
            yaw=np.degrees(np.arctan2(frame[1,0],frame[0,0]))
        else:
            ghost=group>=3;yaw=0.
        size=np.where(ghost,3.,.3)
        fig=plt.figure(figsize=(20,6),facecolor=BACKGROUND)
        values=[np.tile([.25,.5,.8],(len(coord),1)),plt.get_cmap('RdYlGn')(target)[:,:3],
                plt.get_cmap('RdYlGn')(pred)[:,:3],plt.get_cmap('magma')(np.abs(pred-target))[:,:3]]
        for j,(title,palette) in enumerate(zip(['Full input XYZ','Mesh ground truth','Trained keep_score','Absolute score error'],values)):
            ax=fig.add_subplot(1,4,j+1,projection='3d')
            ax.scatter(*coord.T,c=palette,s=size,depthshade=False,linewidths=0,rasterized=True)
            ax.set(xlim=(coord[:,0].min(),coord[:,0].max()),ylim=(coord[:,1].min(),coord[:,1].max()),
                   zlim=(coord[:,2].min(),coord[:,2].max()),xlabel='Map X (m)',ylabel='Map Y (m)',zlabel='Map Z (m)',title=title)
            ax.set_box_aspect(np.ptp(coord,axis=0));ax.view_init(elev=27,azim=yaw-55)
        fig.suptitle(f'Independent {row["scene"]} | {len(coord):,} original points | MAE {np.abs(pred-target).mean():.4f}\n'
            'Same XYZ and point order. Full points, no deletion. Sparse residual markers enlarged.',fontsize=14)
        fig.subplots_adjust(left=.01,right=.96,bottom=.06,top=.84,wspace=.06)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='RdYlGn'),
            cax=fig.add_axes([.965,.45,.007,.3]),label='keep score / GT')
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='magma'),
            cax=fig.add_axes([.965,.12,.007,.25]),label='absolute error')
        fig.savefig(run/'plots/test_scores.png',dpi=150,facecolor=BACKGROUND);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('runs/filter_scoring_mesh_full_20261112'))
    p.add_argument('--threshold-plots',action='store_true',help='Historical threshold-selected evaluations with all three test suites')
    p.add_argument('--prediction',type=Path,help='Render an existing real-scan prediction directory')
    p.add_argument('--output',type=Path)
    p.add_argument('--xyz-reference',type=Path,help='Archived predictions directory supplying same-order XYZ, with labels read from the feature cache')
    a=p.parse_args();run=workspace_path(a.run)
    if a.prediction:
        real_prediction(workspace_path(a.prediction),workspace_path(a.output or run/'plots/rot_stair_scores.png'));return
    if not a.threshold_plots: scoring_results(run,workspace_path(a.xyz_reference) if a.xyz_reference else None);return
    training_curves(run)
    tau=read_json(run/'evaluations/threshold.json')['threshold']
    for suite in ['nominal','sparse','few_positions']:
        rows=read_json(run/f'evaluations/{suite}_predictions/index.json')['samples']
        selected=[r for r in rows if 'straight_railed' in r['scene'] and r['variant']==3][0]
        compare(selected['source_folder'],selected['variant'],selected['prediction'],run/f'plots/test_{suite}.png',tau)
    if (run/'predictions/rot_stair/prediction.pcd').exists():
        real_prediction(run/'predictions/rot_stair',run/'plots/rot_stair.png')
if __name__=='__main__': main()
