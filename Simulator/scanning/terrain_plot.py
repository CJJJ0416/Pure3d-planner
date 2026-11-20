"""Inspect batched terrain scans, their independent LIO outputs and targets."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from common.pointcloud_io import read_pcd_fields
from utils.pointcloud import read_json,write_json
from .training_data import SharedScanBoxes
from .visualize import image_panel


def plot(report_path):
    report=read_json(report_path)
    root=Path(report['settings']['output'])
    out=report_path.parent/'plots'
    out.mkdir(exist_ok=True)
    first=SharedScanBoxes(Path(report['routes'][0]['root'])/'mixed')
    clean=first.clean
    bounds=np.array([clean.min(0)-.1,clean.max(0)+.1])
    for row in report['routes']:
        xyz=np.load(Path(row['root'])/'mixed/arrays_000/coord.npy',mmap_mode='r')
        bounds[0]=np.minimum(bounds[0],xyz.min(0)-.1)
        bounds[1]=np.maximum(bounds[1],xyz.max(0)+.1)
    del first
    colors=['#d64c4c','#246fb9','#8c52bb','#dd9e25']
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for axis,projection in zip(axes,[(0,1),(1,2)]):
        image_panel(axis,clean,projection,bounds,'',color='#e1e5eb')
    for index,row in enumerate(report['routes']):
        color=colors[index%len(colors)]
        folder=Path(row['root'])
        route=next(r for r in report['planning']['routes'] if r['id']==row['id'])
        support=np.load(route['support_path'])
        planned=support+np.array([0,0,.8])
        truth=np.load(folder/'truth_trajectory.npz')
        seconds=read_json(folder/'mixed/generation.json')['captured_prefix_s']
        p=truth['position'][truth['time']<=seconds+1e-6]
        for axis,projection in zip(axes,[(0,1),(1,2)]):
            a,b=projection
            axis.plot(planned[:,a],planned[:,b],linestyle='--',lw=1.1,color=color,alpha=.7)
            axis.plot(p[:,a],p[:,b],lw=2.5,color=color,label=f"{row['id']} | captured {seconds:g} s")
            axis.scatter(planned[0,a],planned[0,b],marker='o',s=40,color=color)
            axis.scatter(planned[-1,a],planned[-1,b],marker='*',s=90,color=color)
    axes[0].legend(loc='upper right',fontsize=8)
    axes[0].set_title('Full terrain and random VST/GPMP paths | top view')
    axes[1].set_title('Selected floor heights | side view')
    fig.suptitle('Dashed: complete planned sensor-height route. Solid: actually scanned segment.\n'
        'Circle: start. Star: goal. Local test uses prefixes of the two long routes.',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.9])
    fig.savefig(out/'routes_overview.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(len(report['routes']),3,figsize=(15,3*len(report['routes'])),squeeze=False)
    checks=[]
    for index,row in enumerate(report['routes']):
        folder=Path(row['root'])/'mixed'
        data=dict(np.load(folder/'attributes.npz'))
        fields=read_pcd_fields(folder/'rough.pcd')
        assert fields.dtype.names==('x','y','z')
        np.testing.assert_array_equal(np.column_stack([fields[k] for k in fields.dtype.names]),data['coord'])
        expected_id=int(row['id'].rsplit('_',1)[1])
        np.testing.assert_array_equal(data['trajectory_id'],expected_id)
        dynamic=data['source_role']==7
        np.testing.assert_array_equal(data['keep_target'][dynamic],0.)
        np.testing.assert_allclose(data['keep_target'][~dynamic],np.exp(-.5*(data['surface_distance_m'][~dynamic]/.03)**2),rtol=1e-6,atol=1e-7)
        paired=SharedScanBoxes(folder)
        owned=[]
        for box in range(len(paired)):
            sample=paired[box]
            ids=sample['point_index']
            np.testing.assert_array_equal(sample['xyz'],data['coord'][ids])
            np.testing.assert_array_equal(sample['keep_target'],data['keep_target'][ids])
            owned.append(ids[sample['core_mask']])
        np.testing.assert_array_equal(np.sort(np.concatenate(owned)),np.arange(len(data['coord'])))
        checks.append(dict(route=row['id'],points=len(data['coord']),boxes=len(paired),
            xyz_pcd_only=True,trajectory_identity=True,mesh_targets=True,unique_core_ownership=True,
            paired_loader=True,ghost_points=int(data['explicit_ghost'].sum()),
            dynamic_points=int(dynamic.sum())))
        image_panel(axes[index,0],data['coord'],(1,2),bounds,f"{row['id']} | LIO XYZ ({len(data['coord']):,})")
        image_panel(axes[index,1],data['coord'],(1,2),bounds,'Filter GT: original point order',data['keep_target']*.05)
        image_panel(axes[index,2],clean,(1,2),bounds,'Complete static reference',color='#4b9d7d')
        del data,paired,fields
    fig.suptitle('Batched Warp -> independent Faster-LIO -> shared XYZ observations\n'
        'GT scores and full reference are supervision, not model inputs.',fontsize=13)
    fig.subplots_adjust(left=.05,right=.95,top=.88,bottom=.08,wspace=.23,hspace=.36)
    cax=fig.add_axes([.965,.23,.01,.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='turbo'),cax=cax,label='keep_target')
    fig.savefig(out/'scan_and_targets.png',dpi=140)
    plt.close(fig)
    names=list(dict.fromkeys(e['route'] for e in sorted(report['events'],key=lambda r:r['start_s'])))
    fig,ax=plt.subplots(figsize=(13,5))
    for event in report['events']:
        y=names.index(event['route'])
        color={'plan':'#b3b8c3','warp':'#398ac4','lio':'#e79534','prepare':'#459f79'}[event['stage']]
        ax.barh(y,event['end_s']-event['start_s'],left=event['start_s'],height=.6,color=color)
        if event['end_s']-event['start_s']>3:
            ax.text((event['start_s']+event['end_s'])/2,y,event['stage'],ha='center',va='center',fontsize=8)
    ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Wall time (s)',
        title=f"Batched rays and isolated LIO pipeline | {report['wall_s']:.2f} s total")
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'timeline.png',dpi=150)
    plt.close(fig)
    write_json(report_path.parent/'direct_checks.json',dict(routes=checks,
        batched_vs_individual_ray_queries=all(check['outputs_equal'] for group in report['ray_batches'] for check in group['single_vs_batched_checks']),
        model_training_started=False))
    print('TERRAIN_PLOTS',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    plot(args.report.resolve())


if __name__=='__main__':main()
