"""Direct serial/pipeline comparisons and paired-data inspection, without training."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.pointcloud import read_json, write_json
from common.pointcloud_io import read_pcd_xyz
from .training_data import SharedScanBoxes
from .visualize import image_panel


def inspect(serial, pipeline, output):
    output.mkdir(parents=True, exist_ok=True)
    reports = [read_json(path) for path in [serial, pipeline]]
    results = []
    fig, axes = plt.subplots(2, 4, figsize=(19, 6))
    for row, (a, b) in enumerate(zip(reports[0]['routes'], reports[1]['routes'])):
        folders = [Path(r['root']) / 'mixed' for r in [a, b]]
        first, second = [dict(np.load(folder / 'attributes.npz')) for folder in folders]
        np.testing.assert_array_equal(first['frame_index'], second['frame_index'])
        np.testing.assert_array_equal(first['ray_id'], second['ray_id'])
        np.testing.assert_array_equal(first['offset_ns'], second['offset_ns'])
        np.testing.assert_array_equal(first['truth_coord'], second['truth_coord'])
        records = read_json(folders[0] / 'scans.json')
        for record in records:
            raw_a, raw_b = [np.load(folder / record['path']) for folder in folders]
            for name in ['coord', 'ray_id', 'offset_ns', 'instance_id', 'explicit_ghost', 'ghost_pull_m']:
                np.testing.assert_array_equal(raw_a[name], raw_b[name])
            raw_a.close()
            raw_b.close()
        dataset = SharedScanBoxes(folders[1])
        core_indices = []
        clean_core_count = 0
        for index in range(len(dataset)):
            item = dataset[index]
            ids = item['point_index']
            np.testing.assert_array_equal(item['xyz'], second['coord'][ids])
            np.testing.assert_array_equal(item['keep_target'], second['keep_target'][ids])
            core_indices.append(ids[item['core_mask']])
            clean_core_count += int(item['clean_core_mask'].sum())
        np.testing.assert_array_equal(np.sort(np.concatenate(core_indices)), np.arange(len(second['coord'])))
        attrs = second
        dynamic = attrs['source_role'] == 7
        ghost = attrs['explicit_ghost']
        ordinary = ~(dynamic | ghost | attrs['edge_mixed'])
        np.testing.assert_array_equal(attrs['keep_target'][dynamic], 0)
        expected = np.exp(-.5*(attrs['surface_distance_m'][~dynamic]/.03)**2)
        np.testing.assert_allclose(attrs['keep_target'][~dynamic], expected, rtol=1e-6, atol=1e-7)
        source_metrics = {}
        for name, mask in [('ordinary_static', ordinary), ('explicit_flying', ghost),
                           ('dynamic_trails', dynamic), ('mixed_edge_returns', attrs['edge_mixed'])]:
            target = attrs['keep_target'][mask]
            source_metrics[name] = dict(points=int(mask.sum()), mean_keep=float(target.mean()),
                low_02=float(np.mean(target < .2)), high_08=float(np.mean(target >= .8)))
        shift = np.linalg.norm(first['coord'] - second['coord'], axis=1)
        results.append(dict(id=b['id'], samples=len(dataset), points=len(attrs['coord']),
            raw_scans_identical=True, output_provenance_identical=True, full_core_ownership=True,
            target_formula_matches=True, paired_loader_matches=True,
            clean_core_target_points=clean_core_count, sources=source_metrics,
            serial_pipeline_point_difference_m=dict(mean=float(shift.mean()), p95=float(np.quantile(shift,.95)), max=float(shift.max()))))
        clean = dataset.clean
        bounds = np.array([np.minimum(clean.min(0),attrs['coord'].min(0))-.1,
            np.maximum(clean.max(0),attrs['coord'].max(0))+.1])
        image_panel(axes[row,0], attrs['coord'], (1,2), bounds, f"{b['id']} | input XYZ ({len(attrs['coord']):,})")
        axes[row,1].imshow(np.ones((2,2,4)),extent=[bounds[0,1],bounds[1,1],bounds[0,2],bounds[1,2]])
        axes[row,1].scatter(attrs['coord'][ordinary,1],attrs['coord'][ordinary,2],s=.03,c='#aaaeb4',rasterized=True)
        for mask,color,label in [(ghost,'#d43cb9','Explicit flying'),(dynamic,'#df8b22','Dynamic trails'),(attrs['edge_mixed'],'#2d9aad','Mixed edge')]:
            axes[row,1].scatter(attrs['coord'][mask,1],attrs['coord'][mask,2],s=.6,c=color,label=label,rasterized=True)
        axes[row,1].set(xlim=bounds[:,1],ylim=bounds[:,2],aspect='equal',title='Teacher-only sources',xlabel='Map Y (m)',ylabel='Map Z (m)')
        axes[row,1].legend(fontsize=7,markerscale=4,loc='upper right')
        # image_panel's 0..0.05 scale is mapped here to the score range 0..1.
        image_panel(axes[row,2],attrs['coord'],(1,2),bounds,'Filter GT keep_target',attrs['keep_target']*.05)
        image_panel(axes[row,3],clean,(1,2),bounds,f'Complete static reference ({len(clean):,})',color='#40987f')
    fig.suptitle('Two isolated Faster-LIO streams | one shared XYZ dataset, two supervision targets\n'
        'All observations retained. Source/score colours are teacher metadata. No model training.',fontsize=14)
    fig.subplots_adjust(top=.88,bottom=.10,left=.045,right=.95,wspace=.27,hspace=.3)
    cax=fig.add_axes([.965,.24,.008,.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='turbo'),cax=cax,label='keep_target')
    fig.savefig(output/'shared_targets.png',dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,5.5),sharex=True)
    colors={'warp':'#398ac4','lio':'#e79534','prepare':'#459f79'}
    for ax,report in zip(axes,reports):
        routes=[r['id'] for r in report['routes']]
        for event in report['events']:
            y=routes.index(event['route'])
            ax.barh(y,event['wall_s'],left=event['start_s'],height=.55,color=colors[event['stage']])
            if event['wall_s']>3:
                ax.text((event['start_s']+event['end_s'])/2,y,event['stage'],va='center',ha='center',fontsize=8)
        ax.set(yticks=range(len(routes)),yticklabels=routes,title=f"{report['schedule']} | total {report['wall_s']:.2f} s | Warp/LIO overlap {report['warp_lio_overlap_s']:.2f} s")
        ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Wall time from batch start (s)')
    fig.tight_layout()
    fig.savefig(output/'pipeline_timeline.png',dpi=150)
    plt.close(fig)
    report=dict(serial_wall_s=reports[0]['wall_s'],pipeline_wall_s=reports[1]['wall_s'],
        speedup=reports[0]['wall_s']/reports[1]['wall_s'],
        pipeline_overlap_s=reports[1]['warp_lio_overlap_s'],routes=results,
        limitation='one building, two routes, local RTX 4060. Not a 4090 benchmark or model quality evaluation.')
    write_json(output/'comparison.json',report)
    print(report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',type=Path,required=True)
    parser.add_argument('--pipeline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    inspect(args.serial,args.pipeline,args.output)


if __name__=='__main__':main()
