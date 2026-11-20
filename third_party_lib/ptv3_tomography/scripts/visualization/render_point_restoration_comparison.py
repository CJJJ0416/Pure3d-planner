#!/usr/bin/env python3
"""Arrange verified native Isaac screenshots into a point restoration comparison figure."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


from ptv3_tomography.paths import relocated_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restoration',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--frame',type=int,default=16)
    parser.add_argument('--view',choices=['detail','overview'],default='detail')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from ptv3_tomography.io import read_json,atomic_json,file_sha256
    report = read_json(args.restoration/'report.json')
    native = read_json(args.native/'report.json')
    row = next(r for r in report['frames_manifest'] if r['frame_index']==args.frame)
    n,m,c = row['valid_returns'],row['voxel_tokens'],row['restored_feature_shape'][1]
    titles = [f'1. Raw LiDAR | {n} valid points',f'2. PTv3 voxel features | {m} x {c}',
              f'3. Inverse-restored features | {n} x {c}',
              f"4. Predicted map cost at LiDAR points\n{row['within_map_domain']} colored + {row['outside_map_domain']} gray"]
    modes = ['raw_lidar','voxel_pca','restored_pca','sampled_map_cost']
    inputs = []
    with plt.rc_context({'font.size':12,'figure.facecolor':'#f8fafc'}):
        fig,axes = plt.subplots(2,2,figsize=(16,11.8))
        for ax,mode,title in zip(axes.flat,modes,titles):
            capture = next(r for r in native['captures'] if r['mode']==mode and r['view']==args.view)
            file = relocated_path(capture['file'])
            ax.imshow(Image.open(file))
            ax.set_title(title,fontsize=14,pad=10,fontweight='bold')
            ax.set_axis_off()
            inputs.append(capture)
        fig.suptitle(f'Frame {args.frame} | PTv3 point restoration in the native Isaac scene\n'
                     f'{m} voxel features -> {n} original LiDAR points | no GT-mask filtering',
                     fontsize=20,fontweight='bold',y=.985)
        fig.text(.5,.035,'Panels 2/3: shared PCA feature colors, not cost or semantic labels. Same-voxel points share a feature.\n'
                 'Panel 4: existing map cost sampled at original point positions. Green 0 -> yellow 30 -> red 60.\n'
                 'Gray: outside map coordinates / height bands. Original XYZ is retained in every LiDAR point view.',
                 ha='center',fontsize=12,color='#334155',linespacing=1.5)
        fig.subplots_adjust(left=.015,right=.985,top=.90,bottom=.13,hspace=.14,wspace=.035)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.output,dpi=150)
        plt.close(fig)
    atomic_json(args.output.with_suffix('.json'),{'status':'passed','frame':args.frame,'view':args.view,
                'input_native_captures':inputs,'figure_sha256':file_sha256(args.output),
                'valid_returns':n,'voxel_tokens':m,'restored_feature_shape':[n,c],
                'all_original_point_coordinates_retained':True,'gt_masks_applied':False,
                'pca_colors_are_not_cost':True,'checkpoint_sha256':report['checkpoint_sha256']})
    print('POINT_RESTORATION_COMPARISON',args.output,flush=True)


if __name__ == '__main__':
    main()
