#!/usr/bin/env python3
"""Render old/new Tomography labels using an unchanged saved RGB and camera."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-labels', type=Path, required=True)
    p.add_argument('--direct-labels', type=Path, required=True)
    p.add_argument('--rgb-reference', type=Path, required=True, help='Existing render_plan_a_rgb_gt JSON camera record')
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image
    from ptv3_tomography.data.lidar_reference_pairs import load_lidar_config
    from ptv3_tomography.io import atomic_json, file_sha256

    with np.load(args.reference_labels, allow_pickle=False) as archive:
        old = {k: archive[k].copy() for k in archive.files}
    with np.load(args.direct_labels, allow_pickle=False) as archive:
        direct = {k: archive[k].copy() for k in archive.files}
    reference = json.loads(args.rgb_reference.read_text())
    rgb_file = args.rgb_reference.with_name(args.rgb_reference.stem + '_rgb.png')
    rgb = np.asarray(Image.open(rgb_file).convert('RGB'))
    cfg = load_lidar_config()
    camera = reference['camera']
    origin, forward, right, up = [np.asarray(camera[k]) for k in ('origin', 'forward', 'right', 'up')]
    k = camera['intrinsics']
    band = reference['band']
    metrics = {}
    for field in ('cost_raw', 'ground_height_gravity_m', 'mask_cost', 'mask_height', 'target_available', 'ambiguous'):
        np.testing.assert_array_equal(old[field], direct[field], err_msg=field)
    for name, field in (('cost', 'cost_raw'), ('height', 'ground_height_gravity_m')):
        mask = old['mask_' + name]
        metrics[name] = dict(supervised_cells=int(mask.sum()), mask_differences=int((mask ^ direct['mask_' + name]).sum()),
                             max_absolute_difference=float(np.max(np.abs(old[field][mask] - direct[field][mask]))) if mask.any() else None)

    def anchors(data, name):
        mask = data['mask_' + name].copy()
        mask[np.arange(mask.shape[0]) != band] = False
        points = data['surface_points_world_m'][mask]
        delta = points - origin
        depth = delta @ forward
        uv = np.column_stack((k['fx'] * (delta @ right) / np.maximum(depth, 1e-8) + k['cx'],
                              k['cy'] - k['fy'] * (delta @ up) / np.maximum(depth, 1e-8)))
        selected = np.asarray(reference['visible_gt_cell_indices'][name], int)
        field = 'cost_raw' if name == 'cost' else 'ground_height_gravity_m'
        return uv[selected], data[field][mask][selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cost_cmap = LinearSegmentedColormap.from_list('cost', ['#16a34a', '#fde047', '#ef4444'])
    low, high = cfg['height_bin_edges_m'][band:band + 2]
    nx, ny = old['cost_raw'].shape[1:]
    resolution = cfg['resolution_m']
    extent = [(-nx // 2 - .5) * resolution, (nx - 1 - nx // 2 + .5) * resolution,
              (-ny // 2 - .5) * resolution, (ny - 1 - ny // 2 + .5) * resolution]
    bench = json.loads(args.benchmark.read_text()) if args.benchmark else None
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 15, 'figure.facecolor': '#f6f8fb'}):
        fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), constrained_layout=True)
        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f'Saved RGB | platform edge | frame {reference["frame_index"]}')
        for row, (data, tag) in enumerate(((old, 'Original full-map projection'), (direct, 'Direct coordinate lookup'))):
            for col, name, title, cmap, lo, hi in ((1, 'cost', 'cost', cost_cmap, 0, 60),
                                                   (2, 'height', 'ground height', 'viridis', low, high)):
                ax = axes[row, col]
                ax.imshow(rgb)
                uv, values = anchors(data, name)
                scatter = ax.scatter(uv[:, 0], uv[:, 1], c=values, s=20, cmap=cmap, vmin=lo, vmax=hi,
                                     edgecolors='#243043', linewidths=.25)
                ax.set_xlim(0, rgb.shape[1] - 1)
                ax.set_ylim(rgb.shape[0] - 1, 0)
                ax.set_title(f'{tag}\nGT {title} | {len(values)} visible cells')
                fig.colorbar(scatter, ax=ax, shrink=.85, pad=.01, label='cost 0..60' if name == 'cost' else 'gravity-relative height (m)')
        axes[1, 0].axis('off')
        text = 'LABEL PARITY\n\nCost values: identical\nGround height: identical\nCost / height masks: identical\nUnknown / ambiguous: unchanged\nCost 60 retained\n'
        if bench:
            text += (f'\n40-frame preparation, same raw scans\n'
                     f'Original: {bench["reference"]["total_s"]:.2f} s\n'
                     f'Direct: {bench["direct"]["total_s"]:.2f} s\n'
                     f'Speedup: {bench["speedup"]:.2f}x\n')
        text += '\nRGB is the unchanged saved mesh render.\nColored markers are supervised map cells.'
        axes[1, 0].text(.03, .98, text, va='top', fontsize=13, linespacing=1.5, color='#213047', transform=axes[1, 0].transAxes)
        fig.suptitle('Tomography supervision | same camera, same scan, same color scales\nOriginal labels and direct-lookup labels', fontsize=20, fontweight='bold')
        fig.savefig(args.output, dpi=170, facecolor=fig.get_facecolor())
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
        for row, (field, title, cmap, lo, hi, mask_name) in enumerate((
            ('cost_raw', 'Cost', cost_cmap, 0, 60, 'mask_cost'),
            ('ground_height_gravity_m', 'Ground height', 'viridis', low, high, 'mask_height'))):
            for col, (data, tag) in enumerate(((old, 'Original'), (direct, 'Direct lookup'))):
                values = np.ma.masked_where(~data['target_available'][band], data[field][band])
                cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
                cm.set_bad('#d7dce4')
                im = axes[row, col].imshow(values.T, origin='lower', cmap=cm, vmin=lo, vmax=hi,
                                           extent=extent)
                mask = data[mask_name][band]
                if mask.any() and not mask.all():
                    axes[row, col].contour(mask.T, levels=[.5], colors='black', linewidths=.6,
                                          origin='lower', extent=extent)
                axes[row, col].set_title(f'{tag} | {title}')
                fig.colorbar(im, ax=axes[row, col], shrink=.85)
            valid = old['target_available'][band] & direct['target_available'][band]
            difference = np.ma.masked_where(~valid, np.abs(old[field][band] - direct[field][band]))
            cm = plt.get_cmap('magma').copy()
            cm.set_bad('#d7dce4')
            axes[row, 2].imshow(difference.T, origin='lower', cmap=cm, vmin=0, vmax=1,
                                extent=extent)
            axes[row, 2].set_title(f'Absolute difference | max = {metrics["cost" if row == 0 else "height"]["max_absolute_difference"]:.0f}')
            for ax in axes[row]:
                ax.set_xlabel('LiDAR X (m)')
                ax.set_ylabel('LiDAR Y (m)')
        fig.suptitle(f'Frame {reference["frame_index"]} | gravity-height band [{low}, {high}) m\nGray: unavailable teacher. Black outline: supervised cells. Difference panels use fixed 0..1 range.', fontsize=15)
        grid_output = args.output.with_name(args.output.stem + '_grid.png')
        fig.savefig(grid_output, dpi=170)
        plt.close(fig)
    atomic_json(args.output.with_suffix('.json'), dict(status='passed', frame=reference['frame_index'], band=band,
        source_rgb_sha256=file_sha256(rgb_file), original_labels_sha256=file_sha256(args.reference_labels),
        direct_labels_sha256=file_sha256(args.direct_labels), metrics=metrics,
        same_saved_rgb_and_camera=True, full_teacher_grid_image=str(grid_output), model_prediction_shown=False))
    print(args.output)


if __name__ == '__main__':
    main()
