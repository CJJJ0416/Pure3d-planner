#!/usr/bin/env python3
"""Plot native VST, original-point lookup and the current local supervision grids."""
import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset, add_teacher_masks
from ptv3_tomography.data.tomography_pairs import load_tomography, query_world_point_costs
from ptv3_tomography.io import atomic_json
from ptv3_tomography.paths import training_run


def audit(dataset_path, output, frame=40):
    steps = LidarStepDataset(dataset_path)
    cfg = steps.spec['config']
    payload, data, scene, _ = load_tomography(steps.asset(steps.spec['scene_directory']))
    collected = {key: [] for key in ('points_world_m', 'native_cost_raw', 'local_cost_raw', 'mask_cost', 'frame_index')}
    counts = dict(all_cells=0, dangerous_cells=0, cost60_cells=0)
    for i, row in enumerate(steps.frames):
        with np.load(steps.asset(row['scan_file']), allow_pickle=False) as a:
            xyz = a['points_sensor_m'][a['hit_mask']]
        with np.load(steps.directory / row['label_file'], allow_pickle=False) as a:
            label = add_teacher_masks({k: a[k].copy() for k in a.files})
        available = label['target_available']
        tf = label['world_from_sensor']
        world = xyz @ tf[:3, :3].T + tf[:3, 3]
        native = query_world_point_costs(data, payload, world, cfg)
        xy = np.rint(xyz[:, :2]/cfg['resolution_m']).astype(int) + np.asarray(cfg['grid_size_xy'])//2
        band = np.searchsorted(cfg['height_bin_edges_m'], xyz @ label['gravity_up_sensor'], side='right')-1
        inside = ((xy >= 0) & (xy < cfg['grid_size_xy'])).all(1) & (band >= 0) & (band < len(cfg['height_bin_edges_m'])-1)
        b = np.clip(band, 0, len(cfg['height_bin_edges_m'])-2)
        x, y = np.clip(xy, 0, np.asarray(cfg['grid_size_xy'])-1).T
        mask = inside & available[b, x, y]
        values = dict(points_world_m=world, native_cost_raw=native['cost_raw'],
                      local_cost_raw=np.where(mask, label['cost_raw'][b, x, y], np.nan),
                      mask_cost=mask, frame_index=np.full(len(xyz), i, np.int32))
        for name, value in values.items():
            collected[name].append(value)
        counts['all_cells'] += int(available.sum())
        counts['dangerous_cells'] += int((available & (label['cost_raw'] >= 50)).sum())
        counts['cost60_cells'] += int((available & (label['cost_raw'] == 60)).sum())
        if i == frame:
            selected = label
    a = {key: np.concatenate(value) for key, value in collected.items()}
    output = training_run(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'stitched_points.npz', **a)
    report = dict(seed=scene['seed'], frames=len(steps), original_returns=len(a['points_world_m']),
        source_dataset=str(steps.directory), supervision_policy='native_vst_available_only',
        mask_definition='mask_cost = mask_height = target_available', frame_cells=counts,
        returns_in_supervised_cells=int(a['mask_cost'].sum()), selected_frame=frame,
        selected_frame_cells=int(selected['target_available'].sum()), renderer='CPU matplotlib', training_run=False)
    atomic_json(output / 'report.json', report)
    plot(a, selected, scene, frame, output)
    print(report, flush=True)
    return report


def plot(a, label, scene, frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('cost', ['#16a34a', '#fde047', '#ef4444'])
    available = label['target_available']
    panels = [(label['surface_points_world_m'][available], label['cost_raw'][available],
               f'Frame {frame}: all available supervision grid cells'),
              (a['points_world_m'], a['native_cost_raw'], 'Original returns: direct native VST lookup'),
              (a['points_world_m'], a['local_cost_raw'], 'Original returns: local-grid supervision')]
    with plt.rc_context({'font.size': 10, 'figure.facecolor': '#f8fafc'}):
        fig = plt.figure(figsize=(18, 10), layout='constrained')
        for row in range(2):
            for col, (xyz, cost, title) in enumerate(panels):
                ax = fig.add_subplot(2, 3, row*3+col+1, projection='3d' if row == 0 else None)
                known = np.isfinite(cost)
                dim = 3 if row == 0 else 2
                options = {'depthshade': False} if row == 0 else {}
                ax.scatter(*xyz[~known, :dim].T, c='#94a3b8', s=.6, alpha=.2, **options)
                h = ax.scatter(*xyz[known, :dim].T, c=cost[known], cmap=cmap, vmin=0, vmax=60, s=2.4, **options)
                if row == 0:
                    ax.view_init(elev=30, azim=-64)
                    ax.set_box_aspect([1, 1, .4]); ax.set_zlabel('World Z (m)')
                else:
                    ax.set_aspect('equal')
                ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
                ax.set_title(f'{title}\n{int(known.sum()):,} colored')
        fig.colorbar(h, ax=fig.axes, location='bottom', shrink=.5, label='Tomography cost (0-60)')
        fig.suptitle(f'Seed {scene["seed"]} | native VST supervision | same cost scale', fontsize=18)
        fig.savefig(output / 'world_stitched_cost_comparison.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=40)
    args = parser.parse_args()
    audit(args.dataset, args.output, args.frame)


if __name__ == '__main__':
    main()
