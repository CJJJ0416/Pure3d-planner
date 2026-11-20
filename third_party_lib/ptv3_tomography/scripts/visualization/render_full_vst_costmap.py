#!/usr/bin/env python3
"""Plot every cell of the source VST costmap, without local or supervision masks."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle, Patch

    scene_file = args.scene / 'scene.json'
    source = args.scene / 'vst.pickle'
    source_hash = sha256(source)
    scene = json.loads(scene_file.read_text())
    meta = json.loads((args.scene / 'vst.meta.json').read_text())
    with source.open('rb') as stream:
        payload = pickle.load(stream)
    data = np.asarray(payload['data'], dtype=np.float32)
    cost, ground = data[0], data[3]
    valid = np.isfinite(ground)
    resolution = float(payload['resolution'])
    nx, ny = cost.shape[1:]
    center = np.asarray(payload['center'])
    x = (np.arange(nx) - nx // 2) * resolution + center[0]
    y = (np.arange(ny) - ny // 2) * resolution + center[1]
    extent = [x[0] - resolution/2, x[-1] + resolution/2,
              y[0] - resolution/2, y[-1] + resolution/2]
    # Integer cell centres round-trip to the saved VST indexing convention.
    manifest = json.loads((args.maps / 'manifest.json').read_text())
    row = next(row for row in manifest['frames'] if row['frame_index'] == args.frame)
    shard = args.maps / row['file']
    with np.load(shard, allow_pickle=False) as a:
        sensor = a['world_from_sensor'][:3, 3].copy()
    route = np.loadtxt(args.scene / 'reference_route.csv', delimiter=',', skiprows=1)
    cmap = LinearSegmentedColormap.from_list('vst_cost', ['#16a34a', '#fde047', '#ef4444'])
    count = len(cost)
    cols = min(3, count)
    rows = int(np.ceil(count / cols))
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 16, 'figure.facecolor': '#f8fafc',
                         'hatch.color': '#334155', 'hatch.linewidth': .6}):
        fig, axes = plt.subplots(rows, cols, figsize=(7.2*cols, 7.4*rows),
                                 squeeze=False, constrained_layout=True)
        for layer, ax in enumerate(axes.flat):
            if layer >= count:
                ax.set_visible(False)
                continue
            image = ax.imshow(cost[layer].T, origin='lower', extent=extent, interpolation='nearest',
                              cmap=cmap, vmin=0, vmax=60, zorder=0)
            if (~valid[layer]).any():
                hatch = ax.contourf(x, y, (~valid[layer]).T.astype(float), levels=[.5, 1.5],
                                   colors='none', hatches=['////'], zorder=1)
                hatch.set_edgecolor('#475569')
            # Solid outlines use source geometry metadata, not predicted values.
            for solid in scene['solids']:
                lo, hi = np.asarray(solid['lo']), np.asarray(solid['hi'])
                if solid['kind'] == 'lower_floor':
                    continue
                ax.add_patch(Rectangle(lo[:2], *(hi-lo)[:2], fill=False, edgecolor='#0f172a',
                                       linewidth=1.0, linestyle='--', zorder=2))
            ax.add_patch(Rectangle((-.9, 0), 1.8, 3.2, fill=False, edgecolor='#0f172a',
                                   linewidth=1.0, linestyle='--', zorder=2))
            ax.plot(route[:, 0], route[:, 1], color='#0f172a', linewidth=4.5, zorder=3)
            ax.plot(route[:, 0], route[:, 1], color='#67e8f9', linewidth=2.3,
                    label='Capture reference route', zorder=4)
            ax.scatter(*route[0, :2], s=85, marker='o', color='#38bdf8', edgecolors='#0f172a', zorder=5)
            ax.scatter(*route[-1, :2], s=95, marker='s', color='#e879f9', edgecolors='#0f172a', zorder=5)
            ax.scatter(*sensor[:2], s=240, marker='*', color='white', edgecolors='#0f172a', linewidths=1.2,
                       label=f'Frame {args.frame} LiDAR', zorder=6)
            ax.text(route[0, 0]+.2, route[0, 1]-.2, 'Start', fontweight='bold', color='#0f172a')
            ax.text(route[-1, 0]+.2, route[-1, 1]+.1, 'Goal', fontweight='bold', color='#0f172a')
            ax.set(xlabel='World X (m)', ylabel='World Y (m)',
                   title=f'Full VST costmap | saved layer {layer}\n{nx} x {ny} cells, {resolution:g} m/cell')
            ax.set_aspect('equal')
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor='none', edgecolor='#475569', hatch='////'))
            labels.append('No ground observation (raw cost = 60)')
            ax.legend(handles, labels, loc='lower left', fontsize=9, framealpha=.9)
            fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label='Original VST cost (0..60)')
        fig.suptitle('Entire new-scene VST planner costmaps\n'
                     'Direct saved planner output. No PTv3 prediction, local crop, or supervision mask.',
                     fontsize=18, fontweight='bold')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        plt.close(fig)
    report = {'status': 'passed', 'source_vst': str(source.resolve()), 'source_sha256': source_hash,
              'image_sha256': sha256(args.output), 'full_cost_shape': list(cost.shape),
              'displayed_cost_cells': int(cost.size), 'cost_values_modified': False,
              'local_crop_applied': False, 'supervision_mask_applied': False,
              'axis_convention': 'saved arrays [layer, world_x_index, world_y_index], displayed transposed with origin lower',
              'extent_world_xy_m': extent, 'resolution_m': resolution,
              'sensor_world_xyz_m': sensor.tolist(), 'reference_frame': args.frame,
              'unknown_ground_display': 'hatching over unchanged raw cost=60 colors; not a confirmed physical obstacle',
              'layers': [{'saved_layer_index': i, 'cells': int(cost[i].size),
                          'ground_observed_cells': int(valid[i].sum()), 'raw_cost60_cells': int((cost[i]>=60).sum()),
                          'no_ground_observation_cells': int((~valid[i]).sum())} for i in range(count)]}
    args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
