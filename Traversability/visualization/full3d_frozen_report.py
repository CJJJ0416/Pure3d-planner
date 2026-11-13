"""Report frozen pretrained-encoder training on complete 3-D geometry."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from Traversability.data.rotating import disk_bytes
from utils.pointcloud import read_json, workspace_path, write_json


def report(output):
    output = Path(output).resolve()
    config = read_json(output / 'config.json')
    summary = read_json(output / 'summary.json')
    history = read_json(output / 'history.json')
    data = workspace_path(config['data'])
    meta = read_json(data / 'manifest.json')
    initial, final = summary['initial_validation'], summary['best_validation']
    figures = output / 'plots'
    figures.mkdir(exist_ok=True)
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name in ['total', 'absolute', 'conditional']:
        axes[0, 0].plot(epochs, [r['loss'][name] for r in history], label=name)
    axes[0, 0].set(title='Training loss', ylabel='SmoothL1')
    axes[0, 1].plot([0, *epochs], [initial['cost_mae_raw'],
        *[r['validation']['cost_mae_raw'] for r in history]], label='Validation')
    axes[0, 1].set(title='Fixed validation', ylabel='Raw cost MAE (0-50)')
    for name in config['condition_names']:
        for ax, field in zip(axes[1], ['changed_delta_mae', 'stable_delta_mae']):
            ax.plot([0, *epochs], [initial['parameter_response'][name][field],
                *[r['validation']['parameter_response'][name][field] for r in history]], label=name)
    for ax, title in zip(axes[1], ['Changed-condition response', 'Stable-condition response']):
        ax.set(title=title, ylabel='Normalized delta MAE')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle(f"Frozen {config['encoder']['name']} | selected epoch {summary['selected_epoch']}")
    fig.savefig(figures / 'training_curves.png', dpi=160)
    plt.close(fig)
    comparisons = []
    for scene in [s for s in meta['scenes'] if s['split'] == 'test']:
        name = scene['id']
        xyz = np.load(data / name / 'coord.npy')
        truth = np.load(data / name / 'cost.npy', mmap_mode='r')[:, 3].astype(np.float32)
        pred = np.load(output / 'predictions' / f'{name}_prediction.npy')
        before = np.load(output / 'initial_predictions' / f'{name}_prediction.npy')
        independent = np.load(output / 'independent_prediction/prediction.npy')
        independent_xyz = np.load(output / 'independent_prediction/geometry/coord.npy')
        check = dict(scene=name, xyz_equal=bool(np.array_equal(xyz, independent_xyz)),
            maximum_prediction_difference=float(np.max(np.abs(pred - independent))))
        assert check['xyz_equal'] and check['maximum_prediction_difference'] < 1e-5, check
        comparisons.append(check)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        panels = [('VST reference', truth), ('Random-head initialization', before),
                  ('Selected head', pred), ('Absolute error', np.abs(pred - truth))]
        for ax, (title, values) in zip(axes, panels):
            ax.scatter(xyz[:, 0], xyz[:, 1], c=values, s=.6, lw=0,
                vmin=0, vmax=.5 if title == 'Absolute error' else 1,
                cmap='magma' if title == 'Absolute error' else 'viridis', rasterized=True)
            ax.set(title=title, xlabel='Map X (m)', ylabel='Map Y (m)', aspect='equal')
        fig.suptitle(name + ' | identical support XYZ | XY projection')
        fig.savefig(figures / f'{name}_GT_before_after_error.png', dpi=150)
        plt.close(fig)
    write_json(output / 'independent_inference_verification.json', comparisons)
    rotations = [read_json(p) for p in sorted((output / 'rotations').glob('*.json'))]
    state = read_json(data / 'generation.json')
    storage = dict(active_seeds=state['seeds'], active_bytes=disk_bytes(data / 'active'),
        fixed_cache_bytes=disk_bytes(data / 'fixed'), archive_bytes=disk_bytes(output / 'scene_archive'),
        rotations=len(rotations),
        all_prior_pools_removed=all(r['previous_active_removed'] and r['previous_wfc_cache_removed'] for r in rotations))
    assert len(rotations) == summary['generations'] - 1 and storage['all_prior_pools_removed']
    write_json(output / 'storage_verification.json', storage)
    text = f"""# Frozen {config['encoder']['name']} full-3D training

Completed epochs: {summary['completed_epochs']}. Selected epoch: {summary['selected_epoch']}.
Scheduler budget: {config['epochs']}. Training scenes: {summary['new_scenes']}.
The pretrained encoder is frozen. Adapter, FiLM and CostHead start from random weights.

| Metric | Initialization | Selected |
|---|---:|---:|
| Validation raw MAE | {initial['cost_mae_raw']:.5f} | {final['cost_mae_raw']:.5f} |
| Mean condition response error | {initial['mean_response_error']:.6f} | {final['mean_response_error']:.6f} |
| Validation selection score | {initial['score']:.6f} | {final['score']:.6f} |
| Test raw MAE | {summary['initial_test']['cost_mae_raw']:.5f} | {summary['test']['cost_mae_raw']:.5f} |

Independent PCD inference matches the cached test predictions at identical support XYZ.
See `training_verification.json` for frozen weights and optimizer progress checks.
See `summary.json` for all four condition responses and fixed-split metrics.
Real-building holdout metrics are recorded separately after training.

Training curves and XY projections are in `plots/`.
"""
    (output / 'REPORT.md').write_text(text, encoding='utf-8')
    return storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report(args.run)
    print(args.run / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
