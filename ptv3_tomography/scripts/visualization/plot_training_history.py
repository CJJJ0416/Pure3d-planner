#!/usr/bin/env python3
"""Save training and terrain-evaluation curves with the CPU Agg backend."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    history = read(args.run / 'logs/epochs.json')
    protocol = read(args.run / 'protocol.json')
    output = args.output or args.run / 'plots/training_metrics.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 10, 'figure.facecolor': '#f8fafc'}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
        epochs = [row['epoch'] for row in history]
        for key, title in (('mean_loss', 'Total loss'), ('cost_loss', 'Cost loss'), ('height_loss', 'Height loss')):
            axes[0, 0].plot(epochs, [row['train'][key] for row in history], label=title)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Training loss')
        axes[0, 0].legend()
        axes[0, 1].plot(epochs, [row['train']['cost_mae_raw'] for row in history], label='Train')
        axes[1, 0].plot(epochs, [100*row['train']['height_mae_m'] for row in history], label='Train')
        for group, title, color in (('fixed', 'Fixed validation', '#f97316'), ('fresh', 'Fresh terrain', '#16a34a')):
            rows = []
            for path in sorted((args.run / 'evaluations').glob(f'stage_*_{group}.json')):
                report = read(path)
                epoch = int(path.stem.split('_')[1])*protocol['segment_epochs']
                dangerous = sum(row['dangerous_cells'] for row in report['seeds'])
                missed = sum(row['dangerous_underprediction_cells'] for row in report['seeds'])
                rate = 100*missed/dangerous if dangerous else float('nan')
                rows.append((epoch, report['cost_mae'], 100*report['height_mae_m'], rate))
            axes[0, 1].plot([r[0] for r in rows], [r[1] for r in rows], 'o-', label=title, color=color)
            axes[1, 0].plot([r[0] for r in rows], [r[2] for r in rows], 'o-', label=title, color=color)
            axes[1, 1].plot([r[0] for r in rows], [r[3] for r in rows], 'o-', label=title, color=color)
        axes[0, 1].set_title('Cost MAE (raw cost units)')
        axes[1, 0].set_title('Ground-height MAE (cm)')
        axes[1, 1].set_title('Dangerous-cost underprediction: GT >= 50, prediction < 30')
        axes[1, 1].set_ylabel('Dangerous labeled cells (%)')
        axes[1, 1].set_ylim(0, 100)
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.legend()
        fig.suptitle('PTv3 | native VST supervision | headless evaluation', fontsize=16)
        fig.savefig(output, dpi=160)
        plt.close(fig)
    print(f'Saved training curves: {output}', flush=True)


if __name__ == '__main__':
    main()
