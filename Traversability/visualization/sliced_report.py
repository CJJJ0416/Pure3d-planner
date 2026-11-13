"""Real local training curves and full-building slice cost comparisons."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils.pointcloud import read_json, write_json, workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    output = parser.parse_args().run.resolve()
    summary = read_json(output / 'summary.json')
    history = read_json(output / 'history.json')
    config = read_json(output / 'protocol.json')['config']
    data = workspace_path(config['data'])
    metadata = read_json(data / 'manifest.json')
    figures = output / 'plots'
    figures.mkdir(exist_ok=True)
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for name in ['total', 'absolute', 'conditional']:
        axes[0, 0].plot(epochs, [r['loss'][name] for r in history], label=name)
    axes[0, 0].set(title='Training losses', xlabel='Epoch', ylabel='SmoothL1')
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [r['validation']['cost_mae_raw'] for r in history], label='Fixed validation')
    axes[0, 1].axhline(summary['initial_validation']['cost_mae_raw'], color='gray', ls='--', label='Initialization')
    axes[0, 1].set(title='Validation Cost MAE', xlabel='Epoch', ylabel='MAE (raw cost units, 0-50)')
    axes[0, 1].legend()
    for name in config['condition_names']:
        for ax, field, title in [(axes[1, 0], 'changed_delta_mae', 'Parameter response: changed regions'),
                                 (axes[1, 1], 'stable_delta_mae', 'Parameter response: stable regions')]:
            ax.plot(epochs, [r['validation']['parameter_response'][name][field] for r in history], label=name)
            ax.set(title=title, xlabel='Epoch', ylabel='Normalized delta MAE')
            ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle(f'Native sliced XYZ | frozen official Tiny | selected epoch {summary["selected_epoch"]}')
    fig.savefig(figures / 'training_curves.png', dpi=160)
    plt.close(fig)
    for scene in [s for s in metadata['scenes'] if s['split'] == 'test']:
        name = scene['id']
        coord = np.load(data / name / 'coord.npy')
        truth = np.load(data / name / 'cost.npy', mmap_mode='r')[:, 3].astype(np.float32)
        predicted = np.load(output / 'predictions' / f'{name}_prediction.npy')
        independent = np.load(output / 'independent_prediction/prediction.npy')
        independent_xyz = np.load(output / 'independent_prediction/geometry/coord.npy')
        verification = dict(independent_xyz_identical=bool(np.array_equal(coord, independent_xyz)),
                            independent_prediction_max_abs_difference=float(abs(predicted-independent).max()),
                            points=len(coord), checkpoint_epoch=summary['selected_epoch'])
        write_json(output / 'independent_inference_verification.json', verification)
        panels = [('VST GT', truth), ('Tiny + FiLM prediction', predicted), ('Absolute error', abs(predicted-truth))]
        scene_info = read_json(Path(scene['source_pcd']).with_name('scene.json'))
        floors = scene_info['config']['floor_z_m']
        groups = np.searchsorted(np.asarray(floors[1:])-.05, coord[:, 2], side='right')
        cost_cmap = LinearSegmentedColormap.from_list('native_cost', ['#16a34a', '#fde047', '#ef4444'])
        fig, axes = plt.subplots(len(floors), 3, figsize=(16, 5*len(floors)), constrained_layout=True,
                                 squeeze=False, sharex=True, sharey=True)
        for row, base in enumerate(floors):
            take = groups == row
            for col, (title, values) in enumerate(panels):
                ax = axes[row, col]
                image = ax.scatter(coord[take, 0], coord[take, 1], c=values[take],
                                   cmap=cost_cmap if col < 2 else 'magma', vmin=0,
                                   vmax=1 if col < 2 else .5, s=1.1, linewidths=0, rasterized=True)
                if row == 0:
                    ax.set_title(title, fontsize=14)
                ax.set_aspect('equal')
                ax.set(xlabel='Map X (m)', ylabel=f'Floor {row} | base Z={base:.2f}m\nMap Y (m)')
                ax.grid(alpha=.15)
                if col == 0:
                    cost_image = image
                if col == 2:
                    error_image = image
        fig.colorbar(cost_image, ax=axes[:, :2], shrink=.7, label='trav_cost (0-1)')
        fig.colorbar(error_image, ax=axes[:, 2], shrink=.7, label='Absolute error (0-1)')
        fig.suptitle(f'{name} held-out building | identical {len(coord):,} slice XYZ | condition [0.5,0.6,0.4,0.05]\n'
                     f'Raw MAE={50*abs(predicted-truth).mean():.3f} | all points displayed, grouped by floor', fontsize=15)
        fig.savefig(figures / f'{name}_GT_prediction_error.png', dpi=160)
        plt.close(fig)
    a, b = summary['initial_validation'], summary['best_validation']
    text = f'''# Native sliced Tiny 本机50轮训练

本次完成50轮，按固定验证 `Cost MAE + 四参数平均响应误差` 选择第 {summary['selected_epoch']} 轮。

| 指标 | 初始化 | 所选权重 |
|---|---:|---:|
| 验证原始 Cost MAE | {a['cost_mae_raw']:.4f} | {b['cost_mae_raw']:.4f} |
| 验证条件平均响应误差 | {a['mean_response_error']:.6f} | {b['mean_response_error']:.6f} |
| 验证综合分数 | {a['score']:.6f} | {b['score']:.6f} |

测试建筑的76个条件列（75个唯一参数组合）分两套40槽完整评估，原始 Cost MAE 为 {summary['test']['cost_mae_raw']:.4f}。
四参数变化区、稳定区、响应幅度和方向指标保存在 summary.json。设施和背景误差分别统计。

输入是原生几何切片与高度合并后的 ground XYZ。教师和学生共用同一坐标数组，保留所有原生有限支撑点，包括障碍代价点。学生只有XYZ和四个机器人参数，Tiny使用官方预训练权重并冻结。适配层、FiLM和CostHead重新训练，不加载旧查询模型的任务头。

6栋现有建筑按原4/1/1划分。测试建筑曾用于几何诊断，本次不参与优化或选优。结果只覆盖本次六建筑实验。

完整独立推理只输入测试建筑PCD与四参数，先生成几何切片，再执行一次每窗口Tiny编码和代价预测。未读取教师标签或外部查询点。

- best.pt：完整所选模型及官方编码器。
- last.pt：第50轮训练状态、优化器和调度器。
- plots/training_curves.png：真实训练曲线。
- plots/：GT、预测和误差图。
- predictions/：全部切片点的完整测试预测PCD。
- independent_prediction/：从完整建筑PCD独立生成的预测。
- independent_inference_verification.json：独立路径与缓存评估路径的一致性。

旧权重及旧实验目录保持。此次没有服务器任务、定时任务、Filter/Recon训练或规划器修改。
'''
    text += '\n## 四参数响应\n\n'
    text += '| 参数 | 变化区差值MAE：初始化→所选 | 稳定区差值MAE | 响应回归增益 | 方向一致率 |\n'
    text += '|---|---:|---:|---:|---:|\n'
    for name in config['condition_names']:
        old, new = a['parameter_response'][name], b['parameter_response'][name]
        text += (f"| {name} | {old['changed_delta_mae']:.5f} → {new['changed_delta_mae']:.5f} "
                 f"| {new['stable_delta_mae']:.5f} | {new['response_gain']:.3f} "
                 f"| {new['direction_correct_pct']:.2f}% |\n")
    (output / 'REPORT.md').write_text(text)
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
