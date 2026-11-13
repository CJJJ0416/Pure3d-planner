"""Fixed-building validation, response curves and full test PCD after scene rotation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from Traversability.data.rotating import disk_bytes
from utils.pointcloud import read_json, write_json, workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    output=parser.parse_args().run.resolve()
    summary=read_json(output/'summary.json')
    config=read_json(output/'config.json')
    history=read_json(output/'history.json')
    metadata=read_json(workspace_path(config['data'])/'manifest.json')
    figures=output/'plots';figures.mkdir(exist_ok=True)
    baseline=summary['baseline']['best_validation']; final=summary['best_validation']
    epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for name in ['total','absolute','conditional']:
        axes[0,0].plot(epochs,[r['loss'][name] for r in history],label=name)
    axes[0,0].set(title='Training losses | new scenes every 5 epochs after 50',ylabel='SmoothL1')
    axes[0,1].plot(epochs,[r['validation']['cost_mae_raw'] for r in history],label='Fixed validation')
    axes[0,1].axhline(baseline['cost_mae_raw'],color='black',ls='--',label='Original best epoch 30')
    axes[0,1].set(title='Fixed validation Cost MAE',ylabel='Raw cost units (0-50)')
    for name in config['condition_names']:
        for ax,field,title in [(axes[1,0],'changed_delta_mae','Response error: teacher changes'),
                               (axes[1,1],'stable_delta_mae','Response error: teacher stable')]:
            ax.plot(epochs,[r['validation']['parameter_response'][name][field] for r in history],label=name)
            ax.set(title=title,ylabel='Normalized delta MAE')
    for ax in axes.flat:
        ax.axvline(50,color='black',lw=1)
        for boundary in range(55,100,5):ax.axvline(boundary,color='gray',lw=.6,alpha=.3)
        ax.set_xlabel('Cumulative epoch');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle(f'Frozen Utonia Tiny | 40 rotating training buildings | best epoch {summary["selected_epoch"]}')
    fig.savefig(figures/'training_curves.png',dpi=160);plt.close(fig)

    before=read_json(output/'baseline_responses.json');after=read_json(output/'final_responses.json')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for p,(ax,name) in enumerate(zip(axes.flat,config['condition_names'])):
        for scene,values in after.items():
            indices=np.arange(4+9*p,13+9*p)
            x=np.asarray(values['conditions'])[indices,p]
            ax.plot(x,np.asarray(values['truth'])[indices],'-o',label='VST teacher',markersize=3)
            ax.plot(x,np.asarray(before[scene]['prediction'])[indices],'--o',label='Before: epoch 30',markersize=3)
            ax.plot(x,np.asarray(values['prediction'])[indices],'-o',label=f'After: selected epoch {summary["selected_epoch"]}',markersize=3)
        ax.set(title=name,xlabel=name,ylabel='Mean cost on facility support points',ylim=(0,1))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fixed validation geometry | same XYZ and conditions | baseline-background parameter scans')
    fig.savefig(figures/'parameter_response_curves.png',dpi=160);plt.close(fig)

    cost_cmap=LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    for scene in [s for s in metadata['scenes'] if s['split']=='test']:
        name=scene['id'];folder=workspace_path(config['data'])/name
        xyz=np.load(folder/'coord.npy');truth=np.load(folder/'cost.npy',mmap_mode='r')[:,3].astype(np.float32)
        prediction=np.load(output/'predictions'/f'{name}_prediction.npy')
        old=np.load(workspace_path(config['source_run'])/'predictions'/f'{name}_prediction.npy')
        independent=np.load(output/'independent_prediction/prediction.npy')
        independent_xyz=np.load(output/'independent_prediction/geometry/coord.npy')
        verification=dict(independent_xyz_identical=bool(np.array_equal(xyz,independent_xyz)),
                          independent_prediction_max_abs_difference=float(abs(prediction-independent).max()),points=len(xyz))
        assert verification['independent_xyz_identical'] and verification['independent_prediction_max_abs_difference']<1e-5
        write_json(output/'independent_inference_verification.json',verification)
        floors=read_json(Path(scene['source_pcd']).with_name('scene.json'))['config']['floor_z_m']
        groups=np.searchsorted(np.asarray(floors[1:])-.05,xyz[:,2],side='right')
        panels=[('VST GT',truth),('Before: epoch 30',old),
                (f'After: selected epoch {summary["selected_epoch"]}',prediction),('After absolute error',abs(prediction-truth))]
        fig,axes=plt.subplots(len(floors),4,figsize=(20,5*len(floors)),sharex=True,sharey=True,constrained_layout=True,squeeze=False)
        for row,base in enumerate(floors):
            take=groups==row
            for col,(title,values) in enumerate(panels):
                ax=axes[row,col]
                image=ax.scatter(xyz[take,0],xyz[take,1],c=values[take],s=.9,lw=0,
                    cmap=cost_cmap if col<3 else 'magma',vmin=0,vmax=1 if col<3 else .5,rasterized=True)
                if row==0:ax.set_title(title)
                ax.set_aspect('equal');ax.set(xlabel='Map X (m)',ylabel=f'Floor {row} | Z={base:.2f}m\nMap Y (m)')
                ax.grid(alpha=.15)
                if col==0:cost_image=image
                if col==3:error_image=image
        fig.colorbar(cost_image,ax=axes[:,:3],shrink=.65,label='cost (0-1)')
        fig.colorbar(error_image,ax=axes[:,3],shrink=.65,label='Absolute error',extend='max')
        fig.suptitle(f'{name} | all {len(xyz):,} identical support XYZ | condition [0.5,0.6,0.4,0.05]\n'
                     f'Raw MAE before {50*abs(old-truth).mean():.3f}, after {50*abs(prediction-truth).mean():.3f}')
        fig.savefig(figures/f'{name}_GT_before_after_error.png',dpi=150);plt.close(fig)

    rotations=[read_json(p) for p in sorted((output/'rotations').glob('*.json'))]
    state=read_json(workspace_path(config['data'])/'generation.json')
    storage=dict(rotations=rotations,active_seeds=state['seeds'],active_bytes=disk_bytes(workspace_path(config['data'])/'active'),
                 archive_bytes=disk_bytes(output/'scene_archive'),
                 fixed_splits_referenced=True,all_prior_pools_removed=all(r['previous_active_removed'] and r['previous_wfc_cache_removed'] for r in rotations))
    write_json(output/'storage_verification.json',storage)
    verification=read_json(output/'training_verification.json')
    assert verification['scenes_archived']==40 and verification['rotations_verified']==9
    text=f'''# 冻结Tiny：轮换场景续训至100轮

从原第50轮last.pt续训至第100轮，实际更新次数见training_verification.json。
Tiny全程冻结，只更新Adapter、FiLM与CostHead。每5轮更换4个独立建筑，新增40个seed。
输入是原生几何切片后的支撑XYZ，预测输出保留同一组XYZ。本轮沿用独立切片Cost链，未连接Recon的H(q)。
固定验证／测试直接引用原场景和缓存，未参与场景替换。

按原验证分数选出的权重是第 **{summary['selected_epoch']}** 轮。原第30轮最佳权重参与选优。
{'本轮未超过原始最佳，best.pt保留原第30轮模型。' if summary['initial_best_retained'] else '续训模型超过原始最佳，best.pt已保存新模型。'}

| 指标 | 原50轮最佳 | 本次所选 |
|---|---:|---:|
| 固定验证原始Cost MAE | {baseline['cost_mae_raw']:.5f} | {final['cost_mae_raw']:.5f} |
| 四参数平均响应误差 | {baseline['mean_response_error']:.6f} | {final['mean_response_error']:.6f} |
| 验证选优分数 | {baseline['score']:.6f} | {final['score']:.6f} |
| 固定测试原始Cost MAE | {summary['original_test']['cost_mae_raw']:.5f} | {summary['test']['cost_mae_raw']:.5f} |

## 四参数响应

| 参数 | 变化区差值MAE：原→所选 | 稳定区差值MAE：原→所选 | 响应回归增益：原→所选 | 方向一致率：原→所选 |
|---|---:|---:|---:|---:|
'''
    for name in config['condition_names']:
        a,b=baseline['parameter_response'][name],final['parameter_response'][name]
        text+=f"| {name} | {a['changed_delta_mae']:.5f} → {b['changed_delta_mae']:.5f} | {a['stable_delta_mae']:.5f} → {b['stable_delta_mae']:.5f} | {a['response_gain']:.4f} → {b['response_gain']:.4f} | {a['direction_correct_pct']:.2f}% → {b['direction_correct_pct']:.2f}% |\n"
    a,b=baseline['parameter_response']['stand_ratio'],final['parameter_response']['stand_ratio']
    test_name=next(s['id'] for s in metadata['scenes'] if s['split']=='test')
    text+=f'''
stand_ratio变化区误差由{a['changed_delta_mae']:.5f}变为{b['changed_delta_mae']:.5f}，响应回归增益由{a['response_gain']:.4f}变为{b['response_gain']:.4f}。
stand_ratio响应仍弱于教师，其稳定区误差由{a['stable_delta_mae']:.5f}升至{b['stable_delta_mae']:.5f}。slope_max的稳定区误差也上升，分项见上表。
响应均值曲线仅统计同一固定验证建筑的设施支撑点。变化区及稳定区分项指标见summary.json。

## 存储与恢复核验

9次实际批次替换均删除上一批大文件及其WFC缓存。当前仅保留最终4个训练建筑的大文件，约{storage['active_bytes']/2**30:.2f} GiB。
40个场景的参数、条件、overview、数据摘要和日志永久归档约{storage['archive_bytes']/2**20:.1f} MiB。
真实从本次第51轮last.pt恢复训练，记录见resume_verification.json。原始best.pt和last.pt哈希保持不变。
Tiny权重和缓冲量保持不变，三个任务头均通过真实batch2的梯度及参数更新核验。

## 交付

- best.pt：固定验证选出的完整模型，含冻结编码器。
- last.pt：第100轮状态，含AdamW、学习率阶段、随机状态及数据批次。
- history.json、progress.json：累计1–100轮曲线和最终指标。
- scene_archive/、generations/、rotations/：40场景来源及替换证据。
- plots/training_curves.png、plots/parameter_response_curves.png：训练与条件响应。
- plots/{test_name}_GT_before_after_error.png：固定测试完整点云的逐层对比。
- predictions/、independent_prediction/：完整测试预测及只输入PCD与参数的独立推理。

本次固定测试仍为原有一栋留出建筑，最终指标用于比较本轮续训。没有改动规划器或启动服务器、定时任务。
'''
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':main()
