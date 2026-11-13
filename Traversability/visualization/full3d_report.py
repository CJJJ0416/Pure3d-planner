"""Full-3D training evidence, original sliced reference, and identical-query comparisons."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    output=parser.parse_args().run.resolve()
    summary=read_json(output/'summary.json'); config=read_json(output/'config.json')
    history=read_json(output/'history.json'); data=workspace_path(config['data'])
    meta=read_json(data/'manifest.json')
    figures=output/'plots';figures.mkdir(exist_ok=True)
    initial=summary['initial_validation']; final=summary['best_validation']
    old=summary['baseline']['old_sliced_validation']
    completed=summary['completed_epochs'];scene_count=summary['new_scenes'];generations=summary['generations']
    epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for name in ['total','absolute','conditional']:
        axes[0,0].plot(epochs,[r['loss'][name] for r in history],label=name)
    axes[0,0].set(title='Training loss | four new buildings every 10 epochs',ylabel='SmoothL1')
    axes[0,1].plot([0,*epochs],[initial['cost_mae_raw'],*[r['validation']['cost_mae_raw'] for r in history]],label='Full3D validation')
    axes[0,1].axhline(old['cost_mae_raw'],ls='--',color='black',label='Old sliced encoder input | epoch 90')
    axes[0,1].set(title='Fixed validation Cost MAE',ylabel='Raw cost (0-50)')
    for name in config['condition_names']:
        for ax,field,title in [(axes[1,0],'changed_delta_mae','Response error: teacher changes'),
                              (axes[1,1],'stable_delta_mae','Response error: teacher stable')]:
            ax.plot([0,*epochs],[initial['parameter_response'][name][field],
                *[r['validation']['parameter_response'][name][field] for r in history]],label=name)
            ax.set(title=title,ylabel='Normalized delta MAE')
    for ax in axes.flat:
        for boundary in range(10,completed,10):ax.axvline(boundary,color='gray',alpha=.25,lw=.7)
        ax.set_xlim(0,completed)
        ax.set_xlabel('New training epoch');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle(f'Frozen Tiny | complete 3D input | {scene_count} rotating buildings | selected epoch {summary["selected_epoch"]}')
    fig.savefig(figures/'training_curves.png',dpi=160);plt.close(fig)
    before=read_json(output/'baseline_responses.json');after=read_json(output/'final_responses.json')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for p,(ax,name) in enumerate(zip(axes.flat,config['condition_names'])):
        ids=np.arange(4+9*p,13+9*p)
        for scene,values in after.items():
            x=np.asarray(values['conditions'])[ids,p]
            for label,array,style in [('VST GT',values['truth'],'-o'),
                ('Full3D initialization',before[scene]['prediction'],'--o'),('Selected full3D',values['prediction'],'-o')]:
                ax.plot(x,np.asarray(array)[ids],style,label=label,markersize=3)
        ax.set(title=name,xlabel=name,ylabel='Mean facility support cost',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(figures/'parameter_response_curves.png',dpi=160);plt.close(fig)
    cmap=LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    for scene in [s for s in meta['scenes'] if s['split']=='test']:
        name=scene['id'];folder=data/name
        xyz=np.load(folder/'coord.npy');truth=np.load(folder/'cost.npy',mmap_mode='r')[:,3].astype(np.float32)
        predicted=np.load(output/'predictions'/f'{name}_prediction.npy')
        init=np.load(output/'initial_predictions'/f'{name}_prediction.npy')
        sliced=np.load(workspace_path(config['source_run'])/'predictions'/f'{name}_prediction.npy')
        independent=np.load(output/'independent_prediction/prediction.npy')
        coords=np.load(output/'independent_prediction/geometry/coord.npy')
        verification=dict(independent_xyz_identical=bool(np.array_equal(xyz,coords)),
            independent_prediction_max_abs_difference=float(abs(predicted-independent).max()),points=len(xyz))
        assert verification['independent_xyz_identical'] and verification['independent_prediction_max_abs_difference']<1e-5
        write_json(output/'independent_inference_verification.json',verification)
        floors=read_json(Path(scene['source_pcd']).with_name('scene.json'))['config']['floor_z_m']
        group=np.searchsorted(np.asarray(floors[1:])-.05,xyz[:,2],side='right')
        panels=[('VST GT',truth),('Old sliced input | epoch 90',sliced),('Full3D initialization',init),
            (f'Full3D selected | epoch {summary["selected_epoch"]}',predicted),('Selected absolute error',abs(predicted-truth))]
        fig,axes=plt.subplots(len(floors),5,figsize=(24,4.8*len(floors)),sharex=True,sharey=True,constrained_layout=True,squeeze=False)
        for r,base in enumerate(floors):
            keep=group==r
            for c,(label,values) in enumerate(panels):
                ax=axes[r,c]
                image=ax.scatter(xyz[keep,0],xyz[keep,1],c=values[keep],s=.9,lw=0,
                    cmap=cmap if c<4 else 'magma',vmin=0,vmax=1 if c<4 else .5,rasterized=True)
                if r==0:ax.set_title(label)
                ax.set_aspect('equal');ax.set(xlabel='Map X (m)',ylabel=f'Floor {r} | Z={base:.2f}m\nMap Y (m)');ax.grid(alpha=.15)
                if c==0:cost_image=image
                if c==4:error_image=image
        fig.colorbar(cost_image,ax=axes[:,:4],shrink=.65,label='cost (0-1)')
        fig.colorbar(error_image,ax=axes[:,4],shrink=.65,label='Absolute error',extend='max')
        fig.suptitle(f'{name} | all {len(xyz):,} identical support queries | condition [0.5,0.6,0.4,0.05]\n'
            f'Raw MAE: old sliced {50*abs(sliced-truth).mean():.3f}, full3D init {50*abs(init-truth).mean():.3f}, selected {50*abs(predicted-truth).mean():.3f}')
        fig.savefig(figures/f'{name}_GT_before_after_error.png',dpi=150);plt.close(fig)
    rotations=[read_json(p) for p in sorted((output/'rotations').glob('*.json'))]
    generation=read_json(data/'generation.json')
    storage=dict(rotations=rotations,active_seeds=generation['seeds'],active_bytes=disk_bytes(data/'active'),
        fixed_cache_bytes=disk_bytes(data/'fixed'),archive_bytes=disk_bytes(output/'scene_archive'),
        all_prior_pools_removed=all(r['previous_active_removed'] and r['previous_wfc_cache_removed'] for r in rotations))
    write_json(output/'storage_verification.json',storage)
    assert len(rotations)==generations-1 and storage['all_prior_pools_removed']
    encodings=[]
    for path in sorted((output/'generations').glob('*.json')):
        encodings += [s for s in read_json(path)['encoding']['scenes'] if s['scene'].startswith('scene_20261114') and s['scene'] not in ['scene_20261114']]
    timings={name:float(sum(r['timing'][name] for r in encodings)) for name in
        ['pcd_read_s','gpu_voxel_s','gpu_correspondence_s','grouping_s','tiny_encoding_s','cache_write_s','total_s']}
    assert len(encodings)==scene_count
    for record in encodings:
        write_json(output/'scene_archive'/record['scene']/'full3d_encoding.json', record)
    write_json(output/'timing_summary.json',dict(training_scene_count=scene_count,preparation=timings,
        head_training_s=sum(r['elapsed_s'] for r in history),fixed_cache=[read_json(data/s['id']/'full3d_encoding.json') for s in meta['scenes'] if s['split']!='train']))
    # Source XYZ and actual matches are inspectable without loading Tiny or the teacher.
    for scene in [s for s in meta['scenes'] if s['split']!='train']:
        folder=data/scene['id']; q=np.load(folder/'coord.npy'); reps=np.load(folder/'geometry_coord.npy',mmap_mode='r')
        idx=np.load(folder/'query_representative_index.npy'); origin=np.load(folder/'query_original_index.npy')
        chosen=np.linspace(0,len(q)-1,min(128,len(q)),dtype=int)
        result=np.column_stack([chosen,q[chosen],origin[chosen],reps[idx[chosen]],np.linalg.norm(q[chosen]-reps[idx[chosen]],axis=1)])
        np.savetxt(output/f'{scene["id"]}_matching_audit.csv',result,delimiter=',',
            header='query_index,qx,qy,qz,original_point_index,source_x,source_y,source_z,distance_m',comments='')
    text=f'''# 完整3D输入，冻结Tiny：{completed}轮结果

从原第90轮best.pt初始化Adapter、FiLM、CostHead，新建AdamW，重新训练{completed}轮。
完整建筑PCD在GPU上按5 cm体素保留真实原始点。Tiny输入完整3D上下文，Tomogram仅提供监督位置与标签。
每10轮替换4栋训练建筑，共{scene_count}个新seed。固定验证、测试的XYZ、条件与标签保持，特征全部重建。
原学习率调度预算保留为{config['epochs']}轮。{'按用户要求在第50轮停止并测试。' if completed==50 else ''}

选中第 **{summary['selected_epoch']}** 轮。{'初始化仍为最佳，本轮未改善完整3D输入的验证分数。' if summary['initial_best_retained'] else '训练超过完整3D输入初始化。'}

| 指标 | 旧切片输入，第90轮 | 完整3D初始化 | 本轮所选 |
|---|---:|---:|---:|
| 固定验证原始Cost MAE | {old['cost_mae_raw']:.5f} | {initial['cost_mae_raw']:.5f} | {final['cost_mae_raw']:.5f} |
| 四参数平均响应误差 | {old['mean_response_error']:.6f} | {initial['mean_response_error']:.6f} | {final['mean_response_error']:.6f} |
| 选优分数 | {old['score']:.6f} | {initial['score']:.6f} | {final['score']:.6f} |
| 固定测试原始Cost MAE | {summary['original_test']['cost_mae_raw']:.5f} | {summary['initial_test']['cost_mae_raw']:.5f} | {summary['test']['cost_mae_raw']:.5f} |

旧切片指标仅作参考，完整3D初始化和各新轮次使用相同完整3D缓存选优。

## 四参数响应

| 参数 | 变化区差值MAE：初始化→所选 | 稳定区差值MAE：初始化→所选 | 响应增益：初始化→所选 | 方向一致率：初始化→所选 |
|---|---:|---:|---:|---:|
'''
    for name in config['condition_names']:
        a,b=initial['parameter_response'][name],final['parameter_response'][name]
        text+=f"| {name} | {a['changed_delta_mae']:.5f} → {b['changed_delta_mae']:.5f} | {a['stable_delta_mae']:.5f} → {b['stable_delta_mae']:.5f} | {a['response_gain']:.4f} → {b['response_gain']:.4f} | {a['direction_correct_pct']:.2f}% → {b['direction_correct_pct']:.2f}% |\n"
    text+='\n## 与旧第90轮模型的条件响应对比\n\n| 参数 | 变化区差值MAE：旧→本轮 | 稳定区差值MAE：旧→本轮 | 响应增益：旧→本轮 |\n|---|---:|---:|---:|\n'
    for name in config['condition_names']:
        a,b=old['parameter_response'][name],final['parameter_response'][name]
        text+=f"| {name} | {a['changed_delta_mae']:.5f} → {b['changed_delta_mae']:.5f} | {a['stable_delta_mae']:.5f} → {b['stable_delta_mae']:.5f} | {a['response_gain']:.4f} → {b['response_gain']:.4f} |\n"
    text+='\n整体MAE和平均条件响应误差改善。stand_ratio相对旧第90轮模型的变化区与稳定区误差均略升，响应增益下降，仍存在响应不足。step_max稳定区误差也略升，分项见上表。\n'
    text+=f'\n## 几何预处理累计耗时\n\n| 项目 | {scene_count}栋训练建筑累计秒数 |\n|---|---:|\n'
    for name,value in timings.items():text+=f'| {name} | {value:.3f} |\n'
    text+=f'''
Tiny权重与缓冲量不变。真实batch2验证和第1轮断点恢复记录保存在first_update.json、resume_verification.json及training_verification.json。
GPU匹配使用三维距离，固定验证和测试均保存128点穷举核验、来源坐标及匹配距离。
{len(rotations)}次轮换已清除旧大文件及WFC缓存。最终训练池{storage['active_bytes']/2**30:.2f} GiB，固定新缓存{storage['fixed_cache_bytes']/2**30:.2f} GiB。

- best.pt、last.pt：完整模型及训练断点。
- plots/training_curves.png、plots/parameter_response_curves.png：真实训练与条件响应。
- plots/*_GT_before_after_error.png：GT、旧切片、完整3D初始化、所选模型和误差。
- predictions/、independent_prediction/：完整固定测试预测PCD与独立推理。
- scene_archive/、generations/、rotations/：{scene_count}栋建筑参数、overview、条件、缓存与清理记录。
- progress.json：最终状态、指标和数据占用。

本轮固定测试为同一栋留出建筑。结果用于对比输入方式及此次训练，规划器未修改。
'''
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
