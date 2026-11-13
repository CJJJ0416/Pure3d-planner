"""Identical-data frozen/unfrozen cost comparison and actual-point-cloud regression."""
import argparse
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from Traversability.data.rotating import disk_bytes
from utils.pointcloud import read_json,write_json,workspace_path


def cost_metrics(predicted,target):
    error=predicted-target;low=target<=.5;high=~low
    return dict(mae_raw=float(50*abs(error).mean()),mae_normalized=float(abs(error).mean()),
        signed_bias_raw=float(50*error.mean()),false_block_pct=float(100*((predicted>.5)&low).sum()/low.sum()),
        false_pass_pct=float(100*((predicted<=.5)&high).sum()/high.sum()),
        false_block_denominator=int(low.sum()),false_pass_denominator=int(high.sum()))


def point_panels(path,xyz,truth,before,after,group,names,title,cmap):
    fig,axes=plt.subplots(len(names),4,figsize=(20,4.7*len(names)),sharex=True,sharey=True,squeeze=False,constrained_layout=True)
    panels=[('VST reference',truth),('Frozen Tiny | best epoch 45',before),('Unfrozen Tiny | selected',after),('Unfrozen absolute error',abs(after-truth))]
    for row,label in enumerate(names):
        take=group==row
        for col,(heading,value) in enumerate(panels):
            ax=axes[row,col]
            image=ax.scatter(xyz[take,0],xyz[take,1],c=value[take],s=.85,lw=0,cmap=cmap if col<3 else 'magma',
                vmin=0,vmax=1 if col<3 else .5,rasterized=True)
            if row==0:ax.set_title(heading)
            ax.set_aspect('equal');ax.grid(alpha=.15);ax.set(xlabel='Map X (m)',ylabel=label+'\nMap Y (m)')
            if col==0:cost_image=image
            if col==3:error_image=image
    fig.colorbar(cost_image,ax=axes[:,:3],shrink=.65,label='cost (0-1)')
    fig.colorbar(error_image,ax=axes[:,3],shrink=.65,label='Absolute error',extend='max')
    fig.suptitle(title)
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true',help='Reuse the completed comparison inference while redrawing plots.')
    args=parser.parse_args();output=args.run.resolve();config=read_json(output/'config.json');summary=read_json(output/'summary.json')
    frozen=workspace_path(config['frozen_reference_run']);reference=read_json(frozen/'summary.json')
    if not args.render_only:
        if not (output/'building_SUST_frozen/prediction.json').exists():
            subprocess.run([config['neural_python'],'-m','Traversability.inference.full3d',
                '--checkpoint',str(frozen/'best.pt'),'--pcd','/home/cjh/Pure3D_ws/datasets/rot_stairs/pcd_files/building_SUST.pcd',
                '--condition','.5','.6','.4','.05','--output',str(output/'building_SUST_frozen')],
                cwd='/home/cjh/Pure3D_ws',check=True)
        if not (output/'legacy_test/geometry/blocks.json').exists():
            subprocess.run([config['geometry_python'],'-m','Traversability.experiments.full3d_legacy_reference',
                '--run',str(output),'--prepare'],cwd='/home/cjh/Pure3D_ws',check=True)
        if not (output/'frozen_thresholds.json').exists():
            subprocess.run([config['neural_python'],'-m','Traversability.experiments.full3d_legacy_reference',
                '--run',str(output)],cwd='/home/cjh/Pure3D_ws',check=True)
    legacy_test=read_json(output/'legacy_test/metrics.json')
    for key,value in read_json(output/'frozen_thresholds.json').items():reference[key].update(value)
    history=read_json(output/'history.json');old_history=read_json(frozen/'history.json')
    figures=output/'plots';figures.mkdir(exist_ok=True)
    epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    for ax,name in zip(axes[0],['total','absolute','conditional']):
        ax.plot(epochs,[r['loss'][name] for r in old_history],label='Frozen')
        ax.plot(epochs,[r['loss'][name] for r in history],label='Unfrozen')
        ax.set(title=name+' training loss',ylabel='SmoothL1')
    axes[1,0].plot([0,*epochs],[reference['initial_validation']['cost_mae_raw'],*[r['validation']['cost_mae_raw'] for r in old_history]],label='Frozen')
    axes[1,0].plot([0,*epochs],[summary['initial_validation']['cost_mae_raw'],*[r['validation']['cost_mae_raw'] for r in history]],label='Unfrozen')
    axes[1,0].set(title='Fixed validation Cost MAE',ylabel='Raw cost (0-50)')
    for rows,label in [(old_history,'Frozen'),(history,'Unfrozen')]:
        axes[1,1].plot(epochs,[r['validation']['mean_response_error'] for r in rows],label=label)
    axes[1,1].set(title='Mean four-parameter response error',ylabel='Normalized delta MAE')
    axes[1,2].plot(epochs,[r['gradients']['encoder'] for r in history],label='Tiny gradient norm')
    axes[1,2].set(title='Tiny participates in backpropagation',ylabel='Last logged gradient norm',yscale='log')
    for ax in axes.flat:
        for boundary in range(10,50,10):ax.axvline(boundary,color='gray',lw=.6,alpha=.25)
        ax.set(xlabel='Epoch',xlim=(0,50));ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle(f'Same initialization and 20 seeds | full3D architecture unchanged | selected unfrozen epoch {summary["selected_epoch"]}')
    fig.savefig(figures/'training_frozen_vs_unfrozen.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    for col,name in enumerate(config['condition_names']):
        for row,field in enumerate(['changed_delta_mae','stable_delta_mae']):
            ax=axes[row,col]
            for records,label in [(old_history,'Frozen'),(history,'Unfrozen')]:
                ax.plot(epochs,[r['validation']['parameter_response'][name][field] for r in records],label=label)
            ax.set(title=name+' | '+('changed' if row==0 else 'stable'),xlabel='Epoch',ylabel='Delta MAE',xlim=(0,50));ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(figures/'parameter_response_errors.png',dpi=160);plt.close(fig)
    before=read_json(frozen/'final_responses.json');after=read_json(output/'final_responses.json')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,(ax,name) in enumerate(zip(axes.flat,config['condition_names'])):
        ids=np.arange(4+9*col,13+9*col)
        for scene,row in after.items():
            x=np.asarray(row['conditions'])[ids,col]
            for label,values,style in [('VST',row['truth'],'-o'),('Frozen',before[scene]['prediction'],'--o'),('Unfrozen',row['prediction'],'-o')]:
                ax.plot(x,np.asarray(values)[ids],style,label=label,markersize=3)
        ax.set(title=name,xlabel=name,ylabel='Mean facility support cost',ylim=(0,1));ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(figures/'parameter_response_curves.png',dpi=160);plt.close(fig)
    cmap=LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    data=workspace_path(config['data']);meta=read_json(data/'manifest.json')
    test=next(s for s in meta['scenes'] if s['split']=='test');name=test['id'];folder=data/name
    xyz=np.load(folder/'coord.npy');truth=np.load(folder/'cost.npy',mmap_mode='r')[:,3].astype(np.float32)
    old=np.load(frozen/'predictions'/f'{name}_prediction.npy');new=np.load(output/'predictions'/f'{name}_prediction.npy')
    independent=np.load(output/'independent_prediction/prediction.npy');independent_xyz=np.load(output/'independent_prediction/geometry/coord.npy')
    check=dict(xyz_equal=bool(np.array_equal(xyz,independent_xyz)),maximum_prediction_difference=float(abs(new-independent).max()),points=len(xyz))
    assert check['xyz_equal'] and check['maximum_prediction_difference']<1e-5
    write_json(output/'independent_inference_verification.json',check)
    floors=read_json(Path(test['source_pcd']).with_name('scene.json'))['config']['floor_z_m']
    group=np.searchsorted(np.asarray(floors[1:])-.05,xyz[:,2],side='right')
    point_panels(figures/f'{name}_GT_frozen_unfrozen_error.png',xyz,truth,old,new,group,
        [f'Floor {i} | base Z={z:.2f}m' for i,z in enumerate(floors)],f'{name} | all identical support XYZ | condition [0.5,0.6,0.4,0.05]',cmap)
    sust_ref=Path('/home/cjh/Pure3D_ws/runs/traversability/building_SUST_comparison_20261113')
    xyz=np.load(sust_ref/'vst/coord.npy');truth=np.load(sust_ref/'vst/cost.npy')
    new=np.load(output/'building_SUST/prediction.npy');old=np.load(output/'building_SUST_frozen/prediction.npy')
    for sub in ['building_SUST','building_SUST_frozen']:
        assert np.array_equal(xyz,np.load(output/sub/'geometry/coord.npy'))
    assert float(abs(old-np.load(sust_ref/'full3d/prediction.npy')).max())<1e-5
    legacy=np.load(sust_ref/'legacy_common/prediction.npy')
    sust=dict(frozen=cost_metrics(old,truth),unfrozen=cost_metrics(new,truth),legacy_querypool=cost_metrics(legacy,truth),
        same_xyz=True,points=len(xyz),condition=[.5,.6,.4,.05],vst_is_rule_reference=True,
        timing={label:read_json(output/sub/'prediction.json') for label,sub in [('frozen','building_SUST_frozen'),('unfrozen','building_SUST')]})
    write_json(output/'building_SUST_comparison.json',sust)
    bands=np.floor(xyz[:,2]/4).astype(int);ids,counts=np.unique(bands,return_counts=True)
    selected=np.sort(ids[np.argsort(counts)[-min(4,len(ids)):]])
    group=np.full(len(xyz),-1,np.int64)
    for i,b in enumerate(selected):group[bands==b]=i
    point_panels(figures/'building_SUST_GT_frozen_unfrozen_error.png',xyz,truth,old,new,group,
        [f'{b*4:g} <= Z < {(b+1)*4:g} m' for b in selected],
        f'building_SUST | identical XYZ | raw MAE: frozen {sust["frozen"]["mae_raw"]:.3f}, unfrozen {sust["unfrozen"]["mae_raw"]:.3f} | display bands only',cmap)
    rotations=[read_json(p) for p in sorted((output/'rotations').glob('*.json'))]
    state=read_json(data/'generation.json')
    labels=[]
    for seed in state['seeds']:
        name=f'scene_{seed}';a=np.load(data/name/'cost.npy',mmap_mode='r')
        b=np.load(workspace_path(config['frozen_reference_data'])/name/'cost.npy',mmap_mode='r')
        labels.append(dict(scene=name,shape=list(a.shape),cost_equal=bool(np.array_equal(a,b)),
            maximum_cost_difference=float(np.max(abs(a.astype(np.float32)-b.astype(np.float32))))))
    assert all(row['cost_equal'] for row in labels)
    write_json(output/'active_label_verification.json',dict(scenes=labels,
        note='Last four buildings retained by both runs. Earlier original label arrays were already removed by rolling data policy.'))
    storage=dict(active_seeds=state['seeds'],active_bytes=disk_bytes(data/'active'),fixed_cache_bytes=disk_bytes(data/'fixed'),
        archive_bytes=disk_bytes(output/'scene_archive'),all_prior_pools_removed=all(r['previous_active_removed'] and r['previous_wfc_cache_removed'] for r in rotations),
        rotations=len(rotations),scenes_archived=len(list((output/'scene_archive').glob('scene_*'))))
    assert storage['all_prior_pools_removed'] and storage['rotations']==4 and storage['scenes_archived']==20
    write_json(output/'storage_verification.json',storage)
    verification=read_json(output/'training_verification.json')
    checks=[read_json(p) for p in sorted((output/'data_verification').glob('*.json'))]
    assert len(checks)==5 and all(all(v for k,v in row.items() if k!='scene') for rows in checks for row in rows)
    text=f'''# 完整3D模型：不冻结Tiny，同起点50轮对照

从相同切片版第90轮best初始化，复现相同20个训练seed、同一条件库和标签。
Tiny从第1步参与反向，完整3D结构、5 cm体素、分块与GPU对应不变。
任务头峰值LR=1e-4，Tiny=1e-5。原100轮学习率阶段保留，第50轮停止。
训练态启用Tiny原有随机层，验证使用eval。全程在线编码，没有复用冻结特征。

本轮选择第 **{summary['selected_epoch']}** 轮。初始化也参与选优。

| 指标 | 冻结Tiny，best45 | 不冻结Tiny，本轮best |
|---|---:|---:|
| 固定验证原始MAE | {reference['best_validation']['cost_mae_raw']:.5f} | {summary['best_validation']['cost_mae_raw']:.5f} |
| 固定验证四参数平均响应误差 | {reference['best_validation']['mean_response_error']:.6f} | {summary['best_validation']['mean_response_error']:.6f} |
| 固定验证选优分数 | {reference['best_validation']['score']:.6f} | {summary['best_validation']['score']:.6f} |
| 固定测试原始MAE | {reference['test']['cost_mae_raw']:.5f} | {summary['test']['cost_mae_raw']:.5f} |
| building_SUST原始MAE | {sust['frozen']['mae_raw']:.5f} | {sust['unfrozen']['mae_raw']:.5f} |

## 四参数响应：固定验证

| 参数 | 变化区差值MAE：冻结→不冻结 | 稳定区差值MAE：冻结→不冻结 | 响应增益：冻结→不冻结 |
|---|---:|---:|---:|
'''
    for name in config['condition_names']:
        a=reference['best_validation']['parameter_response'][name];b=summary['best_validation']['parameter_response'][name]
        text+=f"| {name} | {a['changed_delta_mae']:.5f} → {b['changed_delta_mae']:.5f} | {a['stable_delta_mae']:.5f} → {b['stable_delta_mae']:.5f} | {a['response_gain']:.4f} → {b['response_gain']:.4f} |\n"
    text+='\n## 固定测试：两套条件库，三个模型相同XYZ\n\n| 模型 | 原始MAE | 四参数平均响应误差 | 误阻断率 | 错误放行率 |\n|---|---:|---:|---:|---:|\n'
    test_models=[('冻结Tiny',reference['test']),('不冻结Tiny',summary['test']),('旧QueryPool',legacy_test['metrics'])]
    for label,row in test_models:
        text+=f"| {label} | {row['cost_mae_raw']:.5f} | {row['mean_response_error']:.6f} | {row['false_block_pct']:.2f}% | {row['false_pass_pct']:.2f}% |\n"
    text+='\n所有80条件槽均保留，两个错误率分别以GT低代价、高代价点数作分母。旧QueryPool使用自身原有6×6×4 m分块、局部及全局查询和原精度，不改变权重。旧QueryPool的训练数据和预算不同，作为历史参考，不能将其差异单独归因于网络结构。\n'
    text+='\n| 参数 | 冻结：变化／稳定误差 | 微调：变化／稳定误差 | 旧QueryPool：变化／稳定误差 |\n|---|---:|---:|---:|\n'
    for name in config['condition_names']:
        values=[row['parameter_response'][name] for _,row in test_models]
        text+='| '+name+' | '+' | '.join(f"{v['changed_delta_mae']:.5f} / {v['stable_delta_mae']:.5f}" for v in values)+' |\n'
    text+='\n## building_SUST：相同XYZ与参数\n\n| 模型 | 原始MAE | 误阻断率 | 错误放行率 | 完整PCD推理内部耗时 |\n|---|---:|---:|---:|---:|\n'
    for name,label in [('frozen','冻结Tiny'),('unfrozen','不冻结Tiny'),('legacy_querypool','旧QueryPool，第20轮')]:
        row=sust[name];seconds=f"{sust['timing'][name]['total_s']:.3f}s" if name!='legacy_querypool' else f"{read_json(sust_ref/'legacy_raw/prediction.json')['entry_to_prediction_files_s']:.3f}s（原始点输出）"
        text+=f"| {label} | {row['mae_raw']:.4f} | {row['false_block_pct']:.2f}% | {row['false_pass_pct']:.2f}% | {seconds} |\n"
    text+='\n阈值0.5，分别按VST低代价与高代价点数作分母。VST是规则参考，SUST未用于训练或选优。旧QueryPool采用相同XYZ直接查询结果。\n'
    text+='\n### 本次SUST推理分项耗时\n\n| 阶段 | 冻结Tiny（秒） | 不冻结Tiny（秒） |\n|---|---:|---:|\n'
    for key in sust['timing']['unfrozen']['timing']:
        text+=f"| {key} | {sust['timing']['frozen']['timing'][key]:.4f} | {sust['timing']['unfrozen']['timing'][key]:.4f} |\n"
    text+='\n上述为单次进程内实测，包含模型准备和输出。两种完整3D模型输出相同支撑点。旧QueryPool原生时间输出全部原始点，输出规模不同。\n'
    regressions=[]
    for split,label in [('best_validation','固定验证'),('test','固定测试')]:
        a=reference[split];b=summary[split]
        for key in ['cost_mae_raw','mean_response_error','false_block_pct','false_pass_pct']:
            if key in a and key in b and b[key]>a[key]:regressions.append(f"{label} {key}: {a[key]:.6f} → {b[key]:.6f}")
        for name in config['condition_names']:
            for key in ['changed_delta_mae','stable_delta_mae']:
                av=a['parameter_response'][name][key];bv=b['parameter_response'][name][key]
                if bv>av:regressions.append(f'{label} {name} {key}: {av:.6f} → {bv:.6f}')
    for key in ['mae_raw','false_block_pct','false_pass_pct']:
        a=sust['frozen'][key];b=sust['unfrozen'][key]
        if b>a:regressions.append(f'building_SUST {key}: {a:.6f} → {b:.6f}')
    text+='\n## 相对冻结版的退化项\n\n'+('\n'.join('- '+line for line in regressions) if regressions else '本次列出的绝对MAE、条件变化／稳定误差及阈值错误率均未退化。')+'\n'
    text+=f'''
## 核验与文件

- 初始化预测与冻结版最大差异见initialization_verification.json。
- 首次真实batch2的Tiny与三任务头梯度和参数变化见first_update.json。
- 第1轮真实恢复记录见resume_verification.json，恢复实际学到的Tiny权重。
- 累计{verification['last_step']}次更新，与冻结版相同。Tiny状态中{verification['encoder_updated_keys']}个键发生变化。
- 所有20栋建筑的原始PCD、查询XYZ、条件与配对均与冻结版核对，见data_verification/及scene_archive/。
- 两次运行仍保留的最后4栋建筑，其全部标签矩阵逐值一致，见active_label_verification.json。此前16栋的原标签已按原轮换策略清除。
- 当前训练池{storage['active_bytes']/2**30:.2f} GiB。仅当前4栋保留大文件，4次批次替换已清除旧文件和WFC缓存。
- best.pt、last.pt保存实际训练后的编码器、任务头、优化器和随机状态，原权重不覆盖。
- predictions/、independent_prediction/和building_SUST/提供完整预测PCD。
- plots/提供训练曲线、条件响应和GT／冻结／微调／误差图。

结果用于本轮固定合成留出建筑和building_SUST对比。未接入Filter/Recon，未修改规划器，未创建定时任务。
'''
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':main()
