"""Report A/B training, independent scenes, four-parameter response and R4 regression."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from types import SimpleNamespace
from utils.pointcloud import ROOT,read_json,write_json,workspace_path
from Traversability.experiments.stair_metrics import prepare_evaluation,StairMetrics
from Traversability.experiments.wide_stair_report import fixed_regression

NAMES=['clearance_min','slope_max','step_max','stand_ratio']
COST='RdYlGn_r'
plt.rcParams.update({'font.size':10,'figure.facecolor':'white','savefig.facecolor':'white'})


def savefig(fig,path):
    fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def pair_metric(gt,pred,left,right,mask=None):
    if mask is None:mask=np.ones(len(gt),bool)
    dg=(gt[:,right]-gt[:,left])[mask];dp=(pred[:,right]-pred[:,left])[mask]
    active=abs(dg)>.02
    return dict(changed_points=int(active.sum()),GT_mean_abs_delta=float(abs(dg[active]).mean()) if active.any() else None,
        student_mean_abs_delta=float(abs(dp[active]).mean()) if active.any() else None,
        amplitude_ratio=float(abs(dp[active]).mean()/abs(dg[active]).mean()) if active.any() else None,
        direction_correct_pct=float((dp[active]*dg[active]>0).mean()*100) if active.any() else None,
        stable_mean_abs_drift=float(abs(dp[~active]).mean()) if (~active).any() else None,
        delta_mae=float(abs(dp-dg).mean()))


def r4_report(run):
    source=ROOT/'runs/planner/r4_step_slope_response_20261111'
    teacher=np.load(source/'teacher.npz');gt=teacher['target']
    before=np.load(run/'initial_r4.npz')['prediction'];after=np.load(run/'best_r4.npz')['prediction']
    tread=teacher['step']>0
    result={name:dict(step_030_040=pair_metric(gt,pred,2,8),
        slope_018_070=pair_metric(gt,pred,12,17,tread)) for name,pred in [('before',before),('after',after)]}
    profiles=read_json(source/'protocol.json')['profiles']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,axis in [(axes[0,0],'step_max'),(axes[0,1],'slope_max')]:
        ids=[i for i,p in enumerate(profiles) if p['axis']==axis]
        x=[profiles[i]['value'] for i in ids]
        for label,values,color in [('GT',gt,'#059669'),('Before FP32',before,'#64748b'),('After FP32',after,'#a21caf')]:
            ax.plot(x,values[tread][:,ids].mean(0),'-o',label=label,c=color)
        if axis=='step_max':ax.axvline(.3657008067382322,c='#475569',ls='--',label='Actual riser')
        else:ax.axvspan(.7,1.35,color='#fef3c7',alpha=.5,label='Outside training range')
        ax.set(xlabel=axis,ylabel='Mean cost on 261 treads',title='R4 '+axis);ax.legend(fontsize=8);ax.grid(alpha=.2)
    for ax,key in [(axes[1,0],'step_030_040'),(axes[1,1],'slope_018_070')]:
        if key.startswith('step'):
            x=np.arange(2);ax.bar(x,[result[n][key]['amplitude_ratio'] for n in ['before','after']],color=['#64748b','#a21caf'])
            ax.axhspan(.7,1.3,alpha=.15,color='green');ax.set(xticks=x,xticklabels=['Before','After'],ylabel='Student / GT response amplitude',title='step_max 0.30 to 0.40')
        else:
            ax.bar([0,1],[result[n][key]['stable_mean_abs_drift'] for n in ['before','after']],color=['#64748b','#a21caf'])
            ax.axhline(.02,c='green',ls='--');ax.set(xticks=[0,1],xticklabels=['Before','After'],ylabel='Mean |student change| on stable GT treads',title='slope_max 0.18 to 0.70')
    savefig(fig,run/'plots/r4_parameter_response.png')
    fig,axes=plt.subplots(1,3,figsize=(12,10),sharex=True,sharey=True,layout='constrained')
    for ax,title,values in zip(axes,['GT','Before FP32','After FP32'],[gt[:,8],before[:,8],after[:,8]]):
        ax.scatter(teacher['across'],teacher['display'],c=values,cmap=COST,vmin=0,vmax=1,s=15,lw=0)
        ax.set(title=title,xticks=[0,.5848,1.1696],xticklabels=['Left','Centre','Right'],yticks=range(11),
            yticklabels=['Entry','1','2','3','4','Turn','5','6','7','8','Exit'])
        ax.spines[['top','right']].set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=COST),ax=axes,location='bottom',aspect=40,label='Cost 0-1')
    fig.suptitle('R4 | identical 1,440 support XYZ | condition [0.5, 0.6, 0.4, 0.05]')
    savefig(fig,run/'plots/r4_GT_before_after.png')
    write_json(run/'r4_response.json',result)
    return result


def auxiliary_connectivity(run,data,before,after):
    meta=read_json(data/'evaluation/blocks.json')
    meta['scenes']=[s for s in meta['scenes'] if s['split']=='test' and s['source']=='new']
    names={s['id'] for s in meta['scenes']}
    meta['blocks']=[b for b in meta['blocks'] if b['scene'] in names]
    folder=run/'aux_connectivity';write_json(folder/'blocks.json',meta)
    prepare_evaluation(folder)
    meta=read_json(folder/'blocks.json');report={}
    for label,snapshot in [('before',before),('after',after)]:
        monitor=StairMetrics(SimpleNamespace(metadata=meta,rows=meta['blocks']))
        for name,s in monitor.scenes.items():
            mask=(snapshot['scene']==name)&(snapshot['bank']==0);ids=snapshot['query_index'][mask]
            monitor.prediction[name][ids]=snapshot['prediction'][mask,:4]
            outside=~monitor.mask[name][ids]
            error=abs(snapshot['prediction'][mask,:4][outside]-snapshot['target'][mask,:4][outside])
            monitor.non_stair += [error.sum(),error.size]
        report[label]=monitor.report()
    write_json(run/'aux_connectivity.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run
    (run/'plots').mkdir(exist_ok=True)
    protocol=read_json(run/'protocol.json');data=workspace_path(protocol['settings']['data'])
    selection=read_json(run/'selection.json');winner=selection['winner']
    history={name:read_json(run/name/'history.json') for name in ['A','B']}
    initial=read_json(run/'initial_val.json');final=read_json(run/'best_val.json')
    test_before=read_json(run/'initial_test.json');test_after=read_json(run/'best_test.json')
    selected_epoch=0;score=initial['selection_score']
    for row in history[winner]:
        if row['validation']['selection_score']<score:selected_epoch=row['epoch'];score=row['validation']['selection_score']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for name,rows in history.items():
        x=[r['epoch'] for r in rows]
        for ax,key in zip(axes[0],['total','absolute','conditional']):
            ax.plot(x,[r['losses'][key] for r in rows],label=name);ax.set(title=key+' loss',xlabel='Epoch');ax.legend();ax.grid(alpha=.2)
        for ax,key in [(axes[1,0],'selection_score')]:
            ax.plot(x,[r['validation'][key] for r in rows],label=name);ax.axhline(initial[key],ls='--',c='gray')
            ax.set(title='Fixed validation selection score',xlabel='Epoch');ax.legend();ax.grid(alpha=.2)
        for ax,group,title in [(axes[1,1],'new','New-scene mean response error'),(axes[1,2],'replay','Old-building absolute MAE')]:
            key='mean_response_error' if group=='new' else 'cost_mae'
            ax.plot(x,[r['validation']['groups'][group][key] for r in rows],label=name)
            ax.axhline(initial['groups'][group][key],ls='--',c='gray');ax.set(title=title,xlabel='Epoch');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Training banks alternate by epoch. A/B training objectives differ.\nFixed validation metrics use the same definitions.',fontsize=12)
    savefig(fig,run/'plots/training_AB.png')
    fig,axes=plt.subplots(4,2,figsize=(13,13),layout='constrained')
    for i,param in enumerate(NAMES):
        for col,key in enumerate(['changed_delta_mae','stable_delta_mae']):
            ax=axes[i,col]
            for name,rows in history.items():
                ax.plot([r['epoch'] for r in rows],[r['validation']['groups']['new']['parameter_response'][param][key] for r in rows],label=name)
            ax.axhline(initial['groups']['new']['parameter_response'][param][key],ls='--',c='gray',label='Initial FP32')
            ax.set(title=param+' | '+key,xlabel='Epoch',ylabel='Cost 0-1');ax.grid(alpha=.2);ax.legend(fontsize=8)
    savefig(fig,run/'plots/four_parameter_errors.png')
    before=np.load(run/'initial_test.npz');after=np.load(run/'best_test.npz')
    metadata=read_json(data/'evaluation/blocks.json');scenes={s['id']:s for s in metadata['scenes']}
    for name in np.unique(after['scene']):
        row=scenes[str(name)]
        if row['source']!='new':continue
        mask=(after['scene']==name)&(after['bank']==0);old=(before['scene']==name)&(before['bank']==0)
        xyz=after['coord'][mask];values=[after['target'][mask,3],before['prediction'][old,3],after['prediction'][mask,3]]
        fig=plt.figure(figsize=(15,5))
        for i,(title,v) in enumerate(zip(['Native GT','Before FP32','After FP32'],values),1):
            ax=fig.add_subplot(1,3,i,projection='3d');im=ax.scatter(*xyz.T,c=v,cmap=COST,vmin=0,vmax=1,s=1,lw=0)
            ax.set_box_aspect(np.ptp(xyz,axis=0));ax.view_init(28,-55);ax.set(title=title,xlabel='Map X',ylabel='Map Y',zlabel='Z')
        fig.suptitle(f'{name} | independent {row["kind"]} test | condition [0.5, 0.6, 0.4, 0.05]')
        fig.colorbar(im,ax=fig.axes,shrink=.6,pad=.02);savefig(fig,run/'plots'/f'{name}_GT_before_after.png')
    # One independent scene per facility kind, all four parameter scans.
    for kind in protocol['settings']['kinds']:
        row=next(s for s in scenes.values() if s['source']=='new' and s['split']=='test' and s['kind']==kind)
        name=row['id'];mask=(after['scene']==name)&(after['bank']==0);old=(before['scene']==name)&(before['bank']==0)
        membership=np.load(row['arrays']['facility_mask'])[after['query_index'][mask]]
        conditions=np.load(row['arrays']['conditions'])[row['profile_banks'][0]][:,[0,1,2,4]]
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for k,(param,ax) in enumerate(zip(NAMES,axes.ravel())):
            cols=list(range(4+9*k,13+9*k));x=conditions[cols,k]
            for label,values,color in [('GT',after['target'][mask],'#059669'),('Before',before['prediction'][old],'#64748b'),('After',after['prediction'][mask],'#a21caf')]:
                ax.plot(x,values[membership][:,cols].mean(0),'-o',label=label,c=color)
            ax.set(title=param,xlabel=param,ylabel='Mean facility cost');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{name} | {kind} | same facility queries | independent single-parameter scans')
        savefig(fig,run/'plots'/f'response_{kind}.png')
    r4=r4_report(run)
    auxiliary=auxiliary_connectivity(run,data,before,after)
    fixed=fixed_regression(run)
    old_mae_before=initial['groups']['replay']['cost_mae'];old_mae_after=final['groups']['replay']['cost_mae']
    response_before=initial['groups']['new']['mean_response_error'];response_after=final['groups']['new']['mean_response_error']
    per_param={name:final['groups']['new']['parameter_response'][name]['changed_delta_mae']<initial['groups']['new']['parameter_response'][name]['changed_delta_mae'] for name in NAMES}
    step=r4['after']['step_030_040'];slope=r4['after']['slope_018_070']
    acceptance=dict(response_error_reduction_pct=100*(1-response_after/response_before),
        response_improves_at_least15pct=response_after<=.85*response_before,
        each_parameter_changed_error_improved=per_param,old_MAE_increase_pct=100*(old_mae_after/old_mae_before-1),
        old_MAE_within5pct=old_mae_after<=1.05*old_mae_before,
        R4_step_amplitude_070_130=.7<=step['amplitude_ratio']<=1.3,
        R4_step_direction_at_least90=step['direction_correct_pct']>=90,
        R4_stable_slope_drift_at_most002=slope['stable_mean_abs_drift']<=.02)
    complete_predictions={kind:read_json(run/'predictions'/kind/'prediction.json') for kind in protocol['settings']['kinds']+['large_terrain']}
    summary=dict(complete=True,variant=winner,best_epoch=selected_epoch,last_epoch=history[winner][-1]['epoch'],
        improved=selected_epoch>0,selection=selection,initial_validation=initial,best_validation=final,
        dataset_audit=read_json(run/'dataset_audit.json'),fairness_check=read_json(run/'fairness_check.json'),
        independent_test_before=test_before,independent_test_after=test_after,acceptance=acceptance,
        R4=r4,auxiliary_connectivity=auxiliary,known_stair=fixed,
        all_stairs=read_json(run/'all_stairs_audit/summary.json'),complete_predictions=complete_predictions,
        baseline_checkpoint_preserved=hashlib.sha256(Path(protocol['initial_checkpoint']).read_bytes()).hexdigest()==protocol['initial_sha256'])
    write_json(run/'summary.json',summary)
    lines=['# 四参数解耦增强训练结果','',f'获胜版本：{winner}。第5轮固定验证选优后继续至30轮。最终最佳为新第{selected_epoch}轮，0表示初始化仍最佳。',
        '',f'训练前后均为相同FP32解码。源权重：{protocol["initial_checkpoint"]}，源权重未覆盖。',
        '', '## 验证及独立测试','', '| 指标 | Before | After |','|---|---:|---:|',
        f'| 固定验证综合分数 | {initial["selection_score"]:.6f} | {final["selection_score"]:.6f} |',
        f'| 新验证四参数平均响应误差 | {response_before:.6f} | {response_after:.6f} |',
        f'| 旧验证Cost MAE /50 | {old_mae_before*50:.4f} | {old_mae_after*50:.4f} |',
        f'| 独立新测试Cost MAE /50 | {test_before["groups"]["new"]["cost_mae_raw"]:.4f} | {test_after["groups"]["new"]["cost_mae_raw"]:.4f} |',
        '', '| 参数 | 新验证变化区MAE前→后 | 稳定区MAE前→后 |','|---|---|---|']
    for name in NAMES:
        a=initial['groups']['new']['parameter_response'][name];b=final['groups']['new']['parameter_response'][name]
        lines.append(f'| {name} | {a["changed_delta_mae"]:.5f} → {b["changed_delta_mae"]:.5f} | {a["stable_delta_mae"]:.5f} → {b["stable_delta_mae"]:.5f} |')
    lines+=['','## R4 固定回归','',
        f'阶高0.30→0.40：幅度比 {r4["before"]["step_030_040"]["amplitude_ratio"]:.4f} → {step["amplitude_ratio"]:.4f}，After方向一致率{step["direction_correct_pct"]:.2f}%。',
        f'坡度0.18→0.70，教师不变踏面上的平均绝对漂移：{r4["before"]["slope_018_070"]["stable_mean_abs_drift"]:.6f} → {slope["stable_mean_abs_drift"]:.6f}。',
        '', '## 预先规定的验收目标','', '```json',__import__('json').dumps(acceptance,ensure_ascii=False,indent=2),'```','',
        '训练条件库按奇偶轮交替，训练loss的交替波动包含条件分布变化。A/B训练损失定义不同，固定验证分数才使用共同口径。',
        '全部指标按实际结果报告，未为达到目标修改教师、划分、阈值或损失。测试属于本次设施域随机化分布，不等同于任意真实建筑泛化。',
        '', '## 文件','',
        '- best.pt、last.pt：最终统一权重入口。A/、B/：短跑与续训记录。',
        '- plots/training_AB.png、four_parameter_errors.png：训练与四参数变化/稳定误差。',
        '- plots/r4_parameter_response.png、r4_GT_before_after.png：R4条件回归与展开图。',
        '- plots/scene_*：全部24个独立新seed的GT/Before/After图。',
        '- all_stairs_audit/REPORT.md：原大图8处楼梯。',
        '- predictions/straight、switchback、spiral、ramp、large_terrain：完整输入PCD各原始点的预测。',
        '- aux_connectivity.json：0.45m/0.3m邻接、cost阈值0.3的辅助诊断，不参与选优。','']
    r4_pass=all(acceptance[k] for k in ['R4_step_amplitude_070_130','R4_step_direction_at_least90','R4_stable_slope_drift_at_most002'])
    lines[2:2]=[f'**独立测试Cost MAE：{test_before["groups"]["new"]["cost_mae_raw"]:.3f}/50 → {test_after["groups"]["new"]["cost_mae_raw"]:.3f}/50。R4预设参数响应目标'+('已达标。**' if r4_pass else '仍未达标。**'),'']
    if selected_epoch==0:lines.insert(2,'**初始化仍为最佳，本轮没有改善固定验证综合分数。**')
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT_COMPLETE',run,flush=True)


if __name__=='__main__':main()
