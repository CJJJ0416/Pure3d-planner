"""Plots for the fixed-feature R4 FiLM fit. Geometry Python only, no torch."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

from utils.pointcloud import ROOT,read_json,write_json
from common.pointcloud_io import write_pcd


def mean(value):return float(value.mean()) if len(value) else None


def point_metrics(gt,pred,dg,dp):
    error=pred-gt;absolute=abs(error);active=abs(dg)>.02;stable=~active
    low=gt<=.5;high=~low
    return dict(points=len(gt),MAE=float(absolute.mean()),MAE_raw50=float(absolute.mean()*50),bias=float(error.mean()),
        SmoothL1=float(np.where(absolute<.1,5*absolute**2,absolute-.05).mean()),
        false_block_count=int((low&(pred>.5)).sum()),GT_low_count=int(low.sum()),
        false_block_pct=100*int((low&(pred>.5)).sum())/int(low.sum()) if low.any() else None,
        false_pass_count=int((high&(pred<=.5)).sum()),GT_high_count=int(high.sum()),
        false_pass_pct=100*int((high&(pred<=.5)).sum())/int(high.sum()) if high.any() else None,
        GT_changed_points=int(active.sum()),delta_MAE=float(abs(dp-dg).mean()),
        changed_delta_MAE=mean(abs(dp[active]-dg[active])),GT_abs_delta=mean(abs(dg[active])),
        student_abs_delta=mean(abs(dp[active])),
        changed_direction_pct=mean((dp[active]*dg[active]>0).astype(float)*100),
        stable_prediction_drift=mean(abs(dp[stable])),
        amplitude_ratio=float(abs(dp[active]).mean()/abs(dg[active]).mean()) if active.any() else None)


def csv_rows(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve();config=read_json(run/'config.json')
    os.environ.setdefault('MPLCONFIGDIR',str(run/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','savefig.facecolor':'white'})
    history=read_json(run/'evaluations.json');complete=read_json(run/'training_complete.json')
    arrays={name:np.load(run/f'{file}_predictions.npz') for name,file in [('before','initial'),('best','best'),('last','last')]}
    initial=arrays['before'];gt=initial['target'];ratios=initial['ratios'];baseline=int(np.flatnonzero(ratios==.05)[0])
    xyz=initial['coord'];parts=initial['part'].astype(int)
    display=initial['display'];across=initial['across']
    layout=read_json(ROOT/'runs/planner/r4_cost_connectivity_audit_20261111/layout.json')
    (run/'plots').mkdir(exist_ok=True)
    metrics={};part_metrics={};flat=[];per_parts=[]
    for model,a in arrays.items():
        pred=a['prediction'];metrics[model]=[];part_metrics[model]=[]
        for k,ratio in enumerate(ratios):
            dg=gt[:,k]-gt[:,baseline];dp=pred[:,k]-pred[:,baseline]
            row=dict(stand_ratio=float(ratio),split='train' if ratio in config['train_ratios'] else 'heldout',
                **point_metrics(gt[:,k],pred[:,k],dg,dp))
            metrics[model].append(row);flat.append(dict(model=model,**row))
            for part in range(11):
                take=parts==part
                row=dict(stand_ratio=float(ratio),part=part,label=layout['labels'][part],
                    **point_metrics(gt[take,k],pred[take,k],dg[take],dp[take]))
                part_metrics[model].append(row);per_parts.append(dict(model=model,**row))
    csv_rows(run/'condition_metrics.csv',flat);csv_rows(run/'part_metrics.csv',per_parts)
    # Loss curves use evaluation-mode, post-update values at identical intervals.
    fig,axes=plt.subplots(2,2,figsize=(12.8,9),layout='constrained')
    for ax,key,title in zip(axes.ravel()[:3],['total','absolute','stand'],
            ['Total = absolute + stand','Absolute SmoothL1','Stand-ratio difference SmoothL1']):
        for split,color in [('train','#2563eb'),('validation','#d97706')]:
            ax.plot([r['step'] for r in history],[r[split][key] for r in history],'-o',ms=3,c=color,label=split)
        ax.axvline(complete['best_step'],c='#64748b',ls='--',lw=.9)
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Loss');ax.grid(alpha=.18);ax.legend()
    ax=axes[1,1]
    for ratio in config['validation_ratios']:
        column=int(np.flatnonzero(ratios==ratio)[0])
        ax.plot([r['step'] for r in history],[r['per_condition'][column]['MAE'] for r in history],
            '-o',ms=3,label=f'held-out stand={ratio:g}')
    ax.axvline(complete['best_step'],c='#64748b',ls='--',lw=.9)
    ax.set(title='Held-out condition MAE, same R4 geometry',xlabel='Optimizer updates',ylabel='Cost MAE')
    ax.grid(alpha=.18);ax.legend()
    fig.suptitle(f'R4 controlled fit | only 33,600 FiLM parameters trainable | best step {complete["best_step"]}\n'
        'Same frozen features, CostHead and teacher | 1,440 points x 6 training conditions',fontsize=13)
    fig.savefig(run/'loss_curves.png',dpi=180,bbox_inches='tight');plt.close(fig)

    def frame(ax):
        for row in range(11):ax.axhspan(row-.34,row+.34,color='#f3f6fa',zorder=0)
        ax.set(xlim=(-.06,layout['width_m']+.06),ylim=(-.52,10.52),
            xticks=[0,layout['width_m']/2,layout['width_m']],xticklabels=['Left','Centre','Right'],
            yticks=np.arange(11),yticklabels=layout['labels'],xlabel='Across width, ascending view')
        ax.spines[['top','right']].set_visible(False)
    for k,ratio in enumerate(ratios):
        fig,axes=plt.subplots(1,4,figsize=(15.5,10.5),sharey=True,layout='constrained')
        for j,(title,values) in enumerate([('GT',gt[:,k]),('Before',arrays['before']['prediction'][:,k]),
                (f'Best | step {complete["best_step"]}',arrays['best']['prediction'][:,k]),
                ('Last | step 1000',arrays['last']['prediction'][:,k])]):
            ax=axes[j];frame(ax)
            ax.scatter(across,display,c=values,cmap='RdYlGn_r',vmin=0,vmax=1,s=17,lw=.1,edgecolors='white')
            if j:
                bad=(gt[:,k]<=.5)&(values>.5)
                ax.scatter(across[bad],display[bad],marker='x',s=27,c='#b300b3',lw=1)
                title+=f'\nMAE={metrics[["before","best","last"][j-1]][k]["MAE"]:.4f}'
            ax.set_title(title)
        axes[0].set_ylabel('R4 | entry, 8 treads, turn, exit')
        split='TRAIN' if ratio in config['train_ratios'] else 'HELD-OUT CONDITION'
        fig.suptitle(f'FiLM-only fit | stand_ratio={ratio:.2f} | {split}\n'
            'Fixed clearance=0.5, slope=0.6, step=0.4 | identical XYZ | purple X: false block at cost limit 0.5',fontsize=13)
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='RdYlGn_r'),ax=axes,
            location='bottom',aspect=60,shrink=.8,pad=.025,label='trav_cost 0-1')
        fig.savefig(run/'plots'/f'stand_{ratio:.2f}.png',dpi=175,bbox_inches='tight');plt.close(fig)
        for model,a in arrays.items():
            folder=run/'predictions'/model;folder.mkdir(parents=True,exist_ok=True)
            write_pcd(folder/f'stand_{ratio:.2f}.pcd',xyz,a['prediction'][:,k])
    fig,axes=plt.subplots(2,2,figsize=(12.5,9),layout='constrained')
    colors={'before':'#dc2626','best':'#7c3aed','last':'#2563eb'}
    ax=axes[0,0];ax.plot(ratios,gt.mean(0),'-o',c='#059669',label='GT')
    for model,a in arrays.items():ax.plot(ratios,a['prediction'].mean(0),'-o',c=colors[model],label=model)
    ax.set(title='Mean cost, all R4 queries',ylabel='Mean cost');ax.legend()
    ax=axes[0,1]
    valid=[r for r in metrics['before'] if r['GT_changed_points']]
    x=[r['stand_ratio'] for r in valid]
    ax.plot(x,[r['GT_abs_delta'] for r in valid],'-o',c='#059669',label='GT')
    for model in arrays:
        rows=[r for r in metrics[model] if r['GT_changed_points']]
        ax.plot(x,[r['student_abs_delta'] for r in rows],'-o',c=colors[model],label=model)
    ax.set(title='Response vs 0.05, on GT-changing queries',ylabel='Mean |cost delta|');ax.legend()
    ax=axes[1,0]
    for model in arrays:ax.plot(ratios,[r['MAE'] for r in metrics[model]],'-o',c=colors[model],label=model)
    ax.set(title='Per-condition absolute error',ylabel='MAE');ax.legend()
    ax=axes[1,1]
    for model in arrays:ax.plot(ratios,[r['false_block_pct'] for r in metrics[model]],'-o',c=colors[model],label=model)
    ax.set(title='False block at cost limit 0.5',ylabel='Fraction of GT-low points (%)');ax.legend()
    for ax in axes.ravel():ax.set_xlabel('stand_ratio');ax.grid(alpha=.18)
    fig.suptitle('FiLM-only fitting before / best / last | same geometry, not unseen-building validation',fontsize=13)
    fig.savefig(run/'response_before_after.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12.3,8.5),sharex=True,sharey=True,layout='constrained')
    for i,ax in enumerate(axes.ravel()):
        ax.plot(ratios,initial['watch_target'][i],'-o',c='#059669',label='GT')
        for model,a in arrays.items():ax.plot(ratios,a['watch_prediction'][i],'-o',c=colors[model],label=model)
        ax.axhline(.5,c='#64748b',ls='--',lw=1)
        ax.set(title=f'Raw source point {i+1} | excluded from training',xlabel='stand_ratio',ylabel='Cost 0-1')
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Bottleneck vertex 137176 | four original ground returns | threshold 0.5',fontsize=13)
    fig.savefig(run/'bottleneck_before_after.png',dpi=180,bbox_inches='tight');plt.close(fig)
    with (run/'bottleneck_points.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['point','stand_ratio','GT','before','best','last'])
        for i in range(4):
            for k,r in enumerate(ratios):writer.writerow([i+1,float(r),float(initial['watch_target'][i,k]),
                *[float(arrays[model]['watch_prediction'][i,k]) for model in arrays]])
    initial_e=complete['initial'];last_e=complete['last'];best_e=complete['best']
    improvement={}
    for split in ['train','validation']:
        improvement[split]={}
        for loss in ['total','absolute','stand']:
            a=initial_e[split][loss];b=best_e[split][loss];c=last_e[split][loss]
            improvement[split][loss]=dict(before=a,best=b,last=c,best_relative_reduction_pct=100*(a-b)/a,
                last_relative_reduction_pct=100*(a-c)/a)
    entry_first={}
    for model in arrays:
        entry_first[model]=[r for r in part_metrics[model] if r['stand_ratio']==.05 and r['part'] in [0,1]]
    summary=dict(complete=True,best_step=complete['best_step'],steps=complete['steps'],
        loss_change=improvement,condition_metrics=metrics,part_metrics=part_metrics,
        entry_and_first_tread_at005=entry_first,
        bottleneck={model:a['watch_prediction'][:,baseline].tolist() for model,a in arrays.items()},
        bottleneck_GT_at005=initial['watch_target'][:,baseline].tolist(),
        source_checkpoint=str(ROOT/'runs/cost_all_stairs_dr_v3/best.pt'),
        best_checkpoint=str(run/'best.pt'),last_checkpoint=str(run/'last.pt'),
        training_wall_s=complete['wall_s'],peak_allocated_gib=complete['cuda_peak_allocated_gib'],
        scope='R4 training diagnostic. Validation holds out three stand values on the same geometry.',
        shared_FiLM_other_condition_contexts_not_tested=True)
    summary['minimum_validation_stand_step']=min(history,key=lambda row:row['validation']['stand'])['step']
    summary['conclusions']=dict(
        train_total_decreased=last_e['train']['total']<initial_e['train']['total'],
        train_stand_loss_decreased=last_e['train']['stand']<initial_e['train']['stand'],
        heldout_total_best_decreased=best_e['validation']['total']<initial_e['validation']['total'],
        heldout_stand_last_relative_reduction_pct=improvement['validation']['stand']['last_relative_reduction_pct'],
        watch_GT_low_points=int((initial['watch_target'][:,baseline]<=.5).sum()),
        watch_false_blocks_before=int(((initial['watch_target'][:,baseline]<=.5)&(initial['watch_prediction'][:,baseline]>.5)).sum()),
        watch_false_blocks_best=int(((initial['watch_target'][:,baseline]<=.5)&(arrays['best']['watch_prediction'][:,baseline]>.5)).sum()),
        watch_false_blocks_last=int(((initial['watch_target'][:,baseline]<=.5)&(arrays['last']['watch_prediction'][:,baseline]>.5)).sum()))
    write_json(run/'summary.json',summary)
    lines=['# R4 仅训练FiLM的stand_ratio受控拟合','',
        f'固定完成1000次AdamW更新，学习率5e-5，weight decay=0.05，余弦衰减。只训练condition_head的33,600个参数。最佳留出条件分数位于第{complete["best_step"]}步，初始化也参与选优。','',
        f'**训练总loss下降{improvement["train"]["total"]["last_relative_reduction_pct"]:.2f}%，stand差值loss下降{improvement["train"]["stand"]["last_relative_reduction_pct"]:.2f}%，确认仅更新FiLM也能降低训练误差。** 留出条件的stand差值loss先降后升，在第1000步相对初始化只下降{improvement["validation"]["stand"]["last_relative_reduction_pct"]:.2f}%，没有随训练项持续改善。','',
        f'留出stand差值项的最低点在第{summary["minimum_validation_stand_step"]}步，综合验证分数则选择第{complete["best_step"]}步。best对4个观察点中的3个GT低代价点，将误阻断从{summary["conclusions"]["watch_false_blocks_before"]}个降至{summary["conclusions"]["watch_false_blocks_best"]}个，仍有入口误高。last的误阻断回到{summary["conclusions"]["watch_false_blocks_last"]}个。','',
        '训练条件：[0.02,0.05,0.10,0.20,0.40,0.50]。留出条件：[0.08,0.12,0.30]。每步全部1440个Tomogram查询点，4个原始瓶颈点未参与训练。其余三个输入固定[0.5,0.6,0.4]。','',
        '## 损失是否降低','',
        '| 集合 | 损失 | 初始化 | Best | Last | Last相对下降 |',
        '|---|---|---:|---:|---:|---:|']
    for split in ['train','validation']:
        for loss,row in improvement[split].items():
            lines.append(f'| {split} | {loss} | {row["before"]:.6f} | {row["best"]:.6f} | {row["last"]:.6f} | {row["last_relative_reduction_pct"]:.2f}% |')
    lines+=['','总损失是绝对SmoothL1加stand差值SmoothL1，beta=0.1，lambda=1。训练差值使用6组条件的全部15对，教师变化区与稳定区等权，变化阈值0.02。验证绝对项只看三个留出条件，差值只比较它们各自相对0.05的变化。验证不参与反向传播。','',
        '## 各条件MAE与误阻断','',
        '| stand | 划分 | Before MAE | Best MAE | Last MAE | Before误阻断 | Best误阻断 | Last误阻断 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for k,ratio in enumerate(ratios):
        rows=[metrics[m][k] for m in arrays]
        lines.append(f'| {ratio:.2f} | {rows[0]["split"]} | '+ ' | '.join(f'{r["MAE"]:.5f}' for r in rows)+
            ' | '+' | '.join(f'{r["false_block_pct"]:.2f}%' for r in rows)+' |')
    lines+=['','误阻断分母是GT≤0.5的点。条件响应幅度、方向和差值MAE完整保存在condition_metrics.csv及summary.json。','',
        '## 当前0.05下的入口与第一踏步','',
        '| 部位 | Before MAE | Best MAE | Last MAE |',
        '|---|---:|---:|---:|']
    for i,label in enumerate(['Entry','Tread 1']):
        lines.append(f'| {label} | '+' | '.join(f'{entry_first[m][i]["MAE"]:.5f}' for m in arrays)+' |')
    lines+=['','## 4个未训练的瓶颈原始点，stand=0.05','',
        '| 点 | GT | Before | Best | Last |',
        '|---:|---:|---:|---:|---:|']
    for i in range(4):lines.append(f'| {i+1} | {initial["watch_target"][i,baseline]:.6f} | '+
        ' | '.join(f'{arrays[m]["watch_prediction"][i,baseline]:.6f}' for m in arrays)+' |')
    lines+=['','## 核验与解释范围','',
        '只有共享FiLM发生更新。编码器、投影、局部/全局查询、融合、CostHead及所有非FiLM缓冲量已对best和last逐项比较，原始best.pt哈希保持不变。CostHead权重冻结，但损失经过它反向传播到FiLM。没有新架构、阈值补偿、标签修改或规划。',
        '这是固定R4几何的可学习性实验。留出条件并非独立建筑，本次结果不能证明其他地图泛化改善。共享FiLM更新也可能影响其他三个参数的不同取值，本次没有检验这些组合。','',
        '检查训练总loss、绝对项、stand差值项是否分别下降，不能只用总loss代替调制能力。再看留出条件是否同步改善及局部误阻断是否降低。','',
        '## 输出','',
        '- loss_curves.png：初始化至1000步的三项损失及留出条件MAE。',
        '- response_before_after.png：九组条件的GT与训练前后响应。',
        '- plots/stand_*.png：GT/Before/Best/Last同XYZ展开图。',
        '- bottleneck_before_after.png、bottleneck_points.csv：4个未训练原始点。',
        '- best.pt、last.pt：完整模型结构，保留通用推理入口所需元数据。优化器只包含FiLM，续训应使用本实验的训练逻辑，不能当作原全模型训练器的优化器状态。',
        '- history.json、evaluations.json、condition_metrics.csv、part_metrics.csv：完整日志与指标。',
        '- predictions/：各条件训练前、best、last的R4代价PCD，保留原地图XYZ。',
        '- protocol.json、verification.json：固定配置、来源哈希及冻结权重核验。','']
    (run/'REPORT.md').write_text('\n'.join(lines))
    progress=read_json(run/'progress.json');progress['stage']='complete';progress['report']='REPORT.md'
    write_json(run/'progress.json',progress)
    print('REPORT_COMPLETE',run,flush=True)


if __name__=='__main__':main()
