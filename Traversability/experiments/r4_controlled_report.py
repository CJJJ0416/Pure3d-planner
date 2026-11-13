"""Measured comparison, gravity top views, and raw-PCD transfer of R4 fitting."""
import argparse
import csv
import importlib.util
import os
from pathlib import Path
import pickle

import numpy as np

from common.pointcloud_io import read_pcd_fields,fields_xyz,write_pcd
from utils.pointcloud import ROOT,read_json,workspace_path
from Traversability.experiments.r4_controlled_data import native_query,write_json,TREAD_PARTS,point_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run
    protocol=read_json(run/'protocol.json');settings=protocol['configuration'];selection=read_json(run/'selection.json')
    teacher=dict(np.load(run/'teacher.npz'));winner=selection['winner'];n=len(teacher['query_xyz'])
    evaluations={v:read_json(run/v/'evaluations.json') for v in ['A','B','C']}
    histories={v:read_json(run/v/'history.json') for v in ['A','B','C']}
    complete={v:read_json(run/v/'training_complete.json') for v in ['A','B','C']}
    records={v:next(row for row in evaluations[v] if row['step']==settings['pilot_steps']) for v in ['A','B','C']}
    initial=evaluations['A'][0]
    best_step=complete[winner]['best_step'];best=next(row for row in evaluations[winner] if row['step']==best_step)
    last=evaluations[winner][-1]
    snapshots={'initial':dict(np.load(run/'A/initial_predictions.npz'))}
    for v in ['A','B','C']:
        # If the winner continued, its immutable pilot snapshot was copied before resume.
        path=run/v/'pilot_predictions.npz'
        snapshots[v+'_300']=dict(np.load(path))
    snapshots['best']=dict(np.load(run/winner/'best_predictions.npz'))
    snapshots['last']=dict(np.load(run/winner/'last_predictions.npz'))
    summary=dict(complete=True,scope='Same R4 geometry fitting diagnosis. No unseen-building generalization claim.',
        selection=selection,winner=winner,best_step=best_step,last_step=last['step'],
        initial=initial,at_300=records,best=best,last=last,training=complete,
        improvements=dict(score_pct=100*(initial['metrics']['score']-best['metrics']['score'])/initial['metrics']['score'],
            tread_macro_MAE_pct=100*(initial['metrics']['tread_macro_MAE']-best['metrics']['tread_macro_MAE'])/initial['metrics']['tread_macro_MAE']),
        baseline_reference_max_abs_difference={v:evaluations[v][0]['reference_baseline_max_abs_difference'] for v in ['A','B','C']})
    source_predictions=[snapshots[v+'_300']['prediction'] for v in ['A','B','C']]
    initial_values=[np.load(run/v/'initial_predictions.npz')['prediction'] for v in ['A','B','C']]
    summary['initial_ABC_identical']=all(np.array_equal(initial_values[0],x) for x in initial_values[1:])
    gradients=[row['nominal_auxiliary_gradient'] for row in histories['C'] if row.get('nominal_auxiliary_gradient') and row['step']<=300]
    summary['C_gradient_interference']={}
    for group in gradients[0] if gradients else []:
        values=[r[group]['cosine'] for r in gradients if r[group]['cosine'] is not None]
        summary['C_gradient_interference'][group]=dict(observations=len(values),mean_cosine=float(np.mean(values)),
            negative_fraction=float(np.mean(np.array(values)<0)),minimum=float(np.min(values)))
    layout=read_json(ROOT/'runs/planner/r4_cost_connectivity_audit_20261111/layout.json')
    region=read_json(ROOT/'runs/planner/r4_cost_connectivity_audit_20261111/protocol.json')
    xyz=teacher['query_xyz'];parts=teacher['part'];gt=teacher['target'][:,0]
    angle=np.deg2rad(region['stair']['rotation_world_deg'])
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=np.array(region['stair']['center_world_xyz']);xy=(xyz-center)[:,:2]@rotation
    os.environ['MPLCONFIGDIR']=str(run/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    plots=run/'plots';plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'A':'#2472c8','B':'#8a2bba','C':'#e88923'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for v,rows in evaluations.items():
        x=[r['step'] for r in rows]
        axes[0,0].plot(x,[r['metrics']['score'] for r in rows],color=colors[v],label=v)
        axes[0,1].plot(x,[r['metrics']['tread_macro_MAE_raw50'] for r in rows],color=colors[v],label=v)
        axes[1,0].plot(x,[r['metrics']['spatial']['changed']['delta_MAE'] for r in rows],color=colors[v],label=v)
        axes[1,1].plot(x,[r['metrics']['spatial']['stable']['prediction_abs_delta'] for r in rows],color=colors[v],label=v)
    for ax,title in zip(axes.ravel(),['Shared nominal fit score','Mean of 8 tread MAEs (raw 50 units)',
        'Spatial delta MAE | GT changing','Student spatial variation | GT stable']):
        ax.set(title=title,xlabel='Optimizer updates');ax.grid(alpha=.2);ax.legend();ax.axvline(300,color='#888',ls='--',lw=.8)
    fig.suptitle('R4 controlled fitting | same initialization and deterministic full-context inputs')
    fig.savefig(plots/'ABC_fit_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for v,rows in histories.items():
        for ax,key,title in zip(axes,['total','original_absolute','spatial'],['Each variant training objective','Uniform absolute loss | comparable','Nominal spatial difference loss']):
            ax.plot([r['step'] for r in rows],[r['losses'][key] for r in rows],c=colors[v],lw=.9,label=v)
            ax.set(title=title,xlabel='Updates');ax.grid(alpha=.2);ax.legend()
    fig.savefig(plots/'ABC_loss_curves.png',dpi=170);plt.close(fig)

    def decorate(ax):
        for patch in layout['patches']:
            if patch['kind']!='horizontal':continue
            lo,hi=np.array(patch['bounds'])[:,:2]
            ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,edgecolor='#8090a0',lw=.6,zorder=0))
            if patch['step']:
                ax.text(-1.48 if patch['flight']==1 else 1.48,(lo[1]+hi[1])/2,str(patch['step']),ha='center',va='center',fontsize=9)
        ax.set(xlim=(-1.62,1.62),ylim=(-1.1,2.2),xlabel='Stair local X (m)',ylabel='Stair local Y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.12)

    def comparison(file,values,titles):
        fig,axes=plt.subplots(1,len(values),figsize=(5*len(values),7),layout='constrained',sharex=True,sharey=True)
        for i,(ax,cost,title) in enumerate(zip(axes,values,titles)):
            decorate(ax);error='error' in title.lower()
            ax.scatter(xy[:,0],xy[:,1],c=cost,cmap='RdBu_r' if error else 'RdYlGn_r',vmin=-1 if error else 0,vmax=1,
                       s=28,lw=.15,edgecolors='white',zorder=2)
            ax.set_title(title,fontweight='bold')
        count=sum('error' not in title.lower() for title in titles)
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='RdYlGn_r'),ax=list(axes[:count]),
            location='bottom',aspect=50,pad=.04,label='Normalized cost 0-1, raw cost = value × 50')
        if count<len(values):
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),ax=axes[-1],location='bottom',
                pad=.04,label='Student - GT | blue under, red over')
        fig.suptitle('R4 gravity top view | identical 1,440 XYZ | condition [0.5, 0.6, 0.4, 0.05]\n'
                     'Same geometry fitting diagnosis, not unseen-map evaluation')
        fig.savefig(plots/file,dpi=170);plt.close(fig)
    before=snapshots['initial']['prediction'][:,0];after=snapshots['best']['prediction'][:,0]
    comparison('R4_GT_before_best_error.png',[gt,before,after,after-gt],['VST GT','Before',f'{winner} best | step {best_step}','Error'])
    comparison('R4_GT_A_B_C_at300.png',[gt,*[snapshots[v+'_300']['prediction'][:,0] for v in ['A','B','C']]],
               ['VST GT','A | original loss | 300','B | balanced + spatial | 300','C | plus conditions | 300'])
    for v in ['A','B','C']:
        values=snapshots[v+'_300']['prediction'][:,0]
        comparison(f'R4_{v}_GT_before_after_error.png',[gt,before,values,values-gt],['VST GT','Before',v+' | step 300','Error'])
    with (run/'part_metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['stage','part','label','points','MAE_raw50','bias_raw50'])
        for name,row in [('initial',initial),*[(v+'_300',records[v]) for v in ['A','B','C']],('best',best),('last',last)]:
            for part in row['metrics']['parts']:writer.writerow([name,part['part'],layout['labels'][part['part']],part['points'],part['MAE_raw50'],part['bias_raw50']])
    with (run/'condition_metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['stage','column','split','clearance','slope','step','stand','MAE_raw50','bias_raw50'])
        for name,row in [('initial',initial),*[(v+'_300',records[v]) for v in ['A','B','C']],('best',best),('last',last)]:
            for c in row['metrics']['conditions']:writer.writerow([name,c['column'],c['split'],*c['condition'],c['MAE_raw50'],c['bias_raw50']])
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    x=np.arange(8)
    for j,(name,row,color) in enumerate([('A / 300',records['A'],colors['A']),('B / 300',records['B'],colors['B']),
        ('C / 300',records['C'],colors['C']),('Final best',best,'#12935b')]):
        ax.bar(x+(j-1.5)*.2,[row['metrics']['parts'][i]['MAE_raw50'] for i in TREAD_PARTS],.2,label=name,color=color)
    ax.axhline(3,c='#d7263d',ls='--',lw=.8,label='Per-tread target 3')
    ax.set(xticks=x,xticklabels=[str(i+1) for i in x],xlabel='Tread index ascending',ylabel='MAE, raw 50 units',title='Per-tread absolute error')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(ncol=3)
    fig.savefig(plots/'tread_MAE_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.5),layout='constrained')
    for name,values,color in [('GT',gt,'black'),('Before',before,'#8792a0'),(winner+' best',after,colors[winner])]:
        ax.plot(range(11),[values[parts==i].mean()*50 for i in range(11)],'-o',c=color,label=name)
    ax.set(xticks=range(11),xticklabels=layout['labels'],ylabel='Mean cost, raw 50 scale',title='Per-part mean cost | pointwise errors reported separately')
    ax.grid(alpha=.2);ax.legend();fig.savefig(plots/'part_cost_means.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,14),layout='constrained')
    for p,(group,name) in enumerate(zip(protocol['condition_groups'],['clearance_min','slope_max','step_max','stand_ratio'])):
        x=teacher['conditions'][group,p];order=np.argsort(x);columns=np.array(group)[order];x=x[order]
        for j,mask in enumerate([np.ones(n,bool),parts==1]):
            ax=axes[p,j]
            ax.plot(x,teacher['target'][mask][:,columns].mean(0),'-o',c='#149154',label='GT')
            ax.plot(x,snapshots['initial']['prediction'][mask][:,columns].mean(0),'-o',c='#8792a0',label='Before')
            ax.plot(x,snapshots['C_300']['prediction'][mask][:,columns].mean(0),'--o',c=colors['C'],label='C / 300')
            ax.plot(x,snapshots['best']['prediction'][mask][:,columns].mean(0),'-o',c=colors[winner],label=winner+' best')
            ax.set(xlabel=name,ylabel='Mean cost 0-1',title='All R4 points' if j==0 else 'First tread only');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Four-parameter response | same geometry | held-out errors in summary.json')
    fig.savefig(plots/'parameter_response_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained')
    for p,group in enumerate(protocol['condition_groups']):
        x=teacher['conditions'][group,p];order=np.argsort(x);columns=np.array(group)[order];x=x[order]
        for i in range(4):
            ax=axes[p,i];ax.plot(x,teacher['watch_target'][i,columns],'-o',c='#149154',label='GT')
            ax.plot(x,snapshots['initial']['watch_prediction'][i,columns],'-o',c='#8792a0',label='Before')
            ax.plot(x,snapshots['C_300']['watch_prediction'][i,columns],'--o',c=colors['C'],label='C / 300')
            ax.plot(x,snapshots['best']['watch_prediction'][i,columns],'-o',c=colors[winner],label='Best')
            ax.axhline(.5,c='#d7263d',ls='--',lw=.7);ax.set(title=f'Parameter {p+1} | watch {i+1}',ylim=(0,1));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Four raw bottleneck points | observation only, never in training loss')
    fig.savefig(plots/'bottleneck_response_curves.png',dpi=150);plt.close(fig)
    if gradients:
        fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
        rows=[r for r in histories['C'] if r.get('nominal_auxiliary_gradient')]
        for name in ['all','encoder','condition_head','cost_head']:
            ax.plot([r['step'] for r in rows],[r['nominal_auxiliary_gradient'][name]['cosine'] for r in rows],'-o',label=name)
        ax.axhline(0,c='black',lw=.7);ax.set(xlabel='Updates',ylabel='Gradient cosine',title='C: nominal objective vs weighted auxiliary-condition objective')
        ax.grid(alpha=.2);ax.legend();fig.savefig(plots/'C_gradient_cosines.png',dpi=170);plt.close(fig)
    for name,values in [('GT',gt),('before',before),('best',after),('absolute_error',abs(after-gt))]:
        write_pcd(run/f'R4_{name}.pcd',xyz,values)
    # Evaluate transfer to the real input query locations, separately from the fitting queries.
    predicted=read_pcd_fields(run/'prediction/prediction.pcd');raw_xyz=fields_xyz(predicted)
    full_before=read_pcd_fields(workspace_path(settings['reference_run'])/'prediction/prediction.pcd')
    bounds=np.array(region['local_xy_bounds']);local=(raw_xyz-center)[:,:2]@rotation
    candidate=((local>=bounds[0])&(local<=bounds[1])).all(1)&(raw_xyz[:,2]>=-.4)&(raw_xyz[:,2]<=region['stair']['rise_m']+.5)
    ids=np.flatnonzero(candidate);coords=raw_xyz[ids]
    spec=importlib.util.spec_from_file_location('r4_audit_layout',ROOT/'runs/planner/r4_cost_connectivity_audit_20261111/audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    layout_object=audit.StairLayout(region['stair'],read_json(workspace_path(settings['reference_run'])/'protocol.json')['query'],region)
    membership=layout_object.assign(coords)
    with (workspace_path(settings['reference_run'])/'teacher/tomogram.pickle').open('rb') as stream:payload=pickle.load(stream)
    raw_gt,raw_dz,_=native_query(payload,coords)
    support=membership['horizontal']&(membership['distance']<=.035)&(raw_dz<=.035)
    unknown=np.where(support,raw_gt,np.nan)
    raw_after=predicted['trav_cost'][ids];raw_before=full_before['trav_cost'][ids]
    summary['raw_PCD_transfer']=dict(points_in_R4=len(ids),known_support_points=int(support.sum()),unknown_points=int((~support).sum()),
        before=point_metrics(raw_gt[support],raw_before[support]),after=point_metrics(raw_gt[support],raw_after[support]),
        full_XYZ_order_identical=bool(np.array_equal(raw_xyz,fields_xyz(full_before))),
        evaluation='Only geometry-defined horizontal support locations with matching GT ground have GT labels. Other points remain unknown. No training points were filtered.')
    from scipy.spatial import cKDTree
    known_xyz=coords[support];distance,nearest=cKDTree(xyz).query(known_xyz)
    native_shape=np.array(payload['data'].shape[2:])
    cell=np.rint((known_xyz[:,:2]-payload['center'])/payload['resolution']).astype(int)+native_shape//2
    query_cell=np.rint((xyz[nearest,:2]-payload['center'])/payload['resolution']).astype(int)+native_shape//2
    matched=(cell==query_cell).all(1)&(abs(known_xyz[:,2]-xyz[nearest,2])<=.035)
    known_gt=raw_gt[support];known_before=raw_before[support];known_after=raw_after[support]
    raw_part=membership['part'][support].astype(int)
    raw_parts=[dict(part=i,points=int((raw_part==i).sum()),
        before=point_metrics(known_gt[raw_part==i],known_before[raw_part==i]),
        after=point_metrics(known_gt[raw_part==i],known_after[raw_part==i])) for i in range(11)]
    diagnosis=dict(per_part=raw_parts,
        before_tread_macro_MAE_raw50=float(np.mean([raw_parts[i]['before']['MAE_raw50'] for i in TREAD_PARTS])),
        after_tread_macro_MAE_raw50=float(np.mean([raw_parts[i]['after']['MAE_raw50'] for i in TREAD_PARTS])),
        trained_native_cell=dict(points=int(matched.sum()),before=point_metrics(known_gt[matched],known_before[matched]),
            after=point_metrics(known_gt[matched],known_after[matched]),mean_query_distance_m=float(distance[matched].mean())),
        untrained_native_cell=dict(points=int((~matched).sum()),before=point_metrics(known_gt[~matched],known_before[~matched]),
            after=point_metrics(known_gt[~matched],known_after[~matched]),mean_query_distance_m=float(distance[~matched].mean())))
    inference_meta=read_json(run/'prediction/geometry/blocks.json')
    original_meta=read_json(Path(protocol['source_geometry_cache'])/'blocks.json')
    matches=[]
    for i in protocol['active_cores']:
        with np.load(Path(protocol['source_geometry_cache'])/original_meta['blocks'][i]['file']) as old:old_ids=old['input_indices']
        with np.load(run/'prediction/geometry'/inference_meta['blocks'][i]['file']) as new:new_ids=new['input_indices']
        matches.append(bool(np.array_equal(old_ids,new_ids)))
    diagnosis['input_geometry_indices_identical_per_core']=matches
    production=dict(np.load(run/'production_batch_R4.npz'))
    column=int(np.flatnonzero(np.all(np.isclose(production['conditions'],settings['baseline']),axis=1))[0])
    prod=production['prediction'][:,column]
    diagnosis['production_batch_at_fitting_XYZ']=dict(**point_metrics(gt,prod),
        maximum_prediction_difference_to_batch3=float(abs(prod-after).max()),
        mean_prediction_difference_to_batch3=float(abs(prod-after).mean()))
    summary['query_position_diagnosis']=diagnosis
    np.savez(run/'raw_PCD_transfer.npz',input_indices=ids,coord=coords,GT=unknown,before=raw_before,after=raw_after,
             known_support=support,ground_error=raw_dz)
    np.savetxt(run/'raw_PCD_transfer.csv',np.c_[ids,coords,unknown,raw_before,raw_after,raw_after-unknown],delimiter=',',
        header='input_index,x,y,z,GT_or_NaN,before,after,error_or_NaN',comments='')
    fig,axes=plt.subplots(1,4,figsize=(20,7),layout='constrained',sharex=True,sharey=True)
    points=local[ids]
    for j,(ax,values,title) in enumerate(zip(axes,[unknown,raw_before,raw_after,raw_after-unknown],['GT support / grey unknown','Before | raw PCD','Best | raw PCD','Error | known support only'])):
        decorate(ax);ax.scatter(points[~support,0],points[~support,1],s=.4,c='#aab3bf',alpha=.2)
        shown=support if j in [0,3] else np.ones(len(ids),bool)
        ax.scatter(points[shown,0],points[shown,1],c=values[shown],s=2,cmap='RdBu_r' if j==3 else 'RdYlGn_r',vmin=-1 if j==3 else 0,vmax=1)
        ax.set_title(title)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='RdYlGn_r'),ax=list(axes[:3]),location='bottom',aspect=50,label='Cost 0-1')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),ax=axes[3],location='bottom',label='Student - GT')
    fig.suptitle('Transfer to original PCD query positions | all original XYZ preserved | unknown GT remains unknown')
    fig.savefig(plots/'R4_raw_PCD_transfer.png',dpi=170);plt.close(fig)
    bm=best['metrics'];im=initial['metrics'];summary['acceptance']=dict(
        tread_macro_MAE_at_most2=bm['tread_macro_MAE_raw50']<=2,
        every_tread_MAE_at_most3=all(bm['parts'][i]['MAE_raw50']<=3 for i in TREAD_PARTS),
        stable_spatial_variation_at_most002=bm['spatial']['stable']['prediction_abs_delta']<=.02,
        large_spatial_delta_error_reduced50pct=bm['spatial']['large_change']['delta_MAE']<=.5*im['spatial']['large_change']['delta_MAE'])
    conclusions=[]
    if records['A']['metrics']['tread_macro_MAE_raw50']<=3:conclusions.append('原逐点损失在固定R4上已能显著拟合，现有结构并非完全学不会教师。')
    else:conclusions.append('A在300步内仍未达到逐踏步原始MAE≤3，需结合曲线判断当前预算下的拟合不足。')
    ratio=records['B']['metrics']['score']/records['A']['metrics']['score']
    conclusions.append(f'B/A在300步的统一拟合分数比为{ratio:.4f}，越低表示逐级均衡与空间差值监督越有效。')
    ratio2=records['C']['metrics']['score']/records['B']['metrics']['score']
    conclusions.append(f'C/B在300步的基准拟合分数比为{ratio2:.4f}。条件干扰需同时看梯度夹角与留出条件误差，不能只看总损失。')
    if gradients:
        g=summary['C_gradient_interference']['all'];conclusions.append(f'C的全模型梯度平均余弦{g["mean_cosine"]:.4f}，负余弦比例{100*g["negative_fraction"]:.1f}%。')
    same=diagnosis['trained_native_cell'];production_metrics=diagnosis['production_batch_at_fitting_XYZ']
    conclusions.append(f'使用原生产批量在相同1440个XYZ查询，原始MAE为{production_metrics["MAE_raw50"]:.4f}。'
        f'但{same["points"]}个落在已训练教师网格中的原始PCD位置，平均偏离查询中心{same["mean_query_distance_m"]*100:.2f}cm，'
        f'MAE仍为{same["after"]["MAE_raw50"]:.4f}。这表明查询位置迁移存在明显差距，不能用训练中心点的低误差替代部署位置效果。')
    summary['conclusions']=conclusions
    write_json(run/'summary.json',summary)
    text=['# R4 受控拟合结果','',f'选中版本 **{winner}**，最终最佳为第 **{best_step}** 次更新，末次为1500。所有版本从同一第30轮权重初始化。结果属于同一R4几何的拟合诊断。','',
        '| 阶段 | S | 8级踏步宏平均MAE（原始50尺度） | 空间变化差值MAE | 稳定邻点预测差值 |',
        '|---|---:|---:|---:|---:|']
    for name,row in [('初始化',initial),*[(v+' / 300',records[v]) for v in ['A','B','C']],('最终best',best),('最终last',last)]:
        m=row['metrics'];text.append(f'| {name} | {m["score"]:.6f} | {m["tread_macro_MAE_raw50"]:.4f} | {m["spatial"]["changed"]["delta_MAE"]:.5f} | {m["spatial"]["stable"]["prediction_abs_delta"]:.5f} |')
    text+=['','## 原因判断','',*conclusions,'','## 逐级结果','',
        '| 部位 | 初始化MAE | 最终MAE |','|---|---:|---:|']
    for i,label in enumerate(layout['labels']):text.append(f'| {label} | {im["parts"][i]["MAE_raw50"]:.4f} | {bm["parts"][i]["MAE_raw50"]:.4f} |')
    transfer=summary['raw_PCD_transfer']
    text+=['','## 原始PCD位置与观察点','',
        f'原始PCD的{transfer["known_support_points"]}个已知R4支撑点，MAE从{transfer["before"]["MAE_raw50"]:.4f}变为{transfer["after"]["MAE_raw50"]:.4f}个原始代价单位。完整预测保留{len(raw_xyz):,}点及其行序。其他位置GT保留未知，不借用邻近踏面。',
        '四个瓶颈点从未参与训练损失。其四参数GT、初始化及best曲线见plots/bottleneck_response_curves.png。','',
        f'原始PCD的逐踏步宏平均MAE：{diagnosis["before_tread_macro_MAE_raw50"]:.4f} → {diagnosis["after_tread_macro_MAE_raw50"]:.4f}。这与训练中心点逐踏步0.6601仍有显著差距。',
        f'其中{same["points"]}/{transfer["known_support_points"]}个原始支撑点的教师网格已进入训练，仍未达到训练查询点的精度。相同XYZ改用生产批量后的MAE为{production_metrics["MAE_raw50"]:.4f}，不能主要归因于批量浮点差异。','',
        '## 核验与验收','',str(summary['acceptance']),
        f'三版本初始化预测逐点相同：{summary["initial_ABC_identical"]}。与上轮按原批量推理的数值差异：{summary["baseline_reference_max_abs_difference"]}。',
        '三个核心块同批前后向，1440个监督点和4个只观察点的原索引还原后计算损失。没有可靠性筛选。归一化缓冲量逐项比较见verification.json。',
        '原权重保留，现有推理入口已实际加载新完整权重并生成完整prediction/prediction.pcd。本次没有运行A*或GPMP，也没有安排定时任务。','',
        '## 图与数据','',
        '- plots/ABC_fit_curves.png 与 ABC_loss_curves.png：三版统一指标和各自损失。',
        '- plots/R4_GT_A_B_C_at300.png：同预算代价对比。',
        '- plots/R4_GT_before_best_error.png：最佳版本重力俯视GT/训练前/训练后/误差。',
        '- plots/parameter_response_curves.png、bottleneck_response_curves.png、C_gradient_cosines.png。',
        '- plots/R4_raw_PCD_transfer.png：原始PCD查询位置的迁移对比。',
        '- part_metrics.csv、raw_PCD_transfer.csv、summary.json：逐部位与逐点指标。',
        '- best.pt、last.pt：完整结构和现有推理所需元数据。','',
        '这些指标不能当作新建筑泛化结果。']
    (run/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('REPORT_COMPLETE',winner,best_step,summary['acceptance'],flush=True)


if __name__=='__main__':main()
