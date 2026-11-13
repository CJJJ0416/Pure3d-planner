"""汇总宽深高楼梯微调、独立新 seed 测试与大地图固定楼梯回归。"""
import argparse
import heapq
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.pointcloud import ROOT,read_json,write_json,workspace_path

COST_MAP=LinearSegmentedColormap.from_list('native_cost',['#14c83c','#ffe633','#f02d23'])


def fixed_regression(run):
    """对原固定端点做 minimax 查询，数值为连通所需的最低 cost 阈值。"""
    example=np.load(ROOT/'datasets/wfc_regression_20261127/stair_case.npz')
    points=example['coord'];start,goal=int(example['start']),int(example['goal'])
    adjacency=cKDTree(points).query_ball_point(points,.45)
    def minimum(cost):
        best=np.full(len(cost),np.inf);best[start]=cost[start];queue=[(float(cost[start]),start)]
        while queue:
            score,node=heapq.heappop(queue)
            if score!=best[node]:continue
            if node==goal:return float(score)
            for nxt in adjacency[node]:
                candidate=max(score,float(cost[nxt]))
                if candidate<best[nxt]:best[nxt]=candidate;heapq.heappush(queue,(candidate,nxt))
    cost={'GT':example['GT']};full={};matches={}
    for name in ['initial','best']:
        z=np.load(run/(name+'_regression.npz'));column=z['profiles'].tolist().index('clearance05_slope06_step04_stand005')
        distance,index=cKDTree(z['coord']).query(points);cost[name]=z['prediction'][index,column]
        matches[name]=float(distance.max());full[name]=dict(xyz=z['coord'],gt=z['target'][:,column],prediction=z['prediction'][:,column])
    threshold={name:minimum(v) for name,v in cost.items()}
    scene=read_json(ROOT/'datasets/wfc_unseen_20261127/scene_20261127/scene.json')
    stair=next(r for r in scene['stairs'] if r['id']=='F1_F2_straight_1')
    a=np.deg2rad(stair['rotation_world_deg']);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    xyz=full['initial']['xyz'];local=(xyz[:,:2]-stair['center_world_xyz'][:2])@rot;base=stair['center_world_xyz'][2]
    mask=(abs(local[:,0])<stair['width_m']/2+.1)&(abs(local[:,1])<stair['unrotated_footprint_m'][1]/2-.45)&(xyz[:,2]>base+.03)&(xyz[:,2]<base+stair['rise_m']-.1)
    gt=full['initial']['gt'][mask];low=gt<=.3
    region={name:dict(cost_mae_raw=float(abs(x['prediction'][mask]-gt).mean()*50),false_block_pct=float(100*np.mean(x['prediction'][mask][low]>.3))) for name,x in full.items()}
    report=dict(stair='F1_F2_straight_1',threshold=.3,radius_m=.45,minimum_connecting_cost=threshold,
                connected={k:v<=.3 for k,v in threshold.items()},region=region,max_query_match_distance_m=matches)
    write_json(run/'stair_regression.json',report)
    fig,axes=plt.subplots(1,3,figsize=(11,6),sharex=True,sharey=True)
    for ax,(name,v) in zip(axes,[('GT',gt),('Before',full['initial']['prediction'][mask]),('After',full['best']['prediction'][mask])]):
        im=ax.scatter(local[mask,0],local[mask,1],c=v,s=8,cmap=COST_MAP,vmin=0,vmax=1,linewidths=0)
        ax.set(title=name,aspect='equal',xlabel='Across stair (m)')
    axes[0].set_ylabel('Along stair (m)');fig.suptitle('Large terrain / fixed right-lower stair | identical XYZ | cost 0-1')
    fig.subplots_adjust(top=.87,right=.88);fig.colorbar(im,cax=fig.add_axes([.91,.23,.02,.5]));fig.savefig(run/'plots/known_stair_GT_before_after.png',dpi=160);plt.close(fig)
    path=example['gt_path'];length=np.r_[0,np.linalg.norm(np.diff(points[path],axis=0),axis=1).cumsum()]
    fig,ax=plt.subplots(figsize=(12,4.5))
    for name,v in cost.items():ax.plot(length,v[path],'.-',label=f'{name}: bottleneck {threshold[name]:.3f}')
    ax.axhline(.3,c='red',ls='--',label='cost threshold 0.3');ax.set(xlabel='Distance along fixed GT route (m)',ylabel='Cost',ylim=(0,1.03))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(run/'plots/known_stair_longitudinal.png',dpi=150);plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);run=parser.parse_args().run
    (run/'plots').mkdir(exist_ok=True)
    protocol=read_json(run/'protocol.json');data=workspace_path(protocol['settings']['data'])
    history=read_json(run/'history.json');valid=[r for r in history if r['validation'] is not None]
    selected=min(valid,key=lambda r:tuple(r['validation']['selection_key']))
    initial=read_json(run/'initial_val.json');final=read_json(run/'final_val.json')
    before_test=read_json(run/'initial_test.json');after_test=read_json(run/'best_test.json')
    fig,axes=plt.subplots(2,3,figsize=(15,8));axes=axes.ravel()
    axes[0].plot([r['epoch'] for r in history],[r['loss'] for r in history]);axes[0].set_title('Training loss')
    measures=[('Cost MAE /50',lambda m:m['cost_mae_raw']),('Stair break (%)',lambda m:m['stair_metrics']['radii']['0.45']['break_pct']),
              ('False pass (%)',lambda m:m['stair_metrics']['radii']['0.45']['false_pass_pct']),
              ('Stand difference MAE',lambda m:m['parameter_response']['stand_ratio']['changed_delta_mae']),
              ('Non-stair MAE /50',lambda m:m['stair_metrics']['non_stair_mae']*50)]
    for ax,(title,metric) in zip(axes[1:],measures):
        ax.plot([r['epoch'] for r in valid],[metric(r['validation']) for r in valid],'o-');ax.axhline(metric(initial),ls='--',c='gray',label='Before')
        ax.set_title(title);ax.legend()
    for ax in axes:ax.set_xlabel('New fine-tuning epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(run/'plots/training.png',dpi=150);plt.close(fig)
    before=np.load(run/'initial_test.npz');after=np.load(run/'best_test.npz')
    manifest=read_json(data/'manifest.json');names={f"scene_{s['seed']}":s for s in manifest['scenes']}
    for name in np.unique(after['scene']):
        mask=after['scene']==name;old=before['scene']==name;xyz=after['coord'][mask]
        panels=[('Native VST GT',after['target'][mask,3]),('Before',before['prediction'][old,3]),('After',after['prediction'][mask,3])]
        fig=plt.figure(figsize=(16,5.5))
        for j,(title,v) in enumerate(panels,1):
            ax=fig.add_subplot(1,3,j,projection='3d');im=ax.scatter(*xyz.T,c=v,s=1,cmap=COST_MAP,vmin=0,vmax=1,linewidths=0)
            ax.set_box_aspect(np.ptp(xyz,axis=0));ax.view_init(28,-55);ax.set(title=title,xlabel='Map X (m)',ylabel='Map Y (m)',zlabel='Z (m)')
        spec=names[str(name)]['actual'];depth=spec.get('tread_depth_m',spec.get('tread_depth_at_mid_radius_m'))
        fig.suptitle(f'{name} / fresh {spec["kind"]} test | riser={spec["riser_m"]:.3f}, tread={depth:.3f}, width={spec["width_m"]:.3f} m\ncondition [0.5, 0.6, 0.4, 0.05] | same native query XYZ')
        fig.colorbar(im,ax=fig.axes,shrink=.6,pad=.02);fig.savefig(run/'plots'/f'{name}_gt_prediction.png',dpi=150);plt.close(fig)
    regression=fixed_regression(run)
    retention=read_json(run/'retention_val.json');retention_before=read_json(run/'initial_retention_val.json')
    summary=dict(source_checkpoint=protocol['initial_checkpoint'],source_epoch=protocol['source_epoch'],best_epoch=selected['epoch'],last_epoch=history[-1]['epoch'],
                 data=read_json(data/'data_ready.json'),initial_validation=initial,best_validation=final,
                 initial_test=before_test,best_test=after_test,regression=regression,retention_before=retention_before,retention_after=retention,
                 validation_eligible=not selected['validation']['selection_key'][0],baseline_checkpoint_preserved=True)
    write_json(run/'summary.json',summary)
    title=protocol['settings'].get('title','宽深高直梯数据增强微调')
    lines=['# '+title+'结果','',f"基于 `{protocol['initial_checkpoint']}`（来源第{protocol['source_epoch']}轮）。新训练完成{history[-1]['epoch']}轮，最佳为新第{selected['epoch']}轮。",'',
           '网络、原生教师规则、绝对/条件差值损失保持，更新几何/条件增强、采样比例和微调学习率。独立场景划分，旧大地图仅作回归。','',
           '| 新测试集指标 | Before | After |','|---|---:|---:|']
    for title,metric in [('Cost MAE /50',lambda m:m['cost_mae_raw']),('楼梯断裂率 %',lambda m:m['stair_metrics']['radii']['0.45']['break_pct']),('错误放行率 %',lambda m:m['stair_metrics']['radii']['0.45']['false_pass_pct']),('stand差值MAE',lambda m:m['parameter_response']['stand_ratio']['changed_delta_mae'])]:
        lines.append(f'| {title} | {metric(before_test):.4f} | {metric(after_test):.4f} |')
    lines.extend(['',f"固定失败楼梯最低连通代价：GT={regression['minimum_connecting_cost']['GT']:.4f}，Before={regression['minimum_connecting_cost']['initial']:.4f}，After={regression['minimum_connecting_cost']['best']:.4f}。",
        f"阈值0.3下：Before连通={regression['connected']['initial']}，After连通={regression['connected']['best']}。",'',
        '连通诊断使用固定原生点、固定端点和0.45m邻接，不是机器人动态执行结果。', '',f"最佳权重：`{run/'best.pt'}`。原保留权重仍位于runs/traversability/best.pt。",'',
        '完整原始点预测位于predictions/new_test和predictions/large_terrain。GT对比与训练曲线位于plots。'])
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT COMPLETE',run/'REPORT.md',flush=True)


if __name__=='__main__':main()
