"""Separate same-map region transfer from independent-seed generalization."""
import argparse
import csv
import os
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree

from common.pointcloud_io import read_pcd_fields,fields_xyz
from utils.pointcloud import ROOT,read_json,workspace_path
from Traversability.experiments.r4_controlled_data import point_metrics,native_query,write_json
from Traversability.experiments.all_stair_audit import local_coordinates,region_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);out=p.parse_args().run
    protocol=read_json(out/'protocol.json');before=np.load(out/'initial_regression.npz');after=np.load(out/'best_regression.npz')
    column=after['profiles'].tolist().index('clearance05_slope06_step04_stand005')
    xyz=after['coord'];gt=after['target'][:,column];old=before['prediction'][:,column];new=after['prediction'][:,column]
    summary=dict(same_XYZ=bool(np.array_equal(before['coord'],xyz)),same_GT=bool(np.array_equal(before['target'],after['target'])),
        same_map_scope=protocol['same_map'],independent_scope=protocol['independent_scope'],condition=protocol['condition'],
        same_map_stairs=[],route_regions=[],independent_seeds=[])
    audit=read_json(out/'all_stairs_audit/summary.json')
    baseline_full=read_pcd_fields(ROOT/'runs/planner/r4_replan_decoupling_v4_20261111/prediction/prediction.pcd')
    fitted_full=read_pcd_fields(ROOT/'runs/traversability/r4_controlled_fit_v5/prediction/prediction.pcd')
    pcd_xyz=fields_xyz(fitted_full)
    with (ROOT/'runs/planner/r4_replan_decoupling_v4_20261111/teacher/tomogram.pickle').open('rb') as stream:payload=pickle.load(stream)
    for row in audit['stairs']:
        stair=row['geometry'];mask=region_mask(local_coordinates(xyz,stair),stair)
        record=dict(id=stair['id'],kind=stair['kind'],trained_R4=stair['id']=='F0_F1_switchback_2',
            query_before=point_metrics(gt[mask],old[mask]),query_after=point_metrics(gt[mask],new[mask]))
        raw_mask=region_mask(local_coordinates(pcd_xyz,stair),stair);ids=np.flatnonzero(raw_mask)
        targets,dz,_=native_query(payload,pcd_xyz[ids]);known=dz<=.035
        record.update(raw_support_points=int(known.sum()),raw_unknown_points=int((~known).sum()),
            raw_before=point_metrics(targets[known],baseline_full['trav_cost'][ids][known]),
            raw_after=point_metrics(targets[known],fitted_full['trav_cost'][ids][known]))
        summary['same_map_stairs'].append(record)
    query_protocol=read_json(ROOT/'runs/planner/large_terrain_six_vst_20261111/protocol.json')
    for q in query_protocol['queries']:
        file=ROOT/f'runs/planner/large_terrain_six_vst_20261111/vst15/route_{q["number"]:02d}/support_path.csv'
        path=np.genfromtxt(file,delimiter=',',names=True);path=np.column_stack([path[k] for k in ['x','y','z']])
        pieces=[]
        for a,b in zip(path[:-1],path[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/.05)))
            pieces.append(a+(b-a)*np.arange(count)[:,None]/count)
        dense=np.vstack([*pieces,path[-1:]])
        distance,_=cKDTree(dense).query(xyz);mask=distance<=.5
        summary['route_regions'].append(dict(R=q['number'],id=q['id'],region='Native VST reference support route + fixed 0.5m 3D tube, no cost filtering',
            points=int(mask.sum()),before=point_metrics(gt[mask],old[mask]),after=point_metrics(gt[mask],new[mask])))
    test_before=np.load(out/'initial_test.npz');test_after=np.load(out/'best_test.npz')
    summary['independent_same_XYZ']=bool(np.array_equal(test_before['coord'],test_after['coord']))
    summary['independent_same_GT']=bool(np.array_equal(test_before['target'],test_after['target']))
    metadata=read_json(out/'independent_data/blocks.json');scenes={s['id']:s for s in metadata['scenes']}
    plot_rows=[]
    for item in protocol['independent_scenes']:
        name=item['id'];mask=(test_after['scene']==name)&(test_after['bank']==0)
        xyz_test=test_after['coord'][mask];g=test_after['target'][mask,3]
        a=test_before['prediction'][mask,3];b=test_after['prediction'][mask,3]
        scene=scenes[name];membership=np.load(scene['arrays']['facility_mask'])[test_after['query_index'][mask]]
        record=dict(id=name,kind=item['kind'],query_before=point_metrics(g,a),query_after=point_metrics(g,b),
            facility_before=point_metrics(g[membership],a[membership]),facility_after=point_metrics(g[membership],b[membership]))
        fitted=read_pcd_fields(out/'predictions'/item['kind']/'prediction.pcd')
        baseline=read_pcd_fields(ROOT/'runs/traversability/cost_parameter_decoupling_v4/predictions'/item['kind']/'prediction.pcd')
        raw_xyz=fields_xyz(fitted)
        with (Path(item['pcd']).parent/'conditioned/vst_relaxed.pickle').open('rb') as stream:pkl=pickle.load(stream)
        raw_gt,dz,_=native_query(pkl,raw_xyz);known=dz<=.035
        record.update(raw_support_points=int(known.sum()),raw_unknown_points=int((~known).sum()),
            raw_before=point_metrics(raw_gt[known],baseline['trav_cost'][known]),raw_after=point_metrics(raw_gt[known],fitted['trav_cost'][known]),
            raw_XYZ_order_identical=bool(np.array_equal(raw_xyz,fields_xyz(baseline))))
        summary['independent_seeds'].append(record)
        plot_rows.append((item,xyz_test,g,a,b))
    others=[r for r in summary['same_map_stairs'] if not r['trained_R4']]
    summary['aggregates']=dict(
        other_stairs_query_before=float(np.mean([r['query_before']['MAE_raw50'] for r in others])),
        other_stairs_query_after=float(np.mean([r['query_after']['MAE_raw50'] for r in others])),
        other_stairs_raw_before=float(np.mean([r['raw_before']['MAE_raw50'] for r in others])),
        other_stairs_raw_after=float(np.mean([r['raw_after']['MAE_raw50'] for r in others])),
        other_stairs_query_improved=sum(r['query_after']['MAE']<r['query_before']['MAE'] for r in others),
        independent_query_before=float(np.mean([r['query_before']['MAE_raw50'] for r in summary['independent_seeds']])),
        independent_query_after=float(np.mean([r['query_after']['MAE_raw50'] for r in summary['independent_seeds']])),
        independent_raw_before=float(np.mean([r['raw_before']['MAE_raw50'] for r in summary['independent_seeds']])),
        independent_raw_after=float(np.mean([r['raw_after']['MAE_raw50'] for r in summary['independent_seeds']])),
        independent_query_improved=sum(r['query_after']['MAE']<r['query_before']['MAE'] for r in summary['independent_seeds']))
    write_json(out/'summary.json',summary)
    with (out/'metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['scope','id','trained_R4','query_MAE_before_raw50','query_MAE_after_raw50','raw_MAE_before_raw50','raw_MAE_after_raw50'])
        for kind,rows in [('same_map',summary['same_map_stairs']),('independent',summary['independent_seeds'])]:
            for r in rows:writer.writerow([kind,r['id'],r.get('trained_R4',False),r['query_before']['MAE_raw50'],r['query_after']['MAE_raw50'],r['raw_before']['MAE_raw50'],r['raw_after']['MAE_raw50']])
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for ax,rows,title,field in [(axes[0],summary['route_regions'],'R1-R6 fixed reference corridors','route'),
                              (axes[1],summary['same_map_stairs'],'Same building | all 8 stairs','stairs'),
                              (axes[2],summary['independent_seeds'],'Four held-out test seeds','independent')]:
        x=np.arange(len(rows));left=[r['before']['MAE_raw50'] if field=='route' else r['query_before']['MAE_raw50'] for r in rows]
        right=[r['after']['MAE_raw50'] if field=='route' else r['query_after']['MAE_raw50'] for r in rows]
        ax.bar(x-.18,left,.36,color='#728398',label='Original best.pt');ax.bar(x+.18,right,.36,color='#b13d9c',label='R4 fit best.pt')
        labels=[f'R{r["R"]}' if field=='route' else (f'S{i+1}'+('\nR4 fit' if r.get('trained_R4') else '') if field=='stairs' else r['kind']) for i,r in enumerate(rows)]
        ax.set_xticks(x,labels,fontsize=8);ax.set(title=title,ylabel='Cost MAE, raw 50 units');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.suptitle('No additional training | fixed condition [0.5, 0.6, 0.4, 0.05] | identical native query XYZ')
    fig.savefig(plots/'generalization_summary.png',dpi=170);plt.close(fig)
    for item,x,g,a,b in plot_rows:
        fig=plt.figure(figsize=(15,5),layout='constrained')
        for i,(title,values) in enumerate([('VST GT',g),('Original best.pt',a),('R4 fit best.pt',b)],1):
            ax=fig.add_subplot(1,3,i,projection='3d');image=ax.scatter(*x.T,c=values,s=1,cmap='RdYlGn_r',vmin=0,vmax=1,lw=0)
            ax.set_box_aspect(np.maximum(np.ptp(x,axis=0),.1));ax.view_init(27,-55);ax.set(title=title,xlabel='Map X',ylabel='Map Y',zlabel='Map Z')
        fig.colorbar(image,ax=fig.axes,shrink=.6,label='Cost 0-1');fig.suptitle(item['id']+' | held-out '+item['kind']+' | same XYZ and condition')
        fig.savefig(plots/(item['id']+'_GT_before_after.png'),dpi=160);plt.close(fig)
    rows=summary['aggregates'];text=['# R4拟合权重的区域迁移与独立seed测试','',
        '原权重为四参数解耦第30轮best.pt，新权重为R4受控拟合A组第1325步best.pt。本次没有训练或运行规划器。','',
        '主表条件固定[0.5,0.6,0.4,0.05]，原始代价尺度为0–50。所有比较使用相同XYZ和教师标签。','',
        '## 其他R案例','',
        'R1、R2是同层区域，R3直梯，R4是本次已拟合折返梯，R5旋梯，R6跨两层。这里评估的是原VST参考路径周围0.5m三维范围内的代价误差，没有重新规划，所以不把点误差当作路径成功率。','',
        '| 案例 | 查询点 | 原权重MAE | R4权重MAE |','|---|---:|---:|---:|']
    for row in summary['route_regions']:text.append(f'| R{row["R"]} {row["id"]} | {row["points"]} | {row["before"]["MAE_raw50"]:.3f} | {row["after"]["MAE_raw50"]:.3f} |')
    text+=['','## 同图各楼梯','',
        '其余楼梯标签没有进入R4损失，但完整建筑几何曾作为输入上下文，因此这部分称为同图区域迁移。','',
        '| 楼梯 | 查询MAE前→后 | 原始PCD支撑MAE前→后 |','|---|---:|---:|']
    for row in summary['same_map_stairs']:
        text.append(f'| {row["id"]}'+('（已拟合R4）' if row['trained_R4'] else '')+f' | {row["query_before"]["MAE_raw50"]:.3f} → {row["query_after"]["MAE_raw50"]:.3f} | {row["raw_before"]["MAE_raw50"]:.3f} → {row["raw_after"]["MAE_raw50"]:.3f} |')
    text+=['','## 独立seed','',
        '从既有v4独立测试划分中，预先选定直梯、折返梯、旋梯、斜坡各一个seed。这些几何未参加R4拟合，结果代表相同生成器分布下的独立场景测试。','',
        '| seed / 类型 | 查询MAE前→后 | 原始PCD支撑MAE前→后 |','|---|---:|---:|']
    for row in summary['independent_seeds']:text.append(f'| {row["id"]} / {row["kind"]} | {row["query_before"]["MAE_raw50"]:.3f} → {row["query_after"]["MAE_raw50"]:.3f} | {row["raw_before"]["MAE_raw50"]:.3f} → {row["raw_after"]["MAE_raw50"]:.3f} |')
    text+=['',f'其余7处楼梯中，查询MAE改善{rows["other_stairs_query_improved"]}/7，等权平均 {rows["other_stairs_query_before"]:.3f} → {rows["other_stairs_query_after"]:.3f}。',
        f'4个独立seed中，查询MAE改善{rows["independent_query_improved"]}/4，场景等权平均 {rows["independent_query_before"]:.3f} → {rows["independent_query_after"]:.3f}。',
        '原始PCD仅在对应原生GT ground的支撑位置计算误差，其他位置GT保持未知。完整预测保留所有输入点。','',
        '## 文件','',
        '- plots/generalization_summary.png：R1–R6、8处楼梯与4个独立seed汇总。',
        '- all_stairs_audit/plots：每处楼梯GT／原权重／R4权重的俯视与三维图。',
        '- plots/scene_*：4个独立seed的GT对比图。',
        '- predictions/*/prediction.pcd：独立seed的完整原始PCD位置预测。',
        '- metrics.csv、summary.json：逐区域点误差、阈值错误率和原始PCD结果。',
        '- initial_test.json、best_test.json：两套条件库的四参数响应评估。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('REPORT_COMPLETE',summary['aggregates'],flush=True)


if __name__=='__main__':main()
