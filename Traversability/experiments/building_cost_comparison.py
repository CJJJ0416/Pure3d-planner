"""Actual building PCD: full3D, sliced, legacy QueryPool, and native VST cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/traversability/building_SUST_comparison_20261113'
PCD=ROOT/'datasets/rot_stairs/pcd_files/building_SUST.pcd'
NEURAL='/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python'
GEOMETRY='/home/cjh/glocal2-nav/.venv/bin/python'
WEIGHTS={
    'full3d':ROOT/'runs/traversability/cost_full3d_tiny_rotating100_20261113/best.pt',
    'sliced':ROOT/'runs/traversability/cost_sliced_tiny_rotating100_20261113/best.pt',
    'legacy':ROOT/'runs/cost_all_stairs_dr_v3/best.pt'}
CONDITION=[.5,.6,.4,.05]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def read(path):
    return json.loads(path.read_text())


def teacher(output):
    import cupy as cp
    import numpy as np
    from common.pointcloud_io import read_pcd_xyz,write_pcd
    from Traversability.data.sliced import geometry_cloud
    sys.path.insert(0,str(ROOT/'common/vst_global_planner/src'))
    from vst_global_planner.conditioned_teacher import ConditionedTeacher
    from vst_global_planner.tomogram_cloud import tomogram_points
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();tick=started
    coord=read_pcd_xyz(PCD)
    reading=time.perf_counter()-tick
    cp.cuda.Stream.null.synchronize();tick=time.perf_counter()
    vst=ConditionedTeacher(coord)
    cp.cuda.Stream.null.synchronize();geometry_s=time.perf_counter()-tick
    condition=[*CONDITION[:3],.5,CONDITION[3]]
    tick=time.perf_counter();cost=vst.cost(condition)
    cp.cuda.Stream.null.synchronize();cost_s=time.perf_counter()-tick
    tick=time.perf_counter()
    # Same native half-precision cost/ground export, without unused gradient channels.
    cloud=tomogram_points(dict(data=[cost.astype(np.float16),None,None,vst.ground.astype(np.float16)],
        resolution=vst.resolution,center=vst.center,slice_dh=vst.slice_dh,cost_scale=50.))
    expected,_=geometry_cloud(vst.ground,vst.center,vst.resolution,vst.slice_dh)
    assert np.array_equal(expected,cloud[:,:3])
    np.save(output/'coord.npy',cloud[:,:3]);np.save(output/'cost.npy',cloud[:,3])
    write_pcd(output/'tomogram.pcd',cloud[:,:3],cloud[:,3])
    export_s=time.perf_counter()-tick
    write(output/'prediction.json',dict(source_pcd=str(PCD),raw_points=len(coord),points=len(cloud),
        student_condition=CONDITION,teacher_condition=condition,teacher=vst.metadata(),parameters=vst.parameters(condition),
        query_xyz_matches_student_geometry=True,timings=dict(pcd_read_s=reading,geometry_s=geometry_s,
            cost_s=cost_s,export_and_save_s=export_s),total_s=time.perf_counter()-started,
        purpose='Native VST reference cost, without A* or GPMP'))
    print(output/'tomogram.pcd',flush=True)


def legacy_prepare(output):
    import numpy as np
    from scipy.spatial import cKDTree
    from common.pointcloud_io import read_pcd_xyz
    from Traversability.data.blocks import query_blocks
    from utils.pointcloud import voxel_downsample
    settings=read(output.parent/'legacy_geometry_config.json')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();inputs=read_pcd_xyz(PCD)
    query=np.load(output.parent/'vst/coord.npy')
    global_coord=voxel_downsample(inputs,settings['global_voxel_m'])
    _,global_neighbors=cKDTree(global_coord).query(query,k=settings['query_neighbors']['global_k'])
    for name,value in dict(input_coord=inputs,query_coord=query,global_coord=global_coord,global_neighbors=global_neighbors).items():
        np.save(output/(name+'.npy'),value)
    rows=[]
    for index,(ii,core,qi,lower,upper) in enumerate(query_blocks(inputs,query,settings['core_size_m'],settings['context_m'])):
        _,neighbors=cKDTree(inputs[ii]).query(query[qi],k=settings['query_neighbors']['local_k'])
        mask=neighbors<len(ii);neighbors=np.where(mask,neighbors,0)
        name=f'block_{index:05d}'
        np.savez(output/(name+'.npz'),input_indices=ii,query_indices=qi,query_neighbors=neighbors,query_neighbor_mask=mask)
        rows.append(dict(id=name,scene='inference',file=name+'.npz',core_lower_m=lower.tolist(),
            core_upper_m=upper.tolist(),input_points=len(ii),query_points=len(qi)))
        if (index+1)%100==0:print('COMMON QUERY PREPARE',index+1,flush=True)
    scene=dict(id='inference',arrays={k:k+'.npy' for k in ['input_coord','query_coord','global_coord','global_neighbors']},
        input_points=len(inputs),query_points=len(query),global_points=len(global_coord))
    write(output/'blocks.json',dict(scenes=[scene],blocks=rows,preparation_s=time.perf_counter()-started,
        query_source='Identical native Tomogram support XYZ, supplied for controlled accuracy comparison'))


def legacy_common(output):
    import numpy as np
    import torch
    from Traversability.inference.inference import QueryBlocks
    from Traversability.inference.predict import predict
    from Traversability.models.model import ConditionedCost
    from common.pointcloud_io import write_pcd
    state=torch.load(WEIGHTS['legacy'],map_location='cpu',weights_only=False)
    config=state['config'];torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    output.mkdir(parents=True,exist_ok=True)
    dataset=QueryBlocks(output.parent/'legacy_common_geometry','inference',config['model'],config['seed'],CONDITION)
    model=ConditionedCost(config['model'],state['condition_scale'],pretrained=False).to('cuda')
    model.load_state_dict(state['model'])
    torch.cuda.synchronize();tick=time.perf_counter()
    values=predict(model,dataset,config,torch.device('cuda'))
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    coord=np.load(output.parent/'vst/coord.npy')
    np.save(output/'prediction.npy',values);write_pcd(output/'prediction.pcd',coord,values)
    write(output/'prediction.json',dict(checkpoint=str(WEIGHTS['legacy']),epoch=state['epoch'],
        points=len(values),prediction_s=seconds,blocks=len(dataset),query_source='Shared VST geometry queries, no cost provided to model'))


def report(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from common.pointcloud_io import read_pcd_xyz
    ref=read(output/'vst/prediction.json');xyz=np.load(output/'vst/coord.npy');truth=np.load(output/'vst/cost.npy')
    predictions={name:np.load(output/name/'prediction.npy') for name in ['full3d','sliced','legacy_common']}
    for name in ['full3d','sliced']:
        assert np.array_equal(xyz,np.load(output/name/'geometry/coord.npy'))
    metrics={}
    for name,values in predictions.items():
        error=values-truth;low=truth<=.5;high=~low
        metrics[name]=dict(points=len(values),mae_normalized=float(abs(error).mean()),mae_raw=float(50*abs(error).mean()),
            rmse_raw=float(50*np.sqrt((error**2).mean())),signed_bias_raw=float(50*error.mean()),
            false_block_pct=float(100*((values>.5)&low).sum()/low.sum()),false_block_denominator=int(low.sum()),
            false_pass_pct=float(100*((values<=.5)&high).sum()/high.sum()),false_pass_denominator=int(high.sum()))
    timings={name:read(output/name/'prediction.json') for name in ['vst','full3d','sliced','legacy_raw','legacy_common']}
    write(output/'summary.json',dict(input=str(PCD),raw_points=ref['raw_points'],support_points=len(xyz),
        student_condition=CONDITION,teacher_condition=ref['teacher_condition'],metrics=metrics,timing_records=timings,
        same_xyz_verified=True,legacy_common_supplied_query_xyz=True,
        timing_note='Native legacy output covers every raw point. Full3D, sliced and VST output support points. Common-query legacy is an additional controlled accuracy evaluation.'))
    figures=output/'plots';figures.mkdir(exist_ok=True)
    # Display sampling only. Metrics and PCD exports retain all points.
    rng=np.random.default_rng(20261113)
    take=np.sort(rng.choice(len(xyz),min(350000,len(xyz)),replace=False))
    cmap=LinearSegmentedColormap.from_list('cost',['#16a34a','#fde047','#ef4444'])
    raw=read_pcd_xyz(PCD);raw_take=np.sort(rng.choice(len(raw),min(250000,len(raw)),replace=False))
    panels=[('Raw PCD | Z',raw[raw_take],raw[raw_take,2],'jet',float(raw[:,2].min()),float(raw[:,2].max()))]
    panels += [('VST reference',xyz[take],truth[take],cmap,0,1)]
    labels=dict(full3d='Full3D | epoch 45',sliced='Sliced | epoch 90',legacy_common='Legacy QueryPool | epoch 20')
    for name in predictions:
        panels.append((labels[name],xyz[take],predictions[name][take],cmap,0,1))
    fig,axes=plt.subplots(1,5,figsize=(26,9),constrained_layout=True,sharex=True,sharey=True)
    for ax,(title,points,values,color,lo,hi) in zip(axes,panels):
        image=ax.scatter(points[:,0],points[:,1],c=values,cmap=color,vmin=lo,vmax=hi,s=.45,lw=0,rasterized=True)
        ax.set_title(title);ax.set_aspect('equal');ax.set(xlabel='Map X (m)',ylabel='Map Y (m)');ax.grid(alpha=.15)
        fig.colorbar(image,ax=ax,shrink=.6,label='Z (m)' if title.startswith('Raw') else 'cost (0-1)')
    fig.suptitle(f'building_SUST | gravity top view | {ref["raw_points"]:,} raw points | {len(xyz):,} support points\n'
        'Plot shows overlapping floors. Display sampled only, all points used in metrics.')
    fig.savefig(figures/'raw_vst_students_top.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,9),sharex=True,sharey=True,constrained_layout=True)
    for ax,(name,values) in zip(axes,predictions.items()):
        error=abs(values-truth)
        image=ax.scatter(xyz[take,0],xyz[take,1],c=error[take],cmap='magma',vmin=0,vmax=.5,s=.55,lw=0,rasterized=True)
        ax.set(title=f'{labels[name]}\nRaw MAE={metrics[name]["mae_raw"]:.3f}',xlabel='Map X (m)',ylabel='Map Y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.colorbar(image,ax=axes,shrink=.7,label='Absolute normalized cost error',extend='max')
    fig.savefig(figures/'same_xyz_absolute_error.png',dpi=150);plt.close(fig)
    # Height bands are display views only. The encoder input remains complete 3D.
    bands=np.floor(xyz[:,2]/4.).astype(int)
    ids,counts=np.unique(bands,return_counts=True)
    selected=np.sort(ids[np.argsort(counts)[-min(4,len(ids)):]])
    comparison=[('VST reference',truth),*[(labels[n],v) for n,v in predictions.items()]]
    fig,axes=plt.subplots(len(selected),4,figsize=(20,5*len(selected)),sharex=True,sharey=True,
        squeeze=False,constrained_layout=True)
    for row,band in enumerate(selected):
        keep=bands==band
        for col,(label,values) in enumerate(comparison):
            ax=axes[row,col]
            im=ax.scatter(xyz[keep,0],xyz[keep,1],c=values[keep],cmap=cmap,vmin=0,vmax=1,s=.8,lw=0,rasterized=True)
            ax.set_aspect('equal');ax.grid(alpha=.15)
            if row==0:ax.set_title(label)
            ax.set(xlabel='Map X (m)',ylabel=f'{band*4:g} <= Z < {(band+1)*4:g} m\nMap Y (m)')
    fig.colorbar(im,ax=axes,shrink=.6,label='cost (0-1)')
    fig.suptitle('building_SUST | identical support XYZ | height bands for display only')
    fig.savefig(figures/'same_xyz_height_bands.png',dpi=150);plt.close(fig)
    q3=take[::max(1,len(take)//120000)]
    views=[('Raw PCD | Z',raw[raw_take[::2]],raw[raw_take[::2],2],'jet',float(raw[:,2].min()),float(raw[:,2].max())),
           ('VST reference',xyz[q3],truth[q3],cmap,0,1)]
    views += [(labels[n],xyz[q3],v[q3],cmap,0,1) for n,v in predictions.items()]
    fig=plt.figure(figsize=(27,5.5))
    aspect=np.maximum(np.ptp(raw,axis=0),.1)
    for col,(label,points,values,color,lo,hi) in enumerate(views):
        ax=fig.add_axes([.015+col*.195,.15,.17,.73],projection='3d')
        ax.scatter(*points.T,c=values,cmap=color,vmin=lo,vmax=hi,s=.45,lw=0,depthshade=False,rasterized=True)
        ax.set(title=label,xlabel='Map X (m)',ylabel='Map Y (m)',zlabel='Map Z (m)')
        ax.set_box_aspect(aspect);ax.view_init(elev=22,azim=-60)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(norm=Normalize(float(raw[:,2].min()),float(raw[:,2].max())),cmap='jet'),
        cax=fig.add_axes([.035,.065,.13,.018]),orientation='horizontal',label='Raw Z (m)')
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),
        cax=fig.add_axes([.30,.065,.62,.018]),orientation='horizontal',label='Support cost (0-1)')
    fig.suptitle('building_SUST | raw geometry and conditional support costs | identical camera',y=.98)
    fig.savefig(figures/'raw_vst_students_3d.png',dpi=140,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    text=f'''# building_SUST：模型框架与原生VST对比

原始输入{ref['raw_points']:,}点。坐标保持map米制。学生参数{CONDITION}，教师固定interval_free=0.5、p=2、障碍代价50。
直接使用原始PCD，无Filter、Recon、路线采样或人为裁剪。VST是规则参考，不是实测通行性标注。

## 从PCD到各自原生输出的耗时，单次实测

| 方法 | 输出点数 | 入口内部总耗时 | Python启动至退出墙钟 |
|---|---:|---:|---:|
'''
    stages=read(output/'execution.json')['stages']
    for name,label in [('vst','原生VST'),('full3d','完整3D Tiny，epoch45'),('sliced','切片Tiny，epoch90'),('legacy_raw','旧QueryPool，epoch20')]:
        r=timings[name];elapsed=r.get('total_s',r.get('entry_to_prediction_files_s'))
        wall=next(s['wall_s'] for s in stages if s['stage']==name)
        text+=f"| {label} | {r['points']:,} | {elapsed:.3f}s | {wall:.3f}s |\n"
    text+='\n旧QueryPool原生入口输出所有原始点，其余方法输出支撑点，原生总耗时同时反映输出规模差异。\n'
    raw_geometry=read(output/'legacy_raw/geometry/blocks.json')
    text+='\n### 旧框架CPU准备分项\n\n| 阶段 | 秒 |\n|---|---:|\n'
    for name,value in raw_geometry['timings'].items():text+=f'| {name} | {value:.3f} |\n'
    text+=f"\n旧框架预测循环{timings['legacy_raw']['prediction_s']:.3f}秒，包含数据加载、CPU体素预处理和GPU网络。局部阶段包含分块、局部KD-tree近邻查询和索引缓存保存。\n"
    text+=f'\n另外用同一组{len(xyz):,}个Tomogram XYZ直接查询旧QueryPool，保证误差比较不受最近邻重采样影响。这条受控路径使用外部几何查询位置，只用于比较，不计入上表原生预测时间。\n'
    text+='\n## 同一XYZ上的VST一致性\n\n| 模型 | 原始50尺度MAE | 有符号偏差 | GT低代价点误阻断率 | GT高代价点错误放行率 |\n|---|---:|---:|---:|---:|\n'
    for name,m in metrics.items():
        label=labels[name].replace(' | ',' / ')
        text+=f"| {label} | {m['mae_raw']:.4f} | {m['signed_bias_raw']:.4f} | {m['false_block_pct']:.2f}% | {m['false_pass_pct']:.2f}% |\n"
    text+='\n阈值为0.5。两类错误各以对应GT低代价或高代价点数作分母，详见summary.json。\n'
    text+='\n在该真实建筑上，当前完整3D模型的VST一致性低于切片版和保留的旧QueryPool模型，合成固定测试上的改善没有转化为这栋建筑的改善。\n'
    text+='\n## 输出\n\n- vst/tomogram.pcd：原生VST参考。\n- full3d/prediction.pcd、sliced/prediction.pcd：全部支撑点预测。\n- legacy_raw/prediction.pcd：旧框架全部原始点预测。\n- legacy_common/prediction.pcd：旧框架相同支撑XYZ的直接查询结果。\n- plots/raw_vst_students_3d.png、same_xyz_height_bands.png：统一视角三维图及按高度区间显示的同XYZ对比，显示分组不影响模型输入。\n- plots/raw_vst_students_top.png、same_xyz_absolute_error.png：全场俯视及误差图。全场俯视会叠加不同楼层，采样仅用于显示。\n- execution.json、各子目录prediction.json：真实耗时、参数和数据规模。\n'
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    print(output/'REPORT.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--stage',choices=['vst','legacy_prepare','legacy_common','report'])
    args=parser.parse_args();output=args.output.resolve()
    if args.stage:
        {'vst':teacher,'legacy_prepare':legacy_prepare,'legacy_common':legacy_common,'report':report}[args.stage](output)
        return
    output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',TZ='Asia/Shanghai',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    execution=dict(pid=os.getpid(),status='running',stages=[])
    write(output/'protocol.json',dict(input=str(PCD),condition=CONDITION,weights={k:str(p) for k,p in WEIGHTS.items()},
        checkpoint_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in WEIGHTS.items()},
        comparison='Native output timing plus identical-XYZ VST consistency, no training or planning.'))
    def run(name,interpreter,module,arguments):
        execution['stage']=name;tick=time.perf_counter()
        with (output/(name+'.log')).open('w') as stream:
            process=subprocess.Popen([interpreter,'-u','-m',module,*map(str,arguments)],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT)
            execution['worker_pid']=process.pid;write(output/'execution.json',execution)
            code=process.wait()
        execution['stages'].append(dict(stage=name,wall_s=time.perf_counter()-tick,exit_code=code))
        if code:execution['status']='failed'
        write(output/'execution.json',execution)
        if code:raise subprocess.CalledProcessError(code,module)
        print('COMPLETE',name,execution['stages'][-1]['wall_s'],flush=True)
    own='Traversability.experiments.building_cost_comparison'
    run('vst',GEOMETRY,own,['--stage','vst','--output',output/'vst'])
    for name in ['full3d','sliced']:
        run(name,NEURAL,'Traversability.inference.'+name,['--checkpoint',WEIGHTS[name],'--pcd',PCD,
            '--condition',*CONDITION,'--output',output/name])
    run('legacy_raw',NEURAL,'Traversability.inference.predict',['--checkpoint',WEIGHTS['legacy'],
        '--pcd',PCD,'--geometry-python',GEOMETRY,'--condition',*CONDITION,'--output',output/'legacy_raw'])
    import torch
    state=torch.load(WEIGHTS['legacy'],map_location='cpu',weights_only=False)
    write(output/'legacy_geometry_config.json',state['blocks_config'])
    del state
    run('legacy_prepare',GEOMETRY,own,['--stage','legacy_prepare','--output',output/'legacy_common_geometry'])
    run('legacy_common',NEURAL,own,['--stage','legacy_common','--output',output/'legacy_common'])
    run('report',GEOMETRY,own,['--stage','report','--output',output])
    protocol=read(output/'protocol.json')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==protocol['checkpoint_sha256'][name] for name,p in WEIGHTS.items())
    execution.update(stage='complete',status='complete');write(output/'execution.json',execution)
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
