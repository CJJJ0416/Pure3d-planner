"""Compare real sliced/full3D entrypoints on one PCD, plus a separately timed profile."""
import argparse
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/traversability/cost_sliced_tiny_rotating100_20261113/best.pt'
NEW=ROOT/'runs/traversability/cost_full3d_tiny_rotating100_20261113/best.pt'
PCD=ROOT/'datasets/wfc_capability_dr_v2/scene_20261114/terrain.pcd'
DEFAULT=ROOT/'runs/traversability/cost_inference_comparison_20261113'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def profile(kind,output):
    import torch
    import point_transformer_v3 as ptv3
    from Traversability.models.sliced import SlicedCostHead
    import Traversability.inference.sliced as old
    import Traversability.inference.full3d as new
    counters={}
    def timer(name,function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            torch.cuda.synchronize(); tick=time.perf_counter()
            value=function(*args,**kwargs)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
            row=counters.setdefault(name,dict(seconds=0.,calls=0))
            row['seconds']+=elapsed;row['calls']+=1
            if name=='prepare':
                n=len(args[0]);row['input_points_sum']=row.get('input_points_sum',0)+n
                row['input_points_max']=max(row.get('input_points_max',0),n)
            if name=='tiny_forward':
                n=len(args[1].coord);row['input_voxels_sum']=row.get('input_voxels_sum',0)+n
                row['input_voxels_max']=max(row.get('input_voxels_max',0),n)
            return value
        return wrapped
    ptv3.prepare=timer('prepare',ptv3.prepare)
    ptv3.PointTransformerV3.forward=timer('tiny_forward',ptv3.PointTransformerV3.forward)
    SlicedCostHead.forward=timer('adapter_film_cost_head',SlicedCostHead.forward)
    old.encode_window=timer('encoding_pipeline',old.encode_window)
    new.encode_context=timer('encoding_pipeline',new.encode_context)
    torch.cuda.init()
    sys.argv=['inference','--checkpoint',str(SOURCE if kind=='sliced' else NEW),'--pcd',str(PCD),
        '--condition','.5','.6','.4','.05','--output',str(output)]
    (old.main if kind=='sliced' else new.main)()
    write(output/'profile.json',dict(gpu=torch.cuda.get_device_name(0),counters=counters,
        note='Nested encoding timers overlap. Profile timings are separate from uninstrumented fresh-process trials.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--profile',choices=['sliced','full3d'])
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.profile:
        profile(args.profile,output)
        return
    env=dict(os.environ,PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',TZ='Asia/Shanghai')
    records=[]
    for repeat in range(3):
        for kind in (['sliced','full3d'] if repeat%2==0 else ['full3d','sliced']):
            dest=output/f'fresh_{kind}'
            command=[sys.executable,'-u','-m',f'Traversability.inference.{"sliced" if kind=="sliced" else "full3d"}',
                '--checkpoint',str(SOURCE if kind=='sliced' else NEW),'--pcd',str(PCD),
                '--condition','.5','.6','.4','.05','--output',str(dest)]
            tick=time.perf_counter()
            with (output/f'{kind}_{repeat+1}.log').open('w') as stream:
                subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            elapsed=time.perf_counter()-tick
            result=json.loads((dest/'prediction.json').read_text())
            records.append(dict(kind=kind,repeat=repeat+1,process_wall_s=elapsed,inference=result))
            write(output/'measurements.json',records)
            print(f'{kind} repeat={repeat+1} process={elapsed:.4f}s internal={result["total_s"]:.4f}s',flush=True)
    for kind in ['sliced','full3d']:
        with (output/f'profile_{kind}.log').open('w') as stream:
            subprocess.run([sys.executable,'-u','-m','Traversability.experiments.benchmark_cost_inference',
                '--profile',kind,'--output',str(output/f'profile_{kind}')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'{kind} profile complete',flush=True)
    import numpy as np
    models={}
    for kind in ['sliced','full3d']:
        rows=[r for r in records if r['kind']==kind]
        internal=[r['inference']['total_s'] for r in rows]
        wall=[r['process_wall_s'] for r in rows]
        timing=dict(total_s=internal,process_wall_s=wall,median_total_s=float(np.median(internal)),
            median_process_wall_s=float(np.median(wall)),minimum_total_s=min(internal),maximum_total_s=max(internal))
        details={}
        if kind=='sliced':
            for key in ['geometry_s','prediction_s']:
                details[key]=float(np.median([r['inference'][key] for r in rows]))
        else:
            for key in rows[0]['inference']['timing']:
                details[key]=float(np.median([r['inference']['timing'][key] for r in rows]))
        counters=json.loads((output/f'profile_{kind}/profile.json').read_text())['counters']
        models[kind]=dict(timing=timing,stage_medians=details,profile=counters,
            output_points=rows[0]['inference']['points'],encoder_calls=rows[0]['inference']['encoder_calls'])
    a=np.load(output/'fresh_sliced/geometry/coord.npy');b=np.load(output/'fresh_full3d/geometry/coord.npy')
    assert np.array_equal(a,b)
    gpu=json.loads((output/'profile_full3d/profile.json').read_text())['gpu']
    report=dict(source_pcd=str(PCD),gpu=gpu,conditions=[.5,.6,.4,.05],repeats=3,
        methodology='Three alternating, sequential fresh-process CLI requests per model. Existing OS/JIT caches kept. No feature cache used. Profile measurements collected separately.',
        identical_output_xyz=True,models=models,
        slowdown_ratio=models['full3d']['timing']['median_total_s']/models['sliced']['timing']['median_total_s'])
    write(output/'summary.json',report)
    old_s=models['sliced']['timing']['median_total_s'];new_s=models['full3d']['timing']['median_total_s']
    text=f'''# 单PCD代价预测测速

相同PCD：{PCD}。GPU为{gpu}。相同参数[0.5,0.6,0.4,0.05]，各3次独立新进程、交替顺序运行，不复用特征缓存。
保留现有系统文件和Triton编译缓存。输出XYZ完全一致，共{len(a):,}点，各244块。

| 指标 | 旧切片模型，第90轮 | 完整3D模型，第45轮 |
|---|---:|---:|
| 入口内部总耗时中位数 | {old_s:.4f}s | {new_s:.4f}s |
| 含Python进程启动的墙钟中位数 | {models['sliced']['timing']['median_process_wall_s']:.4f}s | {models['full3d']['timing']['median_process_wall_s']:.4f}s |
| 内部总耗时范围 | {models['sliced']['timing']['minimum_total_s']:.4f}–{models['sliced']['timing']['maximum_total_s']:.4f}s | {models['full3d']['timing']['minimum_total_s']:.4f}–{models['full3d']['timing']['maximum_total_s']:.4f}s |

完整3D内部总耗时增加{100*(new_s/old_s-1):.2f}%。吞吐按输出支撑点计，旧{len(a)/old_s:.0f}点/秒，新{len(a)/new_s:.0f}点/秒。
两者内部计时均包括读取权重、原生几何查询准备、模型构造、预测及输出文件写入，不包括Python导入阶段。

## 单独插桩的前向拆分

| 项目 | 旧切片 | 完整3D |
|---|---:|---:|
'''
    for label,key in [('Tiny纯前向','tiny_forward'),('Tiny预处理','prepare'),('Adapter+FiLM+CostHead','adapter_film_cost_head'),('编码与特征映射流程，包含Tiny前向及预处理','encoding_pipeline')]:
        text+=f"| {label} | {models['sliced']['profile'][key]['seconds']:.4f}s | {models['full3d']['profile'][key]['seconds']:.4f}s |\n"
    text+='\n插桩使用CUDA同步计时，单独运行，不混入上面的三次未插桩总耗时。编码流程计时包含其内部子阶段，不能重复相加。\n\n完整3D未插桩各阶段中位数见summary.json的stage_medians。逐次原始数值见measurements.json。\n'
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
