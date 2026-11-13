"""从当前 best.pt 执行宽深高楼梯增强、微调、独立测试和大图回归。"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from utils.pointcloud import ROOT,read_json,write_json,workspace_path


def main(default_config=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=default_config or ROOT/'Traversability/configs/wide_stair_v2.json')
    args=parser.parse_args();settings=read_json(args.config)
    run=workspace_path(settings['output']);data=workspace_path(settings['data']);initial=workspace_path(settings['initial_checkpoint'])
    run.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['PYTHONPATH']=str(ROOT)
    os.environ['MPLCONFIGDIR']=str(ROOT/'datasets/cache/matplotlib');os.environ['CUPY_CACHE_DIR']=str(ROOT/'datasets/cache/cupy')
    jobs=[]
    protocol=dict(settings=settings,initial_checkpoint=str(initial),initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
        model_changes=False,loss_changes=False,teacher_changes=False,
        replay_fraction=1-settings['training']['stair_blocks']/settings['training']['blocks_per_epoch'],
        selection='固定验证保留约束优先，再按楼梯断裂率、stand响应MAE、整体MAE排序',
        regression_seed=20261127,regression_used_for_training=False,source_epoch=10,new_epoch_counter=True,
        geometry_template_source=settings.get('template_scene'),cost_scale=50,normalized_targets=True,teacher_p=2)
    write_json(run/'protocol.json',protocol)
    (run/'pipeline.pid').write_text(str(os.getpid())+'\n')
    with (run/'progress.log').open('a') as log:
        progress=subprocess.Popen(['/usr/bin/python3','-m','Traversability.visualization.progress',
                                   '--run',str(run),'--watch'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (run/'progress.pid').write_text(str(progress.pid)+'\n')
    def stage(name,arguments,python=sys.executable):
        start=time.perf_counter();print('START',name,flush=True)
        with (run/(name+'.log')).open('a') as log:
            p=subprocess.Popen([python,*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write_json(run/'execution.json',dict(complete=False,active=name,pid=p.pid,jobs=jobs))
            status=p.wait()
        jobs.append(dict(name=name,seconds=time.perf_counter()-start,status=status))
        write_json(run/'execution.json',dict(complete=False,active=None,jobs=jobs))
        if status:raise subprocess.CalledProcessError(status,p.args)
        print('DONE',name,jobs[-1]['seconds'],flush=True)
    if not (data/'data_ready.json').exists():
        stage('data_preparation',['-m',settings.get('data_module','Traversability.experiments.wide_stair_data'),
                                 '--config',args.config],settings['geometry_python'])
    stage('prepare_connectivity',['-m','Traversability.experiments.stair_metrics','--prepare',data/'evaluation'],settings['geometry_python'])
    stage('prepare_retention_connectivity',['-m','Traversability.experiments.stair_metrics','--prepare',data/'retention'],settings['geometry_python'])
    if not (run/'initial_val.json').exists():
        stage('initial_val',['-m','Traversability.training.evaluate','--checkpoint',initial,'--data',data/'evaluation','--split','val',
                             '--output',run/'initial_val.json','--predictions',run/'initial_val.npz'])
    stage('initial_retention_val',['-m','Traversability.training.evaluate','--checkpoint',initial,'--data',data/'retention',
                           '--split','val','--output',run/'initial_retention_val.json'])
    ckpt=torch.load(initial,map_location='cpu',weights_only=False);config=copy.deepcopy(ckpt['config']);del ckpt
    config.update(settings['training'])
    config.update(data=str(data/'train'),validation_data=str(data/'evaluation'),output=str(run),seed=settings['seed'],
                  stair_sampling=dict(stair_blocks=settings['training']['stair_blocks'],source='new'),
                  stair_selection_baseline=str(run/'initial_val.json'),initialization_checkpoint=str(initial))
    if settings['training'].get('stair_balance_by'):
        config['stair_sampling']['balance_by']=settings['training']['stair_balance_by']
    config['_comment']=settings.get('title','宽深高直梯独立seed增强')+'；原网络、损失和教师规则保持。'
    write_json(run/'train_config.json',config)
    weights=['--resume',run/'last.pt'] if (run/'last.pt').exists() else ['--config',run/'train_config.json','--initialize',initial]
    stage('train',['-m','Traversability.training.train',*weights])
    stage('final_val',['-m','Traversability.training.evaluate','--checkpoint',run/'best.pt','--data',data/'evaluation','--split','val',
                       '--output',run/'final_val.json','--predictions',run/'final_val.npz'])
    for label,checkpoint in [('initial',initial),('best',run/'best.pt')]:
        stage(label+'_test',['-m','Traversability.training.evaluate','--checkpoint',checkpoint,'--data',data/'evaluation','--split','test',
                             '--output',run/(label+'_test.json'),'--predictions',run/(label+'_test.npz')])
        stage(label+'_regression',['-m','Traversability.training.evaluate','--checkpoint',checkpoint,'--data',workspace_path(settings['regression_data']),
               '--split','test','--output',run/(label+'_regression.json'),'--predictions',run/(label+'_regression.npz')])
    stage('retention_val',['-m','Traversability.training.evaluate','--checkpoint',run/'best.pt','--data',data/'retention',
                           '--split','val','--output',run/'retention_val.json'])
    manifest=read_json(data/'manifest.json');new_test=next(s for s in manifest['scenes'] if s['split']=='test')
    pcds=[('new_test',data/'scenes'/f"scene_{new_test['seed']}"/'terrain_downsampled.pcd'),
          ('large_terrain',ROOT/'datasets/wfc_unseen_20261127/scene_20261127/terrain_downsampled.pcd')]
    if settings.get('template_scene'):
        for kind in ['switchback','spiral']:
            example=next(s for s in manifest['scenes'] if s['split']=='test' and s['actual']['kind']==kind)
            pcds.append(('new_test_'+kind,data/'scenes'/f"scene_{example['seed']}"/'terrain_downsampled.pcd'))
    for name,pcd in pcds:
        stage('predict_'+name,['-m','Traversability.inference.predict','--checkpoint',run/'best.pt','--pcd',pcd,
              '--geometry-python',settings['geometry_python'],'--condition',.5,.6,.4,.05,'--output',run/'predictions'/name])
    stage('report',['-m',settings.get('report_module','Traversability.experiments.wide_stair_report'),
                    '--run',run],settings['geometry_python'])
    write_json(run/'execution.json',dict(complete=True,active=None,jobs=jobs))
    print('COMPLETE',run,flush=True)


if __name__=='__main__':main()
