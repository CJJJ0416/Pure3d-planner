"""Prepare 144 scenes, compare A/B for five epochs, continue the winner and report."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from utils.pointcloud import ROOT,read_json,write_json,workspace_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'Traversability/configs/parameter_decoupling_v4.json')
    args=p.parse_args();settings=read_json(args.config)
    run=workspace_path(settings['output']);data=workspace_path(settings['data']);initial=workspace_path(settings['initial_checkpoint'])
    run.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',MPLCONFIGDIR=str(ROOT/'datasets/cache/matplotlib'),CUPY_CACHE_DIR=str(ROOT/'datasets/cache/cupy'))
    (run/'pipeline.pid').write_text(str(os.getpid())+'\n')
    source=torch.load(initial,map_location='cpu',weights_only=False)
    config=copy.deepcopy(source['config'])
    config.update({k:settings[k] for k in ['seed','epochs','batch_size','blocks_per_epoch','num_workers','cpu_threads','amp',
        'freeze_encoder_epochs','learning_rate','encoder_learning_rate','weight_decay','smooth_l1_beta','change_threshold']})
    config.update(data=str(data/'train'),output=str(run),device='cuda:0',initialization_checkpoint=str(initial),
                  baseline_metrics=str(run/'initial_val.json'))
    config['model'].update(decoder_precision='fp32',augment_yaw=False)
    write_json(run/'evaluation_config.json',config)
    if not (run/'initial_fp32.pt').exists():
        baseline=source.copy();baseline['config']=copy.deepcopy(config)
        torch.save(baseline,run/'initial_fp32.pt')
    del source
    protocol=dict(settings=settings,initial_checkpoint=str(initial),initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
                  source_epoch=20,teacher_p=2,cost_scale=50,normalized_targets=True,model_architecture_changed=False,
                  decoder_precision='fp32',selection='scene macro absolute MAE + equal four-parameter changed/stable response MAE; new/replay 0.7/0.3',
                  regression_used_for_training=False)
    write_json(run/'protocol.json',protocol)
    jobs=[]

    def progress(stage,variant=None):
        value=dict(stage=stage,variant=variant,complete=False,settings=settings,jobs=jobs)
        if stage=='data_preparation':
            log=run/'data_preparation.log'
            lines=log.read_text().split('\n')[:-1] if log.exists() else []
            ready=[line for line in lines if line.startswith('READY ')]
            done=int(ready[-1].split()[1]) if ready else 0
            value['data']=dict(done=done,total=149,bar='['+'█'*int(done/149*20)+'░'*(20-int(done/149*20))+f'] {done}/149')
        if variant:
            log=run/variant/'train.log';lines=log.read_text().split('\n')[:-1] if log.exists() else []
            last=next((x for x in reversed(lines) if x.startswith('TRAIN ')),None)
            if last:
                _,epoch,step,loss=last.split(' ',3)
                value.update(epoch=int(epoch),step=int(step),losses=ast.literal_eval(loss))
            history=run/variant/'history.json'
            if history.exists():
                h=read_json(history)
                value['validation_history']=[dict(epoch=r['epoch'],selection_score=r['validation']['selection_score'],groups=r['validation']['groups']) for r in h]
        temporary=run/'progress.json.tmp';write_json(temporary,value);temporary.replace(run/'progress.json')

    def stage(name,arguments,python=None,variant=None):
        marker=run/(name+'.complete.json')
        if marker.exists():jobs.append(read_json(marker));return
        log=run/variant/'train.log' if variant else run/(name+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        print('START',name,flush=True);began=time.perf_counter()
        with log.open('a') as stream:
            child=subprocess.Popen([python or settings['neural_python'],*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write_json(run/'execution.json',dict(complete=False,active=name,pid=child.pid,jobs=jobs))
            while child.poll() is None:
                progress(name,variant);time.sleep(1)
        record=dict(name=name,status=child.returncode,seconds=time.perf_counter()-began)
        jobs.append(record)
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,child.args)
        write_json(marker,record);print('DONE',name,record['seconds'],flush=True)

    if not (data/'data_ready.json').exists():
        if (run/'data.pid').exists():
            data_pid=int((run/'data.pid').read_text())
            while Path(f'/proc/{data_pid}').exists() and not (data/'data_ready.json').exists():
                progress('data_preparation');time.sleep(1)
            if not (data/'data_ready.json').exists():raise RuntimeError('Data process exited. Read data_preparation.log')
        else:
            stage('data_preparation',['-m','Traversability.experiments.decoupling_data','--config',args.config],settings['geometry_python'])
    progress('initial_val')
    stage('initial_val',['-m','Traversability.experiments.decoupling_eval','--checkpoint',run/'initial_fp32.pt',
        '--config',run/'evaluation_config.json','--output',run/'initial_val.json','--predictions',run/'initial_val.npz'])
    for name,variant in settings['variants'].items():
        candidate=copy.deepcopy(config);candidate.update(variant);candidate.update(variant=name,output=str(run/name))
        write_json(run/name/'train_config.json',candidate)
        weights=['--resume',run/name/'last.pt'] if (run/name/'last.pt').exists() else ['--config',run/name/'train_config.json']
        stage('pilot_'+name,['-m','Traversability.training.decoupling_train',*weights,'--stop-after-epoch',5],variant=name)
    if not (run/'selection.json').exists():
        a,b=[read_json(run/name/'history.json')[-1]['validation'] for name in ['A','B']]
        if abs(a['selection_score']-b['selection_score'])<1e-6:
            ma,mb=a['groups']['replay']['cost_mae'],b['groups']['replay']['cost_mae'];winner='A' if ma<=mb else 'B'
        else:winner='A' if a['selection_score']<b['selection_score'] else 'B'
        write_json(run/'selection.json',dict(winner=winner,selection_epoch=5,A=a['selection_score'],B=b['selection_score'],
            rule=protocol['selection'],resume=str(run/winner/'last.pt')))
    winner=read_json(run/'selection.json')['winner']
    stage('continue_'+winner,['-m','Traversability.training.decoupling_train','--resume',run/winner/'last.pt','--stop-after-epoch',30],variant=winner)
    for name in ['best.pt','last.pt']:
        link=run/name
        if not link.is_symlink():link.symlink_to(Path(winner)/name)
    for label,checkpoint in [('initial',run/'initial_fp32.pt'),('best',run/'best.pt')]:
        for split in ['val','test']:
            stage(label+'_'+split+'_final',['-m','Traversability.experiments.decoupling_eval','--checkpoint',checkpoint,
                '--config',run/'evaluation_config.json','--split',split,'--output',run/(label+'_'+split+'.json'),
                '--predictions',run/(label+'_'+split+'.npz')])
        stage(label+'_regression',['-m','Traversability.training.evaluate','--checkpoint',checkpoint,
            '--data',ROOT/'datasets/wfc_regression_20261127','--split','test','--output',run/(label+'_regression.json'),
            '--predictions',run/(label+'_regression.npz')])
        stage(label+'_r4',['-m','Traversability.experiments.decoupling_r4','--checkpoint',checkpoint,'--output',run/(label+'_r4.npz')])
    manifest=read_json(data/'manifest.json')
    examples=[(kind,next(s['pcd'] for s in manifest['scenes'] if s['source']=='new' and s['split']=='test' and s['kind']==kind)) for kind in settings['kinds']]
    examples.append(('large_terrain',str(ROOT/'datasets/wfc_unseen_20261127/scene_20261127/terrain_downsampled.pcd')))
    for name,pcd in examples:
        stage('predict_'+name,['-m','Traversability.inference.predict','--checkpoint',run/'best.pt','--pcd',pcd,
            '--geometry-python',settings['geometry_python'],'--condition',.5,.6,.4,.05,'--output',run/'predictions'/name])
    stage('all_stairs_audit',['-m','Traversability.experiments.all_stair_audit','--run',run],settings['geometry_python'])
    stage('report',['-m','Traversability.experiments.decoupling_report','--run',run],settings['geometry_python'])
    write_json(run/'execution.json',dict(complete=True,active=None,jobs=jobs))
    write_json(run/'progress.json',dict(stage='complete',complete=True,winner=winner,summary=read_json(run/'summary.json')))
    print('COMPLETE',run,flush=True)


if __name__=='__main__':main()
