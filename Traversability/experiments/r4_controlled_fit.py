"""Run the authorized local A/B/C R4 diagnosis and its final PCD inference."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

from utils.pointcloud import ROOT,read_json,workspace_path
from Traversability.experiments.r4_controlled_data import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'Traversability/configs/r4_controlled_fit_v5.json')
    args=p.parse_args();config=read_json(args.config);run=workspace_path(config['output']);run.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',CUPY_CACHE_DIR=str(ROOT/'datasets/cache/cupy'),
                      MPLCONFIGDIR=str(run/'matplotlib'))
    write_json(run/'execution.json',dict(stage='preparing',pid=os.getpid(),complete=False,configuration=config))
    started=time.perf_counter();jobs=[]

    def execute(name,python,module,arguments,log_path,variant=None):
        log_path.parent.mkdir(parents=True,exist_ok=True)
        command=[python,'-u','-m',module,*arguments]
        with log_path.open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            job=dict(name=name,pid=process.pid,command=command,log=str(log_path));jobs.append(job)
            while process.poll() is None:
                state=dict(stage=name,complete=False,pid=os.getpid(),child_pid=process.pid,jobs=jobs,
                           elapsed_s=time.perf_counter()-started)
                if variant and (run/variant/'progress.json').exists():state['training']=read_json(run/variant/'progress.json')
                if name=='teacher' and (run/'preparation.json').exists():state['preparation']=read_json(run/'preparation.json')
                if 'training' in state:
                    training=state['training'];done=training['step'];total=config['pilot_steps'] if name.endswith('pilot') else config['total_steps']
                    state['text_progress']='['+'█'*min(20,int(done/total*20))+'░'*max(0,20-int(done/total*20))+f'] {variant} {done}/{total}'
                write_json(run/'progress.json',state);write_json(run/'execution.json',state)
                time.sleep(3)
            job['returncode']=process.returncode
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
        print('STAGE_COMPLETE',name,flush=True)

    if not (run/'preparation.json').exists() or not read_json(run/'preparation.json')['complete']:
        execute('teacher',config['geometry_python'],'Traversability.experiments.r4_controlled_data',
                ['--config',str(args.config.resolve())],run/'teacher.log')
    for variant in ['A','B','C']:
        state=run/variant/'training_complete.json'
        if state.exists() and read_json(state)['step']>=config['pilot_steps']:
            if not (run/variant/'pilot_predictions.npz').exists():
                shutil.copy2(run/variant/'last_predictions.npz',run/variant/'pilot_predictions.npz')
            continue
        arguments=['--config',str(args.config.resolve()),'--variant',variant,'--stop-step',str(config['pilot_steps'])]
        if (run/variant/'last.pt').exists():arguments.append('--resume')
        execute(variant+'_pilot',config['neural_python'],'Traversability.experiments.r4_controlled_train',arguments,
                run/variant/'train.log',variant)
        if not (run/variant/'pilot_predictions.npz').exists():
            shutil.copy2(run/variant/'last_predictions.npz',run/variant/'pilot_predictions.npz')
    selection_path=run/'selection.json'
    if selection_path.exists():selection=read_json(selection_path)
    else:
        rows={v:next(r for r in read_json(run/v/'evaluations.json') if r['step']==config['pilot_steps']) for v in ['A','B','C']}
        winner='A'
        for v in ['B','C']:
            if rows[v]['metrics']['score']<rows[winner]['metrics']['score']-1e-6:winner=v
        selection=dict(winner=winner,step=config['pilot_steps'],scores={v:r['metrics']['score'] for v,r in rows.items()},
            rule='Nominal S=0.6 tread macro MAE +0.4 landing macro MAE +0.25 balanced spatial delta MAE. Ties <1e-6: A/B/C.',
            resume=str(run/winner/'last.pt'))
        write_json(selection_path,selection)
    winner=selection['winner'];state=read_json(run/winner/'training_complete.json')
    if state['step']<config['total_steps']:
        execute('winner_continue',config['neural_python'],'Traversability.experiments.r4_controlled_train',
            ['--config',str(args.config.resolve()),'--variant',winner,'--stop-step',str(config['total_steps']),'--resume'],
            run/winner/'train.log',winner)
    for name in ['best.pt','last.pt']:
        target=run/name
        if not target.exists():target.symlink_to(Path(winner)/name)
    if not (run/'prediction/prediction.json').exists():
        execute('full_PCD_prediction',config['neural_python'],'Traversability.inference.predict',
            ['--checkpoint',str(run/'best.pt'),'--pcd',str(workspace_path(config['source_pcd'])),
             '--geometry-python',config['geometry_python'],'--condition',*map(str,config['baseline']),
             '--output',str(run/'prediction')],run/'prediction.log')
    if not (run/'production_batch_R4.npz').exists():
        execute('production_batch_R4_check',config['neural_python'],'Traversability.experiments.decoupling_r4',
            ['--checkpoint',str(run/'best.pt'),'--output',str(run/'production_batch_R4.npz')],run/'production_batch_R4.log')
    execute('report',config['geometry_python'],'Traversability.experiments.r4_controlled_report',
            ['--run',str(run)],run/'report.log')
    protocol=read_json(run/'protocol.json')
    write_json(run/'verification.json',dict(source_checkpoint_unchanged=sha(workspace_path(config['source_checkpoint']))==protocol['source_checkpoint_sha256'],
        source_PCD_unchanged=sha(workspace_path(config['source_pcd']))==protocol['source_pcd_sha256'],
        variants={v:read_json(run/v/'training_complete.json') for v in ['A','B','C']}))
    state=dict(complete=True,stage='complete',winner=winner,elapsed_s=time.perf_counter()-started,
               best_checkpoint=str(run/'best.pt'),report=str(run/'REPORT.md'),jobs=jobs)
    summary=read_json(run/'summary.json')
    state.update(best_step=summary['best_step'],nominal=summary['best']['metrics']['nominal'],
                 tread_MAE_raw50=summary['best']['metrics']['tread_macro_MAE_raw50'],
                 spatial=summary['best']['metrics']['spatial'],acceptance=summary['acceptance'])
    write_json(run/'execution.json',state);write_json(run/'progress.json',state)
    print('ALL_COMPLETE',run,flush=True)


if __name__=='__main__':main()
