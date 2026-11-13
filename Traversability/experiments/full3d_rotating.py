"""Local full-3D cost training, one real resume, independent inference and report."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import time

from Traversability.data.rotating import generation_seeds
from Traversability.data.sliced import progress
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'Traversability/configs/full3d_tiny_rotating100.json')
    args=parser.parse_args()
    config=read_json(args.config)
    target=config.get('stop_epoch',config['epochs'])
    output=workspace_path(config['output']); output.mkdir(parents=True,exist_ok=True)
    source=workspace_path(config['source_run'])
    if not (output/'protocol.json').exists():
        write_json(output/'protocol.json',dict(config=config,status='running',
            seed_plan=[dict(index=g,seeds=generation_seeds(config,g),first_epoch=1+10*g,last_epoch=10*(g+1)) for g in range(10)],
            source_checkpoint_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['best.pt','last.pt']},
            teacher_sha256=hashlib.sha256(workspace_path(config['teacher_config']).read_bytes()).hexdigest(),
            encoder_sha256=hashlib.sha256(workspace_path(config['encoder']['weights']).read_bytes()).hexdigest(),
            fixed_splits='Original XYZ, labels, conditions and core partitions. Newly encoded full3D feature caches.',
            storage='Only current four training buildings retain large artifacts. Fixed caches and all light archives persist.'))
    execution=read_json(output/'execution.json') if (output/'execution.json').exists() else dict(stages=[])
    execution.update(pid=os.getpid(),status='running',started=time.strftime('%Y-%m-%d %H:%M:%S'))
    if (output/'stop_request.json').exists():
        protocol=read_json(output/'protocol.json')
        protocol.update(stop_request=read_json(output/'stop_request.json'),effective_target_epoch=target)
        write_json(output/'protocol.json',protocol)
    def run(stage,interpreter,module,arguments):
        execution['stage']=stage
        with (output/(stage+'.log')).open('a') as stream:
            process=subprocess.Popen([interpreter,'-u','-m',module,*map(str,arguments)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT)
            execution['worker_pid']=process.pid
            write_json(output/'execution.json',execution)
            code=process.wait()
        execution['stages'].append(dict(stage=stage,exit_code=code))
        if code: execution.update(status='failed',failed_stage=stage)
        write_json(output/'execution.json',execution)
        if code: raise subprocess.CalledProcessError(code,module)
    options=['--config',args.config.resolve()]
    if not (output/'last.pt').exists():
        run('train_01',config['neural_python'],'Traversability.training.full3d',[*options,'--stop-after-epoch',1])
    if not (output/'summary.json').exists():
        run(f'train_resume_to_{target}',config['neural_python'],'Traversability.training.full3d',[*options,'--resume'])
    summary=read_json(output/'summary.json')
    meta=read_json(workspace_path(config['data'])/'manifest.json')
    test=next(s for s in meta['scenes'] if s['split']=='test')
    progress(config,'independent_inference',completed_epochs=target,last_validation=summary['best_validation'],test=summary['test'])
    run('independent_inference',config['neural_python'],'Traversability.inference.full3d',
        ['--checkpoint',output/'best.pt','--pcd',test['source_pcd'],'--condition','.5','.6','.4','.05',
         '--output',output/'independent_prediction'])
    run('report',config['geometry_python'],'Traversability.visualization.full3d_report',['--run',output])
    protocol=read_json(output/'protocol.json')
    assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==v for n,v in protocol['source_checkpoint_sha256'].items())
    protocol.update(status='complete',completed_epochs=target,selected_epoch=summary['selected_epoch'])
    write_json(output/'protocol.json',protocol)
    execution.update(status='complete',stage='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(output/'execution.json',execution)
    storage=read_json(output/'storage_verification.json')
    progress(config,'complete',completed_epochs=target,selected_epoch=summary['selected_epoch'],
        last_validation=summary['best_validation'],test=summary['test'],report=str(output/'REPORT.md'),
        generations_completed=summary['generations'],new_scenes_generated=summary['new_scenes'],
        active_seeds=storage['active_seeds'],temporary_bytes=storage['active_bytes'])
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
