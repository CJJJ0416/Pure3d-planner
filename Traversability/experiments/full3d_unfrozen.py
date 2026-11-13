"""Same-seed unfrozen Tiny training, real resume, held-out inference and comparison."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import time

from Traversability.data.sliced import progress
from utils.pointcloud import ROOT,read_json,write_json,workspace_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'Traversability/configs/full3d_tiny_unfrozen50.json')
    args=parser.parse_args();config=read_json(args.config)
    output=workspace_path(config['output']);output.mkdir(parents=True,exist_ok=True)
    source=workspace_path(config['source_run']);frozen=workspace_path(config['frozen_reference_run'])
    if not (output/'protocol.json').exists():
        checkpoints=[source/'best.pt',source/'last.pt',frozen/'best.pt',frozen/'last.pt']
        write_json(output/'protocol.json',dict(config=config,status='running',
            source_checkpoint_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints},
            seed_plan=[dict(index=g,seeds=list(range(2026111400+4*g,2026111404+4*g)),first_epoch=10*g+1,last_epoch=10*g+10) for g in range(5)],
            teacher_sha256=hashlib.sha256(workspace_path(config['teacher_config']).read_bytes()).hexdigest(),
            initial_epoch=90,encoder_trainable_from_first_update=True,epochs=50,scheduler_budget_epochs=100,
            frozen_features_used=False,test_selection='Only original fixed validation score selects weights. SUST is evaluated after training.'))
    execution=read_json(output/'execution.json') if (output/'execution.json').exists() else dict(stages=[])
    execution.setdefault('initial_started',time.strftime('%Y-%m-%d %H:%M:%S'))
    execution.update(pid=os.getpid(),status='running')
    def run(name,interpreter,module,arguments):
        execution['stage']=name;tick=time.perf_counter()
        with (output/(name+'.log')).open('a') as stream:
            proc=subprocess.Popen([interpreter,'-u','-m',module,*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            execution['worker_pid']=proc.pid;write_json(output/'execution.json',execution)
            code=proc.wait()
        execution['stages'].append(dict(stage=name,exit_code=code,wall_s=time.perf_counter()-tick))
        if code:execution.update(status='failed',failed_stage=name)
        write_json(output/'execution.json',execution)
        if code:raise subprocess.CalledProcessError(code,module)
    opts=['--config',args.config.resolve()]
    if not (output/'last.pt').exists():
        run('train_01',config['neural_python'],'Traversability.training.full3d_finetune',[*opts,'--stop-after-epoch',1])
    if not (output/'summary.json').exists():
        run('train_02_50',config['neural_python'],'Traversability.training.full3d_finetune',[*opts,'--resume'])
    summary=read_json(output/'summary.json')
    meta=read_json(workspace_path(config['data'])/'manifest.json');test=next(s for s in meta['scenes'] if s['split']=='test')
    for label,pcd in [('independent_prediction',test['source_pcd']),
                      ('building_SUST',ROOT/'datasets/rot_stairs/pcd_files/building_SUST.pcd')]:
        if (output/label/'prediction.json').exists():continue
        progress(config,label,completed_epochs=50,last_validation=summary['best_validation'],test=summary['test'])
        run(label,config['neural_python'],'Traversability.inference.full3d',
            ['--checkpoint',output/'best.pt','--pcd',pcd,'--condition','.5','.6','.4','.05','--output',output/label])
    run('report',config['geometry_python'],'Traversability.visualization.full3d_unfrozen_report',['--run',output])
    protocol=read_json(output/'protocol.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in protocol['source_checkpoint_sha256'].items())
    protocol.update(status='complete',completed_epochs=50,selected_epoch=summary['selected_epoch'])
    write_json(output/'protocol.json',protocol)
    execution.pop('failed_stage',None)
    execution.update(status='complete',stage='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'));write_json(output/'execution.json',execution)
    storage=read_json(output/'storage_verification.json')
    progress(config,'complete',completed_epochs=50,selected_epoch=summary['selected_epoch'],last_validation=summary['best_validation'],
        test=summary['test'],report=str(output/'REPORT.md'),temporary_bytes=storage['active_bytes'],generations_completed=5,
        new_scenes_generated=20)
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':main()
