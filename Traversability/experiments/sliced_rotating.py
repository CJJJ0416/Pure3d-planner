"""Local four-scene rotation, resumed cost training, full-PCD inference and report."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import time

import torch

from Traversability.data.rotating import generation_seeds
from Traversability.data.sliced import progress
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Traversability/configs/sliced_tiny_rotating100.json')
    args = parser.parse_args()
    config = read_json(args.config)
    output = workspace_path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    source = workspace_path(config['source_run'])
    if not (output/'protocol.json').exists():
        count = (config['epochs']-config['start_epoch'])//config['rotation']['epochs_per_generation']
        write_json(output/'protocol.json', dict(config=config, status='running',
            seed_plan=[dict(index=g, seeds=generation_seeds(config,g),
                first_epoch=config['start_epoch']+1+g*config['rotation']['epochs_per_generation'],
                last_epoch=config['start_epoch']+(g+1)*config['rotation']['epochs_per_generation']) for g in range(count)],
            source_checkpoint_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['best.pt','last.pt']},
            teacher_sha256=hashlib.sha256(workspace_path(config['teacher_config']).read_bytes()).hexdigest(),
            encoder_sha256=hashlib.sha256(workspace_path(config['encoder']['weights']).read_bytes()).hexdigest(),
            scene_storage='Only the active training generation retains mesh, PCD, labels, features and WFC caches.',
            fixed_splits='Existing validation/test geometry, labels and features are referenced through symlinks.'))
    execution = read_json(output/'execution.json') if (output/'execution.json').exists() else dict(stages=[])
    execution.update(pid=os.getpid(), status='running', started=time.strftime('%Y-%m-%d %H:%M:%S'))

    def run(stage, interpreter, module, arguments):
        execution['stage'] = stage
        with (output/(stage+'.log')).open('a') as log:
            process = subprocess.Popen([interpreter,'-u','-m',module,*map(str,arguments)], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT)
            execution['worker_pid'] = process.pid
            write_json(output/'execution.json',execution)
            status = process.wait()
        execution['stages'].append(dict(stage=stage,exit_code=status))
        write_json(output/'execution.json',execution)
        if status:
            execution.update(status='failed',failed_stage=stage)
            write_json(output/'execution.json',execution)
            raise subprocess.CalledProcessError(status,module)

    options = ['--config',args.config.resolve()]
    if not (output/'last.pt').exists():
        run('train_51',config['neural_python'],'Traversability.training.rotating',
            [*options,'--stop-after-epoch',config['start_epoch']+1])
    run('train_52_100',config['neural_python'],'Traversability.training.rotating',[*options,'--resume'])
    summary = read_json(output/'summary.json')
    metadata = read_json(workspace_path(config['data'])/'manifest.json')
    test = next(s for s in metadata['scenes'] if s['split']=='test')
    progress(config,'independent_inference',completed_epochs=config['epochs'],
             last_validation=summary['best_validation'],test=summary['test'])
    run('independent_inference',config['neural_python'],'Traversability.inference.sliced',
        ['--checkpoint',output/'best.pt','--pcd',test['source_pcd'],
         '--condition','.5','.6','.4','.05','--output',output/'independent_prediction'])
    run('report',config['geometry_python'],'Traversability.visualization.rotating_report',['--run',output])
    protocol = read_json(output/'protocol.json')
    assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==v for n,v in protocol['source_checkpoint_sha256'].items())
    protocol.update(status='complete',completed_epochs=config['epochs'],selected_epoch=summary['selected_epoch'])
    write_json(output/'protocol.json',protocol)
    execution.update(status='complete',stage='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(output/'execution.json',execution)
    storage = read_json(output/'storage_verification.json')
    progress(config,'complete',completed_epochs=config['epochs'],selected_epoch=summary['selected_epoch'],
             last_validation=summary['best_validation'],test=summary['test'],report=str(output/'REPORT.md'),
             generation=9,generations_completed=10,new_scenes_generated=40,
             active_seeds=storage['active_seeds'],temporary_bytes=storage['active_bytes'])
    print(output/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
