"""Local native-slice data -> frozen Tiny cache -> train -> independent inference."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from utils.pointcloud import ROOT, read_json, write_json, workspace_path
from Traversability.data.sliced import progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'Traversability/configs/sliced_tiny_local50.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    output, data = workspace_path(config['output']), workspace_path(config['data'])
    output.mkdir(parents=True, exist_ok=True)
    execution = dict(pid=os.getpid(), status='running', started=time.strftime('%Y-%m-%d %H:%M:%S'),
                     config=str(args.config.resolve()), stages=[])

    def run(stage, interpreter, module, arguments):
        execution['stage'] = stage
        write_json(output / 'execution.json', execution)
        if stage in ['independent_inference', 'report']:
            summary = read_json(output / 'summary.json')
            progress(config, stage, completed_epochs=config['epochs'],
                     last_validation=summary['best_validation'], test=summary['test'])
        with (output / (stage + '.log')).open('a') as log:
            process = subprocess.Popen([interpreter, '-u', '-m', module, *map(str, arguments)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            execution['worker_pid'] = process.pid
            write_json(output / 'execution.json', execution)
            status = process.wait()
        execution['stages'].append(dict(stage=stage, exit_code=status))
        if status:
            execution.update(status='failed', failed_stage=stage)
            write_json(output / 'execution.json', execution)
            raise subprocess.CalledProcessError(status, module)

    options = ['--config', args.config.resolve()]
    if not (data / 'manifest.json').exists() or not read_json(data / 'manifest.json').get('complete'):
        run('prepare', config['geometry_python'], 'Traversability.data.sliced', options)
    if not (data / 'encoding.json').exists():
        run('encode', config['neural_python'], 'Traversability.training.sliced', [*options, '--stage', 'encode'])
    if not args.resume:
        run('train_01', config['neural_python'], 'Traversability.training.sliced',
            [*options, '--stage', 'train', '--stop-after-epoch', '1'])
    run('train_02_50', config['neural_python'], 'Traversability.training.sliced',
        [*options, '--stage', 'train', '--resume'])
    manifest = read_json(data / 'manifest.json')
    test = next(s for s in manifest['scenes'] if s['split'] == 'test')
    run('independent_inference', config['neural_python'], 'Traversability.inference.sliced',
        ['--checkpoint', output / 'best.pt', '--pcd', test['source_pcd'],
         '--condition', '.5', '.6', '.4', '.05', '--output', output / 'independent_prediction'])
    run('report', config['geometry_python'], 'Traversability.visualization.sliced_report', ['--run', output])
    execution.update(status='complete', stage='complete', finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(output / 'execution.json', execution)
    summary = read_json(output / 'summary.json')
    progress(config, 'complete', completed_epochs=config['epochs'], selected_epoch=summary['selected_epoch'],
             last_validation=summary['best_validation'], test=summary['test'], report=str(output/'REPORT.md'))
    print(output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
