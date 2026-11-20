"""Single CUDA producer overlapped with isolated CPU Faster-LIO consumers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter

from utils.pointcloud import read_json, write_json
from .generate import load_config


def run(config_path, schedule, output=None, report_output=None):
    settings = read_json(config_path)
    workspace = Path(__file__).resolve().parents[2]
    root = Path(output or Path(settings['output']) / schedule).resolve()
    reports = Path(report_output or Path(settings['report_output']) / schedule).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    base = load_config(settings['base_config'])
    started = perf_counter()
    events, jobs = [], []
    lock = Lock()

    def command(stage, route, argv):
        begin = perf_counter() - started
        log_path = reports / f'{route}_{stage}.log'
        with log_path.open('w') as log:
            subprocess.run(argv, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=True)
        end = perf_counter() - started
        event = dict(route=route, stage=stage, start_s=begin, end_s=end, wall_s=end-begin, log=str(log_path))
        with lock:
            events.append(event)
            write_json(reports / 'timeline.json', sorted(events, key=lambda row: row['start_s']))
        print(json.dumps(event), flush=True)

    def consume(row):
        command('lio', row['id'], [sys.executable, '-m', 'Simulator.scanning.pipeline',
            '--config', row['config'], '--profiles', settings['profile'],
            '--ros-domain-id', str(row['ros_domain_id']), '--stages', 'replay'])
        command('prepare', row['id'], [sys.executable, '-m', 'Simulator.scanning.prepare',
            '--root', row['root'], '--profile', settings['profile']])

    # The queue is bounded by the two worker slots. Only one process owns CUDA.
    pending = []
    with ThreadPoolExecutor(max_workers=settings['max_in_flight']) as workers:
        for index, route in enumerate(settings['routes']):
            if len(pending) == settings['max_in_flight']:
                pending.pop(0).result()
            folder = root / route['id']
            folder.mkdir(parents=True, exist_ok=True)
            config = {**base, **route, 'output': str(folder), 'profiles': [settings['profile']],
                'playback_rate': settings['playback_rate'], 'split': 'train',
                'explicit_ghost_probability': settings['explicit_ghost_probability'],
                'explicit_ghost_pull_m': settings['explicit_ghost_pull_m'],
                'warp_cache': str(root / '.warp_cache')}
            if index:
                config['scene_source'] = jobs[0]['root']
            actual = folder / 'config.requested.json'
            write_json(actual, config)
            row = dict(id=route['id'], root=str(folder), config=str(actual),
                ros_domain_id=settings['ros_domain_base']+index)
            jobs.append(row)
            command('warp', row['id'], [sys.executable, '-m', 'Simulator.scanning.generate',
                '--config', str(actual), '--profiles', settings['profile']])
            if schedule == 'pipeline':
                pending.append(workers.submit(consume, row))
            else:
                consume(row)
        for future in pending:
            future.result()
    wall = perf_counter() - started
    scenes, results = [], []
    for row in jobs:
        folder = Path(row['root']) / settings['profile']
        metrics = read_json(folder / 'metrics.json')
        replay = read_json(folder / 'replay.json')
        generation = read_json(folder / 'generation.json')
        scene_id = str(folder.relative_to(root))
        scenes.append(dict(scene_id=scene_id, split='train', geometry_group_id=f"fasterlio_{base['seed']}"))
        results.append(dict(id=row['id'], root=row['root'], ros_domain_id=replay['ros_domain_id'],
            input_frames=replay['input_frames'], registered_frames=replay['registered_frames'],
            missing_frames=replay['missing_frames'], points=metrics['points'], boxes=metrics['boxes'],
            explicit_ghost_points=metrics['explicit_ghost_points'], dynamic_points=metrics['dynamic_points'],
            mixed_return_points=metrics['mixed_return_points'], keep_target=metrics['keep_target'],
            mesh_mae_m=metrics['scan_to_mesh']['mean_m'], generation=generation,
            replay=replay, preparation=metrics['timings_s']))
    manifest = dict(version='shared_scans_v1', scenes=scenes, test_suites=['nominal'],
        split_unit='geometry_group_id, all routes of the same building stay together',
        input='single fused XYZ map, no sensor history as model input',
        tasks=['Filter score regression', 'Complete static reference diagnostics'],
        reference_loader='Simulator.scanning.training_data.SharedScanBoxes',
        training_started=False)
    write_json(root / 'manifest.json', manifest)
    warp = [row for row in events if row['stage'] == 'warp']
    lio = [row for row in events if row['stage'] == 'lio']
    overlap = sum(max(0, min(a['end_s'], b['end_s'])-max(a['start_s'], b['start_s']))
        for a in warp for b in lio)
    report = dict(schedule=schedule, wall_s=wall, routes=results, events=events,
        warp_lio_overlap_s=overlap, input_frames=sum(r['input_frames'] for r in results),
        registered_frames=sum(r['registered_frames'] for r in results), points=sum(r['points'] for r in results),
        input_frames_per_s=sum(r['input_frames'] for r in results)/wall,
        points_per_s=sum(r['points'] for r in results)/wall, settings=settings,
        manifest=str(root / 'manifest.json'), training_started=False)
    write_json(reports / 'report.json', report)
    write_json(reports / 'config.actual.json', settings)
    print('BATCH_COMPLETE', schedule, wall, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('Simulator/scanning/batch_config.json'))
    parser.add_argument('--schedule', choices=['serial', 'pipeline'], default='pipeline')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report-output', type=Path)
    args = parser.parse_args()
    run(args.config, args.schedule, args.output, args.report_output)


if __name__ == '__main__':
    main()
