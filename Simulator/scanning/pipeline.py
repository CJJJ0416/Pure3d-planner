"""Generate -> real Faster-LIO replay -> supervision/blocks -> visual comparison."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from utils.pointcloud import write_json
from .generate import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--playback-rate', type=float)
    parser.add_argument('--ros-domain-id', type=int, default=78)
    parser.add_argument('--stages', nargs='+', default=['generate', 'replay', 'prepare', 'plot'])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output is not None:
        config['output'] = str(args.output)
    if args.frames is not None:
        config['frames'] = args.frames
    if args.playback_rate is not None:
        config['playback_rate'] = args.playback_rate
    root = Path(config['output']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    actual = root / 'config.actual.json'
    write_json(actual, config)
    profiles = args.profiles or config['profiles']
    workspace = Path(__file__).resolve().parents[2]
    if 'generate' in args.stages:
        subprocess.run([sys.executable, '-m', 'Simulator.scanning.generate', '--config', str(actual),
                        '--profiles', *profiles], cwd=workspace, check=True)
    for profile in profiles:
        if 'replay' in args.stages:
            quote = shlex.quote
            script = '\n'.join([
                'source /opt/ros/humble/setup.bash',
                'source ' + quote(str(Path(config['lio_workspace'])/'install/local_setup.bash')),
                'export ROS_DOMAIN_ID=' + str(args.ros_domain_id),
                'export ROS_LOG_DIR=' + quote(str(root/profile/'ros_logs')),
                'export PYTHONPATH=' + quote(str(workspace)) + ':${PYTHONPATH}',
                'exec /usr/bin/python3 -m Simulator.scanning.replay --config ' + quote(str(actual)) +
                ' --profile ' + quote(profile)])
            subprocess.run(['/bin/bash', '-c', script], cwd=workspace, check=True)
        for stage, module in [('prepare', 'prepare'), ('plot', 'visualize')]:
            if stage in args.stages:
                subprocess.run([sys.executable, '-m', 'Simulator.scanning.' + module,
                    '--root', str(root), '--profile', profile], cwd=workspace, check=True)
    write_json(root / 'pipeline.json', dict(profiles=profiles, stages=args.stages,
        ros_domain_id=args.ros_domain_id,
        requested_frames=config.get('frames', round(config['duration_s']*config['scan_hz'])),
        training_started=False))


if __name__ == '__main__':
    main()
