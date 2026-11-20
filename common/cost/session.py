"""Independent cost launcher. Replace only this workspace's cost processes."""
from common.paths import ROOT
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


MODULES = {'common.cost.session', 'common.cost.local', 'common.cost.worker', 'common.cost.monitor'}
# 只用于识别同工作区迁移前的旧进程，不提供旧模块导入或启动入口。
LEGACY_MODULES = {
    'Entry.runtime.cost_session': 'common.cost.session',
    'Entry.runtime.local': 'common.cost.local',
    'Entry.predict': 'common.cost.worker',
    'Entry.runtime.monitor_local': 'common.cost.monitor',
}


def status_print(message):
    """A closed terminal pipe must not stop inference or its monitor."""
    try:
        print(message,flush=True)
    except BrokenPipeError:
        # Keep the existing stdout buffer valid through Python finalization.
        with open(os.devnull,'w') as sink:
            os.dup2(sink.fileno(),sys.stdout.fileno())
        return False
    return True


def process_info(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path/'stat').read_text().rsplit(') ', 1)[1].split()
        args = (path/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
        env = dict(row.split('=', 1) for row in (path/'environ').read_bytes().decode(
            errors='replace').split('\0') if '=' in row)
        return dict(pid=int(pid), start=fields[19], state=fields[0], args=args,
                    ppid=int(fields[1]), project=env.get('PURE3D_WS'),
                    cwd=str((path/'cwd').resolve()), domain=env.get('ROS_DOMAIN_ID', '0'))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def cost_module(info, domain):
    if not info or info['state'] == 'Z' or info['domain'] != str(domain):
        return None
    args = info['args']
    if '-m' not in args or args.index('-m')+1 >= len(args):
        return None
    module = args[args.index('-m')+1]
    module = LEGACY_MODULES.get(module, module)
    if module in MODULES and info['cwd'] == str(ROOT):
        return module
    return None


def alive(info):
    current = process_info(info['pid'])
    return bool(current and current['start'] == info['start'] and current['state'] != 'Z')


def send(info, value):
    if alive(info):
        try:
            os.kill(info['pid'], value)
        except ProcessLookupError:
            pass


def wait_exit(infos, timeout):
    end = time.monotonic()+timeout
    while time.monotonic() < end and any(alive(row) for row in infos):
        time.sleep(.1)
    return [row for row in infos if alive(row)]


def clean_previous(domain):
    previous = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit() and int(path.name) != os.getpid():
            row = process_info(int(path.name))
            if cost_module(row, domain):
                previous.append(row)
    # An existing launcher owns its receiver. Let it close that receiver once.
    sessions = [row for row in previous if cost_module(row, domain) == 'common.cost.session']
    for row in sessions:
        send(row, signal.SIGINT)
    wait_exit(sessions, 40)
    # Let receiver.close() drain and stop its separately-sessioned GPU worker.
    for row in previous:
        if cost_module(row, domain) != 'common.cost.worker':
            send(row, signal.SIGINT)
    remaining = wait_exit(previous, 35)
    for row in remaining:
        send(row, signal.SIGTERM)
    if wait_exit(remaining, 5):
        raise RuntimeError('Old cost processes did not exit. No other modules were stopped.')
    return [dict(pid=row['pid'], module=cost_module(row, domain)) for row in previous]


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['tiny', 'small'], default='tiny')
    parser.add_argument('--config', type=Path, default=ROOT/'Entrance/config/local_cost.yaml')
    parser.add_argument('--output', type=Path, help='New session directory, containing local/ and monitor/')
    parser.add_argument('--rviz', action='store_true', help='Open the local cost RViz layout')
    parser.add_argument('--no-monitor', action='store_true', help='Run prediction without the timing sidecar')
    args = parser.parse_args()
    # Help and config errors must have no effect on a running session.
    from common.mapping.local_map import load_local_config
    from common.io import atomic_json
    import yaml
    checked = load_local_config(args.config, args.model)
    if not Path(checked['checkpoint']).is_file() or not Path(checked['network_python']).is_file():
        parser.error('Cost checkpoint or network Python does not exist')
    if args.output and args.output.exists():
        parser.error('--output must be a new directory')
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    children, lock, startup_lock, run = {}, None, None, None
    state, error = 'starting', None
    try:
        # Serialize replacements before either launcher can signal the other.
        # Release at readiness so a later explicit launch can replace this one.
        startup_lock = open(f'/tmp/pure3d_cost_startup_{os.getuid()}_domain_{domain}.lock', 'a')
        fcntl.flock(startup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cleaned = clean_previous(domain)
        lock = open(f'/tmp/pure3d_cost_session_{os.getuid()}_domain_{domain}.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('cost_%Y%m%d_%H%M%S_%f')
        run = (args.output or ROOT/'outputs/entry_local'/name).resolve()
        run.mkdir(parents=True, exist_ok=False)
        cfg = yaml.safe_load(args.config.read_text())
        cfg.pop('step_completion',None)
        cfg.update(model=args.model)
        config = run/'local_config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest = dict(domain=domain, model=args.model, knn_filter=checked.get('knn_filter',{}), independent_cost=True,
                        scan_cost_integration=False, cleaned=cleaned, processes={})

        def spawn(name, command):
            with (run/(name+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            row = process_info(child.pid)
            if row is None:
                raise RuntimeError(f'{name} exited before startup. See {run/name}.log')
            children[name] = (child, row)
            manifest['processes'][name] = dict(pid=child.pid, start=row['start'],
                command=command, log=str(run/(name+'.log')))
            atomic_json(run/'processes.json', manifest)

        own = process_info(os.getpid())
        manifest['processes']['cost_session'] = dict(pid=os.getpid(), start=own['start'])
        # 外部单项入口为 Entrance/launch/sim/local.launch.py；内部子进程直接复用同一 Python 模块。
        command = ['/usr/bin/python3', '-u', '-m', 'common.cost.local', '--config', str(config), '--model',
                   args.model, '--output', str(run/'local')]
        if args.rviz:
            command.append('--rviz')
        spawn('local', command)
        (ROOT/'outputs/entry_local/current_cost_run.txt').write_text(str(run)+'\n')
        status_print(f'Cost session: {run}\nWaiting for LIO odometry, world TF and /faster_lio/cloud_registered. '
              'Ctrl+C stops this cost session only.')
        last_print = 0.
        while True:
            for name, (child, _) in children.items():
                if child.poll() is not None:
                    raise RuntimeError(f'{name} exited with {child.returncode}. See {run/name}.log')
            path = run/'local/status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            if status.get('error'):
                raise RuntimeError(status['error'])
            if state == 'starting' and status.get('valid'):
                if not args.no_monitor:
                    spawn('monitor', ['/usr/bin/python3', '-m', 'common.cost.monitor', '--run', str(run)])
                state = 'ready'
                startup_lock.close()
                startup_lock = None
                status_print('Cost ready. Sensors, LIO and keyboard remain independent.')
            atomic_json(run/'session_status.json', dict(state=state, domain=domain,
                model=args.model, valid=status.get('valid', False), output=str(run)))
            if time.monotonic()-last_print >= 5:
                summary = run/'monitor/summary.json'
                latency = json.loads(summary.read_text()).get('full_frame_to_cost_ms', {}) if summary.exists() else {}
                status_print(f"state={status.get('state', 'starting')} valid={status.get('valid', False)} "
                      f"points={status.get('points', 0)} "
                      f"E2E median/P95(ms)={latency.get('median')}/{latency.get('p95')}")
                last_print = time.monotonic()
            time.sleep(.5)
    except KeyboardInterrupt:
        state = 'stopped'
    except Exception as exc:
        state, error = 'failed', str(exc)
        status_print(f'Cost startup failed: {error}')
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        rows = [row for _, row in children.values()]
        workers = []
        if run:
            for path in Path('/proc').iterdir():
                if path.name.isdigit():
                    row = process_info(int(path.name))
                    if cost_module(row, domain) == 'common.cost.worker' and any(
                            str(run) in value for value in row['args']):
                        workers.append(row)
        for row in reversed(rows):
            send(row, signal.SIGINT)
        remaining = wait_exit(rows+workers, 35)
        for row in remaining:
            send(row, signal.SIGTERM)
        remaining = wait_exit(remaining, 5)
        for child, _ in children.values():
            if child.poll() is not None:
                child.wait()
        if run:
            atomic_json(run/'session_status.json', dict(state=state, error=error,
                remaining_pids=[row['pid'] for row in remaining], domain=domain, model=args.model))
        if lock:
            lock.close()
        if startup_lock:
            startup_lock.close()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
