"""Run one command in its own terminal, recording PID identity and a tee log."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Missing stage command')
    from Entry.utils.io import atomic_json
    from Entry.runtime.cost_session import process_info,send,wait_exit
    child=None;interrupted=False
    def stop(_signum,_frame):raise KeyboardInterrupt
    for event in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(event,stop)
    try:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               start_new_session=True)
        identity=process_info(child.pid)
        atomic_json(args.run/'stages'/(args.name+'.json'),dict(stage=process_info(os.getpid()),
                    process=identity,command=command,state='running'))
        def tee():
            with (args.run/(args.name+'.log')).open('wb') as log:
                while True:
                    data=child.stdout.read1(4096)
                    if not data:break
                    log.write(data);log.flush()
                    try:sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
                    except (BrokenPipeError,OSError):pass
        thread=threading.Thread(target=tee,daemon=True);thread.start()
        while child.poll() is None:time.sleep(.2)
        thread.join(timeout=2)
    except KeyboardInterrupt:
        interrupted=True
    finally:
        for event in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(event,signal.SIG_IGN)
        if child is not None:
            if child.poll() is None:
                send(identity,signal.SIGINT)
                if alive_child := process_info(child.pid):
                    if alive_child['start']==identity['start']:
                        try:
                            # Popen created this session. Reach ros2 run's
                            # keyboard child as a terminal Ctrl+C would.
                            if os.getpgid(child.pid)==child.pid:os.killpg(child.pid,signal.SIGINT)
                        except ProcessLookupError:pass
                remaining=wait_exit([identity],40)
                for row in remaining:send(row,signal.SIGTERM)
                wait_exit(remaining,5)
            atomic_json(args.run/'stages'/(args.name+'.json'),dict(stage=process_info(os.getpid()),
                process=identity,command=command,state='stopped',exit_code=child.poll()))
    if not interrupted and sys.stdin.isatty():
        print(f'\n{args.name} exited ({child.returncode}). Log: {args.run/(args.name+".log")}\n'
              'Press Enter to close this terminal.',flush=True)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        try:input()
        except (KeyboardInterrupt,EOFError):pass
    return 0 if interrupted else child.returncode


if __name__=='__main__':raise SystemExit(main())
