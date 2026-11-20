"""整体方案的内部终端托管：记录 PID 与日志，转发退出信号。

Usage：由 ./Entrance/shell/total_terminal.sh 自动调用，无需手动运行。
"""
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
    # 参数：--run 为本次整体启动的输出目录，--name 为阶段名（gazebo/keyboard/lio/cost）。
    # 剩余参数是要托管的真实命令，允许带 '--' 分隔符。
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Missing stage command')
    # 复用主运行库的进程信息/信号工具，保证“整体启动”和“清理旧会话”使用同一套判定。
    from common.io import atomic_json
    from common.cost.session import process_info,send,wait_exit
    child=None;interrupted=False
    # 统一把 SIGINT/SIGTERM/SIGHUP 转成 KeyboardInterrupt，走同一条清理路径。
    def stop(_signum,_frame):raise KeyboardInterrupt
    for event in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(event,stop)
    try:
        # start_new_session 让子进程成为独立进程组，便于退出时向 ros2 run 及其键盘子进程发信号。
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               start_new_session=True)
        identity=process_info(child.pid)
        # 阶段状态落盘，供 total_terminal.py 判断该终端是否仍在运行。
        atomic_json(args.run/'stages'/(args.name+'.json'),dict(stage=process_info(os.getpid()),
                    process=identity,command=command,state='running'))
        def tee():
            # 子进程输出同时写入 <stage>.log 并回显到当前终端；终端关闭导致的 BrokenPipe 被忽略。
            with (args.run/(args.name+'.log')).open('wb') as log:
                while True:
                    data=child.stdout.read1(4096)
                    if not data:break
                    log.write(data);log.flush()
                    try:sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
                    except (BrokenPipeError,OSError):pass
        thread=threading.Thread(target=tee,daemon=True);thread.start()
        # 轮询等待子进程结束；不使用 child.wait()，以便及时响应信号。
        while child.poll() is None:time.sleep(.2)
        thread.join(timeout=2)
    except KeyboardInterrupt:
        interrupted=True
    finally:
        # 清理期间忽略再次到达的信号，避免清理被中断。
        for event in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(event,signal.SIG_IGN)
        if child is not None:
            if child.poll() is None:
                # 先向已验证身份的进程发 SIGINT；若仍存活且是会话首进程，再向整个进程组补发，
                # 这样 ros2 run 下面的 keyboard_input 也能像终端 Ctrl+C 一样收到中断。
                send(identity,signal.SIGINT)
                if alive_child := process_info(child.pid):
                    if alive_child['start']==identity['start']:
                        try:
                            # Popen created this session. Reach ros2 run's
                            # keyboard child as a terminal Ctrl+C would.
                            if os.getpgid(child.pid)==child.pid:os.killpg(child.pid,signal.SIGINT)
                        except ProcessLookupError:pass
                # 给进程 40 s 正常退出，仍残留则升级为 SIGTERM。
                remaining=wait_exit([identity],40)
                for row in remaining:send(row,signal.SIGTERM)
                wait_exit(remaining,5)
            # 记录最终退出码，total_terminal.py 可据此区分正常结束与异常退出。
            atomic_json(args.run/'stages'/(args.name+'.json'),dict(stage=process_info(os.getpid()),
                process=identity,command=command,state='stopped',exit_code=child.poll()))
    # 人工中断时不等待回车；正常结束时保留终端窗口，方便查看最后的日志。
    if not interrupted and sys.stdin.isatty():
        print(f'\n{args.name} exited ({child.returncode}). Log: {args.run/(args.name+".log")}\n'
              'Press Enter to close this terminal.',flush=True)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        try:input()
        except (KeyboardInterrupt,EOFError):pass
    # 被中断视为成功（用户主动停止），否则返回被托管命令的退出码。
    return 0 if interrupted else child.returncode


if __name__=='__main__':raise SystemExit(main())
