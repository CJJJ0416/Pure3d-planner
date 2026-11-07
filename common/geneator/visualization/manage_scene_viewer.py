#!/usr/bin/env python3
"""安装和管理当前用户的场景网页服务，每分钟检查响应并按需恢复。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[3]


def systemctl(*args, capture=False):
    return subprocess.run(["systemctl", "--user", *args], check=True, text=True,
                          stdout=subprocess.PIPE if capture else None)


def unit_quote(value, command=False):
    value = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"'+value+'"'


def names(port):
    service = f"glocal2-viewer-{port}"
    return service+".service", service+"-health.service", service+"-health.timer"


def properties(service):
    output = systemctl("show", service, "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID",
                       capture=True).stdout
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def healthy(port, pid):
    # 回环检查不经过用户的 HTTP 代理；同时校验服务类型、工作区和所属 PID。
    with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
        data = json.loads(response.read(4096))
        return response.status == 200 and data == dict(
            service="glocal2-scene-viewer", state="ok", pid=pid, root=str(ROOT))


def check(port):
    service, _, _ = names(port)
    state = properties(service)
    if state["LoadState"] != "loaded":
        raise RuntimeError(f"Service is not installed: {service}")
    if state["ActiveState"] in ("activating", "deactivating"):
        return
    for attempt in range(3):
        state = properties(service)
        if state["ActiveState"] in ("activating", "deactivating"):
            return
        pid = int(state["MainPID"])
        try:
            if pid and healthy(port, pid):
                return
        except (OSError, URLError, ValueError):
            pass
        if attempt < 2:
            time.sleep(2)
    print(f"{service} did not respond to three checks; restarting its own processes only.", flush=True)
    systemctl("restart", service)


def install(scene, port):
    scene = scene.resolve()
    if scene.name == "current" and scene.parent.name == "random_maps" and (scene/"random_map.json").is_file():
        scene = scene.parent.parent
    for name in ("scene.json", "terrain.pcd"):
        if not (scene/name).is_file():
            raise RuntimeError(f"Missing scene file: {scene/name}")
    if not any((scene/name).is_file() for name in ("viewer_data.json", "index.html")):
        raise RuntimeError("Build viewer_data.json before installing the service")
    python = ROOT/".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"Missing Python environment: {python}")
    service, health_service, timer = names(port)
    # 已有非托管监听者时明确报错，不按端口任意杀进程。
    state = properties(service)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            managed = int(state.get("MainPID", "0"))
            if not managed or not healthy(port, managed):
                raise RuntimeError(f"Port {port} is occupied by an unmanaged service; inspect it before installing")
    except ConnectionRefusedError:
        pass
    command = " ".join(unit_quote(arg, command=True) for arg in
                       (python, ROOT/"common/geneator/visualization/serve_scene_viewer.py", scene, "--port", port))
    probe = " ".join(unit_quote(arg, command=True) for arg in
                     (python, Path(__file__).resolve(), "check", "--port", port))
    units = {
        service: f"""[Unit]
Description=glocal2 scene viewer on port {port}
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory={str(ROOT).replace('%', '%%')}
ExecStart={command}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Restart=always
RestartSec=3
TimeoutStopSec=20
KillMode=control-group

[Install]
WantedBy=default.target
""",
        health_service: f"""[Unit]
Description=Check glocal2 viewer on port {port}

[Service]
Type=oneshot
ExecStart={probe}
TimeoutStartSec=40
""",
        timer: f"""[Unit]
Description=Check glocal2 viewer every minute on port {port}

[Timer]
OnStartupSec=60
OnUnitActiveSec=60
AccuracySec=1
Unit={health_service}

[Install]
WantedBy=timers.target
""",
    }
    destination = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home()/".config")))/"systemd/user"
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in units.items():
        (destination/name).write_text(body)
    subprocess.run(["systemd-analyze", "--user", "verify", *(str(destination/name) for name in units)],
                   check=True)
    systemctl("daemon-reload")
    # 桌面变量只从已存在的会话继承，使网页按钮仍可打开 RViz。
    desktop = [name for name in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS")
               if name in os.environ]
    if desktop:
        systemctl("import-environment", *desktop)
    systemctl("enable", service, timer)
    systemctl("restart", service, timer)
    for _ in range(30):
        try:
            if healthy(port, int(properties(service)["MainPID"])):
                print(f"Viewer ready: http://127.0.0.1:{port}/index.html\nService: {service}\nHealth timer: {timer}")
                return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Viewer did not become ready; inspect journalctl --user -u {service}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "check", "status", "restart", "stop"))
    parser.add_argument("scene", type=Path, nargs="?", help="Saved scene directory (required for install)")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    service, _, timer = names(args.port)
    try:
        if args.action == "install":
            if args.scene is None:
                parser.error("install requires a saved scene directory")
            install(args.scene, args.port)
        elif args.action == "check":
            check(args.port)
        elif args.action == "status":
            systemctl("status", "--no-pager", service, timer)
        elif args.action == "stop":
            # 同时停定时器，避免用户主动停止后又被健康检查启动。
            systemctl("stop", timer, service)
        else:
            systemctl("restart", service, timer)
    except (OSError, URLError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
