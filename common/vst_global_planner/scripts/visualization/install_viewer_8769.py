#!/usr/bin/env python3
"""Point the existing managed 8769 viewer at a prepared playground route bundle."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
SERVICE="glocal2-viewer-8769.service"


def fetch(port,endpoint):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/{endpoint}",timeout=5) as response:
        return json.load(response)


def systemctl(*args):
    return subprocess.run(["systemctl","--user",*args],check=True,capture_output=True,text=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    p.add_argument("--stop-test-port",type=int)
    args=p.parse_args()
    scene=args.scene.resolve()
    manifest=json.loads((scene/"viewer_manifest.json").read_text())
    if not manifest["route_ids"]: raise ValueError("Prepare complete route playback first")
    if args.stop_test_port:
        state=fetch(args.stop_test_port,"api/random-route")
        health=fetch(args.stop_test_port,"api/health")
        if state["scene"]!=str(scene) or health["service"]!="glocal2-scene-viewer":
            raise ValueError("Temporary viewer is not the expected playground")
        if state.get("state") in ("planning","rendering"):
            raise ValueError("Wait for the temporary viewer's random route to finish")
        os.kill(health["pid"],signal.SIGTERM)
    path=Path.home()/".config/systemd/user"/(SERVICE+".d")/"playground.conf"
    previous=path.read_bytes() if path.exists() else None
    backup=ROOT/".cache"/("viewer8769_deploy_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True,exist_ok=False)
    (backup/"previous_unit.txt").write_text(systemctl("cat",SERVICE).stdout,encoding="utf-8")
    if previous is not None: (backup/"previous_dropin.conf").write_bytes(previous)
    path.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-X","utf8",
             str(ROOT/"scripts/visualization/serve_playground.py"),str(scene),"--port","8769"]
    quote=lambda v:'"'+v.replace('\\','\\\\').replace('"','\\"').replace('%','%%')+'"'
    body="[Service]\nExecStart=\nExecStart="+" ".join(quote(x) for x in command)+"\n"
    body+="WorkingDirectory="+str(ROOT)+"\nEnvironment=PYTHONUTF8=1\nEnvironment=PYTHONIOENCODING=utf-8\n"
    path.write_text(body,encoding="utf-8")
    try:
        systemctl("daemon-reload")
        systemctl("restart",SERVICE)
        for _ in range(40):
            try:
                health=fetch(8769,"api/health")
                current=fetch(8769,"api/random-route")
                if health["state"]=="ok" and current["scene"]==str(scene):
                    record=dict(scene=str(scene),url="http://127.0.0.1:8769/index.html",
                                service=SERVICE,dropin=str(path),backup=str(backup),pid=health["pid"])
                    (scene/"viewer_deployment.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
                    print(json.dumps(record),flush=True)
                    return
            except (OSError,ValueError): pass
            time.sleep(.25)
        raise RuntimeError("8769 did not load the expected playground")
    except Exception:
        if previous is None: path.unlink(missing_ok=True)
        else: path.write_bytes(previous)
        systemctl("daemon-reload")
        systemctl("restart",SERVICE)
        raise


if __name__=="__main__": main()

