#!/usr/bin/env python3
"""Render saved native playground routes for the shared 8769 viewer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.route_viewer import build_viewer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    p.add_argument("routes",type=Path)
    p.add_argument("--glocal2-root",type=Path,default=ROOT.parents[1])
    p.add_argument("--frames",type=int)
    args=p.parse_args()
    result=build_viewer(args.scene,args.routes,args.glocal2_root,args.frames)
    print(json.dumps(dict(routes=len(result["route_ids"]),data=str(args.routes/"viewer_data.json"))),flush=True)


if __name__=="__main__": main()
