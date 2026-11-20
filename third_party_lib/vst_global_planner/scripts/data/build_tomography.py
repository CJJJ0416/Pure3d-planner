#!/usr/bin/env python3
"""Generate fresh VST supervision for an existing generated platform."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.supervision import build_supervision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene",type=Path)
    parser.add_argument("--glocal2-root",type=Path,default=ROOT.parents[1])
    args=parser.parse_args()
    summary=build_supervision(args.scene,args.glocal2_root)
    print(json.dumps(dict(output=str(args.scene.resolve()),**summary)),flush=True)


if __name__=="__main__":
    main()

