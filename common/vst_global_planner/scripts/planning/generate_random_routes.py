#!/usr/bin/env python3
"""Generate native region-constrained random VST/GPMP reference routes."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.terrain import load_backend
from vst_global_planner.routes import generate_routes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--glocal2-root",type=Path,default=ROOT.parents[1])
    p.add_argument("--seed",type=int,default=20261104)
    p.add_argument("--per-region",type=int,default=2)
    p.add_argument("--max-attempts",type=int,default=30)
    args=p.parse_args()
    load_backend(args.glocal2_root)
    records=generate_routes(args.scene,args.output,args.glocal2_root,args.seed,args.per_region,args.max_attempts)
    print(f"Planned {sum(r['optimized'] for r in records)}/{len(records)} routes",flush=True)


if __name__=="__main__":
    main()
