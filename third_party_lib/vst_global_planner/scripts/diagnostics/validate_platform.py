#!/usr/bin/env python3
"""Check geometric openings, surfaces and connectivity across platform seeds."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.terrain import create_scene,validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",nargs="+",type=int,default=[20261104,20261105,20261106])
    p.add_argument("--config",type=Path,default=ROOT/"configs/training_platform.json")
    p.add_argument("--glocal2-root",type=Path,default=ROOT.parents[1])
    p.add_argument("--output",type=Path,default=ROOT/"outputs/seed_geometry_validation.json")
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    reports=[]
    for seed in args.seeds:
        cfg["seed"]=seed
        scene,mesh,groups,paths=create_scene(cfg,args.glocal2_root)
        report=dict(seed=seed,**validate(scene,mesh,paths))
        reports.append(report)
        print(json.dumps(dict(seed=seed,passed=report["passed"],checks=len(report["checks"]))),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(reports,indent=2)+"\n")
    if not all(r["passed"] for r in reports):
        raise SystemExit(1)


if __name__=="__main__":
    main()
