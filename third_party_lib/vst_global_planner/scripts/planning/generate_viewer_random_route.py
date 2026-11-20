#!/usr/bin/env python3
"""One replaceable random route for the shared playground viewer."""
import argparse
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.routes import RegionRoutes
from vst_global_planner.route_geometry import RouteGeometry
from vst_global_planner.route_viewer import render_one
from vst_global_planner.supervision import write_json_utf8


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--region",required=True)
    p.add_argument("--job-id",required=True)
    p.add_argument("--glocal2-root",type=Path,default=ROOT.parents[1])
    args=p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}",args.job_id): p.error("Invalid job identifier")
    scene,args.output=args.scene.resolve(),args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    def status(state,message):
        write_json_utf8(args.output/"status.json",dict(state=state,message=message))
    try:
        status("planning","正在当前建筑内随机选点并规划…")
        routes=RegionRoutes(scene,args.glocal2_root)
        if args.region not in routes.scene["training_regions"]: raise ValueError("Unknown building region")
        geometry=RouteGeometry(scene/"terrain.obj",routes.scene["config"]["validation"])
        route_seed=int(args.job_id[:16],16)
        rng=np.random.default_rng(route_seed)
        from stair_nav.planning.vst import save_path
        original=Path.cwd()
        failures=[]
        try:
            os.chdir(args.output)
            for attempt in range(1,31):
                try:
                    path,record=routes.plan(args.region,rng,reverse=bool(rng.integers(2)))
                    record["geometry_validation"]=geometry.check(path)
                    save_path(path,args.output/"random.gpmp")
                    break
                except (ValueError,RuntimeError,IndexError) as exc:
                    failures.append(str(exc))
            else: raise RuntimeError("当前建筑内未找到通过检查的随机路线，请再次生成")
        finally:
            os.chdir(original)
        record.update(attempts=attempt,failures=failures,random_seed=route_seed,job_id=args.job_id)
        status("rendering","规划与网格检查完成，正在渲染 RGB 和深度回放…")
        route=render_one(scene,args.output/"random.gpmp.csv","random_"+args.job_id,record)
        # The service atomically renames .pending to current after this process exits.
        # Persist fingerprints against the published path rather than the temporary slot.
        published=scene/"random_routes/current"
        fingerprints={}
        for file,digest in route["report"]["source_fingerprints"].items():
            path=Path(file)
            key=published/path.relative_to(args.output) if path.is_relative_to(args.output) else path
            fingerprints[str(key)]=digest
        route["report"].update(source_fingerprints=fingerprints,random_seed=route_seed,job_id=args.job_id)
        write_json_utf8(args.output/"request.json",record)
        write_json_utf8(args.output/"route.json",route)
        status("ready",route["report"]["label"]+" · 已通过离线网格检查")
    except Exception as exc:
        status("error",str(exc))
        raise


if __name__=="__main__": main()

