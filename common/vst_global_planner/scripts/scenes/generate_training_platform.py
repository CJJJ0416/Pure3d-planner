#!/usr/bin/env python3
"""Generate one combined scene with glocal2-nav's existing mesh primitives."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.terrain import create_scene
from vst_global_planner.export import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/training_platform.json")
    parser.add_argument("--glocal2-root",type=Path,default=ROOT.parents[1])
    parser.add_argument("--seed",type=int)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--skip-points",action="store_true",help="Geometry/view checks only")
    parser.add_argument("--tomography",action="store_true",help="Also build real GPU VST labels")
    args = parser.parse_args()
    if args.tomography and args.skip_points:
        parser.error("--tomography requires the generated point cloud")
    if args.output.exists():
        parser.error("Output already exists. Choose a new directory to preserve its assets.")
    cfg = json.loads(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    print(f"Generating seed {cfg['seed']} ...",flush=True)
    scene,mesh,groups,paths = create_scene(cfg,args.glocal2_root)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles. Validating/exporting ...",flush=True)
    report = save(args.output,cfg,scene,mesh,groups,paths,args.glocal2_root,not args.skip_points)
    print(json.dumps(dict(output=str(args.output.resolve()),geometry_passed=report["passed"],
                          checks=len(report["checks"]),point_count=scene.get("point_count")),ensure_ascii=False),flush=True)
    if args.tomography:
        from vst_global_planner.supervision import build_supervision
        summary=build_supervision(args.output,args.glocal2_root)
        print(json.dumps(dict(tomography_shape=summary["shape"],planning_regions=len(summary["planning_regions"]))),flush=True)


if __name__ == "__main__":
    main()

