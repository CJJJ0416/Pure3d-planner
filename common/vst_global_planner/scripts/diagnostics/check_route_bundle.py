#!/usr/bin/env python3
"""Validate saved route coordinates, source fingerprints and rendered frames."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import base64

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vst_global_planner.terrain import load_backend
from vst_global_planner.supervision import write_json_utf8


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    args=p.parse_args()
    load_backend(ROOT.parents[1])
    from stair_nav.core.io import sha256
    from stair_nav.visualization.data import ensure_viewer_data
    scene=args.scene.resolve()
    ensure_viewer_data(scene)
    bundle=json.loads((scene/"viewer_data.json").read_text())
    records=json.loads((scene/"routes.report.json").read_text())["queries"]
    by_id={r["id"]:r for r in records if r["optimized"]}
    checked=[]
    for route in bundle["routes"]:
        r=route["report"]
        record=by_id[r["query"]]
        path=np.load(scene/"paths"/(r["query"]+".gpmp.npy"))
        bounds=np.asarray(record["planning_bounds_xy"])
        assert ((path[:,:2]>=bounds[:2]-1e-7)&(path[:,:2]<=bounds[2:]+1e-7)).all()
        assert record["geometry_validation"]["passed"]
        assert len(route["frames"])==r["frame_count"]
        for frame in (route["frames"][0],route["frames"][-1]):
            assert np.isfinite(frame["pose"]).all()
            for stream in ("rgb","depth"):
                with Image.open(BytesIO(base64.b64decode(frame[stream].split(",",1)[1]))) as image:
                    image.verify()
        for file,digest in r["source_fingerprints"].items():
            assert sha256(Path(file))==digest
        checked.append(dict(query=r["query"],region=r["region_id"],length_m=r["reference_length_m"],
                            frames=len(route["frames"]),renderer=r["renderer"]["backend"]))
    report=dict(passed=True,routes=len(checked),regions=len({r["region"] for r in checked}),
        total_frames=sum(r["frames"] for r in checked),routes_checked=checked,
        scope="Native GPMP, offline sampled mesh checks, image integrity and source immutability. No student or dynamics execution.")
    write_json_utf8(scene/"route_bundle_validation.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="routes_checked"}))


if __name__=="__main__": main()
