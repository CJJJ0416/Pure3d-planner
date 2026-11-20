#!/usr/bin/env python3
"""Check planning crop identity and report raw VST probe-label coverage."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene",type=Path)
    args=parser.parse_args()
    folder=args.scene.resolve()
    with (folder/"vst.pickle").open("rb") as stream:
        full=pickle.load(stream)
    array=full["data"]
    shape=np.array(array.shape[2:])
    errors=[]
    for path in sorted((folder/"regions").iterdir()):
        meta=json.loads((path/"planning_region.json").read_text())
        with (path/"vst.pickle").open("rb") as stream:
            crop=pickle.load(stream)
        i,j=meta["first_source_cell"]
        h,w=crop["data"].shape[2:]
        if not np.array_equal(crop["data"],array[:,:,i:i+h,j:j+w],equal_nan=True):
            raise ValueError(f"Crop differs from source: {path.name}")
        if meta["coordinate_equivalence_error_m"]>=1e-12:
            raise ValueError(f"Crop shifted world coordinates: {path.name}")
        errors.append(meta["coordinate_equivalence_error_m"])
    paths=json.loads((folder/"geometry_paths.json").read_text())["paths"]
    quality={}
    for name,points in paths.items():
        points=np.asarray(points)
        ij=np.round((points[:,:2]-full["center"])/full["resolution"]).astype(int)+shape//2
        ground=array[3,:,ij[:,0],ij[:,1]]
        cost=array[0,:,ij[:,0],ij[:,1]]
        delta=np.where(np.isfinite(ground),abs(ground-points[:,2,None]),np.inf)
        candidate=delta<=full["slice_dh"]
        mincost=np.min(np.where(candidate,cost,np.inf),axis=1)
        quality[name]=dict(samples=len(points),height_candidate_fraction=float(candidate.any(axis=1).mean()),
            cost_at_most_10_fraction=float((mincost<=10).mean()),
            max_nearest_support_height_error_m=float(np.min(delta,axis=1).max()))
    report=dict(crop_count=len(errors),arrays_equal_to_source=True,max_coordinate_error_m=max(errors),
        probe_label_quality=quality,
        scope="Label/crop diagnostics. Probe sequences are not planned robot routes and may include high-cost cells.")
    (folder/"label_validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(crops=len(errors),arrays_equal=True,max_coordinate_error_m=max(errors))))


if __name__=="__main__":
    main()
