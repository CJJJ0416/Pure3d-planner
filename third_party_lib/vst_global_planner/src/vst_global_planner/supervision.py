"""Full-scene VST labels plus independent world-aligned teacher planning crops."""
import json
import os
from pathlib import Path
import pickle

import numpy as np

from .terrain import load_backend


def write_json_utf8(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def build_supervision(output,backend_root):
    output=Path(output).resolve()
    load_backend(backend_root)
    from stair_nav.core.io import read_json,sha256
    from stair_nav.planning.vst import build_tomogram
    scene=read_json(output/"scene.json")
    if (output/"vst.pickle").exists():
        raise FileExistsError("Labels already exist. Generate a new scene directory for a new label version.")
    if sha256(output/"terrain.obj")!=scene["mesh_sha256"]:
        raise ValueError("Terrain mesh changed after point-cloud generation")
    root=Path(__file__).resolve().parents[2]
    os.environ.setdefault("CUDA_PATH","/usr/local/cuda")
    os.environ.setdefault("CUPY_CACHE_DIR",str(root/".cache/cupy"))
    os.environ.setdefault("MPLCONFIGDIR",str(root/".cache/matplotlib"))
    metadata=build_tomogram(output)
    with (output/"vst.pickle").open("rb") as stream:
        payload=pickle.load(stream)
    array=payload["data"]
    old_shape=np.array(array.shape[2:])
    center=np.asarray(payload["center"])
    res=float(payload["resolution"])
    crops=[]
    for name,region in scene["training_regions"].items():
        bounds=np.array(region["planning_bounds_xy"])
        first=np.maximum(0,np.ceil((bounds[:2]-center)/res+old_shape//2).astype(int))
        last=np.minimum(old_shape,np.floor((bounds[2:]-center)/res+old_shape//2).astype(int)+1)
        crop=array[:,:,first[0]:last[0],first[1]:last[1]].copy()
        shape=np.array(crop.shape[2:])
        if not np.all(shape>0):
            raise ValueError(f"Empty planning crop: {name}")
        crop_center=center+(first+shape//2-old_shape//2)*res
        local=dict(payload,data=crop,center=crop_center)
        folder=output/"regions"/name
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/"vst.pickle").open("wb") as stream:
            pickle.dump(local,stream,protocol=pickle.HIGHEST_PROTOCOL)
        valid=np.isfinite(crop[3])
        np.savez_compressed(folder/"tomography.npz",**local,valid_ground=valid)
        queries=[q for q in scene["queries"] if q["region_id"]==name]
        write_json_utf8(folder/"queries.json",dict(region_id=name,queries=queries,
            scope="Candidate endpoints only. Teacher routes have not been generated or executed."))
        crop_meta=dict(region_id=name,shape=list(crop.shape),bounds_xy=bounds.tolist(),
            source_vst_sha256=metadata["tomogram_sha256"],
            crop_vst_sha256=sha256(folder/"vst.pickle"),
            world_center=crop_center.tolist(),first_source_cell=first.tolist(),
            coordinate_equivalence_error_m=float(np.max(abs(
                crop_center-shape//2*res-(center+(first-old_shape//2)*res)))),
            planning_contract="Load this cropped VST for this region. Do not plan on the full shared VST.",
            sensor_contract="Always raycast the full terrain.obj. The planning crop is not sensor geometry.")
        write_json_utf8(folder/"planning_region.json",crop_meta)
        crops.append(crop_meta)
    valid=np.isfinite(array[3])
    summary=dict(shape=list(array.shape),channel_order=["cost","gradient_x","gradient_y","ground","ceiling"],
        valid_ground_cells_per_layer=valid.sum(axis=(1,2)).tolist(),
        barrier_cost=scene["config"]["vst"]["cost_barrier"],
        planner_threshold=scene["config"]["planner"]["cost_threshold"],
        raw_cost_range=[float(np.nanmin(array[0])),float(np.nanmax(array[0]))],
        planning_regions=crops,gpu=metadata["gpu"],
        note="Full-scene privileged labels and isolated teacher crops. No cost threshold filtering. LiDAR visibility masks and local-frame alignment remain dataset work.")
    write_json_utf8(output/"supervision_summary.json",summary)
    scene["tomography_status"]="generated"
    scene["tomography_source_mesh_sha256"]=scene["mesh_sha256"]
    write_json_utf8(output/"scene.json",scene)
    return summary

