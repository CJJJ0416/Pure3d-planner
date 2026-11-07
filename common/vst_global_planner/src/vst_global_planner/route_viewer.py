"""Adapt native GPMP references to the existing port-8769 sensor viewer."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .routes import LABELS
from .supervision import write_json_utf8
from .terrain import load_backend


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_scene(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    original=read_json(source/"scene.json")
    records=read_json(output/"routes.report.json")["queries"]
    good=[r for r in records if r["optimized"]]
    if not good: raise ValueError("No native GPMP routes were generated")
    for name in ("terrain.obj","terrain.pcd","vst.pickle","vst.meta.json","vst_debug",
                 "regions","geometry_paths.json","training_regions.json","config.resolved.json"):
        path=output/name
        if not path.exists(): path.symlink_to(source/name,target_is_directory=(source/name).is_dir())
    scene=deepcopy(original)
    scene.update(geometry_source=str(source),solids=[],stair_flights=[],
        viewer_geometry_note="Display uses the authoritative world mesh; construction-local analytic records are not used.",
        queries=[dict(id=r["id"],region_id=r["region_id"],start_support_xyz=r["start_support_xyz"],
                      goal_support_xyz=r["goal_support_xyz"],planning_bounds_xy=r["planning_bounds_xy"]) for r in good])
    write_json_utf8(output/"scene.json",scene)
    write_json_utf8(output/"paths.report.json",dict(backend="VST + native A* + native GPMP",
        optimized_count=len(good),queries=records,geometry_validation="offline sampled mesh checks",
        scope="Reference planning and sensor replay, no policy or dynamics rollout"))
    return scene,good


def add_route_metadata(route,scene,record):
    from stair_nav.core.io import sha256
    r=route["report"]
    name=record["region_id"]
    delta=record["goal_support_xyz"][2]-record["start_support_xyz"][2]
    direction="上行" if delta>.5 else "下行" if delta<-.5 else "反向穿行" if record["reverse"] else "正向穿行"
    r.update(region_id=name,region_label=LABELS[name],label=f"{LABELS[name]} · {direction} · {r['reference_length_m']:.1f} m",
        start_xyz=record["start_support_xyz"],goal_xyz=record["goal_support_xyz"],
        geometry_validation=record["geometry_validation"],cost_threshold=record["cost_threshold"],
        geometry_source=scene["geometry_source"],policy_training=False,
        planning_bounds_xy=record["planning_bounds_xy"],collision_validation="offline sampled mesh, not dynamics")
    route["figure"]["layout"]["uirevision"]=name
    x0,y0,x1,y1=record["planning_bounds_xy"]
    z1=float(max(record["start_support_xyz"][2],record["goal_support_xyz"][2],scene["height"])+1.2)
    s=route["figure"]["layout"]["scene"]
    s.update(aspectmode="manual",aspectratio=dict(x=1.,y=(y1-y0)/(x1-x0),z=(z1+.2)*1.7/(x1-x0)),
        xaxis=dict(s["xaxis"],range=[x0-1.,x1+1.]),yaxis=dict(s["yaxis"],range=[y0-1.,y1+1.]),
        zaxis=dict(s["zaxis"],range=[-.2,z1]))
    yaw=scene["training_regions"][name]["yaw_rad"]
    eye=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])@np.array([1.15,-1.35])
    s["camera"]=dict(eye=dict(x=float(eye[0]),y=float(eye[1]),z=1.))
    route["figure"]["data"][0]["opacity"]=.45
    return route


def build_viewer(source,output,backend_root,max_frames=None):
    load_backend(backend_root)
    scene,records=prepare_scene(source,output)
    from stair_nav.visualization.scene import build_scene_viewer
    from stair_nav.core.io import sha256
    output=Path(output).resolve()
    data_file=build_scene_viewer(output,max_frames=max_frames)
    bundle=read_json(data_file)
    by_id={r["id"]:r for r in records}
    for route in bundle["routes"]:
        add_route_metadata(route,scene,by_id[route["report"]["query"]])
    write_json_utf8(data_file,bundle)
    manifest=read_json(output/"viewer_manifest.json")
    manifest.update(data_sha256=sha256(data_file),playground=True,
                    scope="Native random GPMP references and GPU camera replay. No student-policy rollout.")
    write_json_utf8(output/"viewer_manifest.json",manifest)
    return manifest


def render_one(scene_dir,path_file,query,record,max_frames=None):
    from stair_nav.visualization.motion import create_playback
    scene_dir=Path(scene_dir).resolve()
    payload=create_playback(scene_dir,query,path_file=path_file,output=Path(path_file).parent/"replay",
                            export_artifacts=False,max_frames=max_frames)
    add_route_metadata(payload,read_json(scene_dir/"scene.json"),record)
    return {key:payload[key] for key in ("report","frames","figure")}

