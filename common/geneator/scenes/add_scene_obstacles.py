#!/usr/bin/env python3
"""为已有场景添加配置中的实体障碍物，并生成对应的新场景产物。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache/matplotlib"))
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT/".cache/cupy"))
os.environ["PYTHONPATH"] = str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH", "")

import numpy as np
import trimesh
from stair_nav.scenes.geometry import rotation, sample_surface, support_heights, to_world
from stair_nav.core.io import read_json, sha256, write_json, write_pcd

COLORS = dict(crate=[202, 99, 75], pillar=[48, 151, 164], low_barrier=[177, 119, 183])


def obstacle_geometry(spec, scene):
    """由单个障碍物配置生成 (局部实体记录,世界坐标闭合网格)。

    spec 给出所属楼层、局部中心、尺寸/半径、旋转角和高度；box 或多边形近似圆柱
    的底部依据解析支撑贴合楼面/坡台。拒绝跨出支撑或占用洞口/桥通行区域的摆放。
    返回记录中的 vertices_xy/上下平面保持局部坐标，网格才应用建筑 yaw。
    """
    center = np.asarray(spec["center_xy"], float)
    height = float(spec["height_m"])
    floor = next(f for f in scene["floors"] if f["id"] == spec["floor"])
    if height <= 0 or center.shape != (2,):
        raise ValueError(f"Invalid obstacle dimensions: {spec['id']}")
    if spec["shape"] == "box":
        size = np.asarray(spec["size_xy"], float)
        if size.shape != (2,) or (size <= 0).any():
            raise ValueError(f"Invalid obstacle footprint: {spec['id']}")
        polygon = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])*size/2
        polygon = polygon @ rotation(np.deg2rad(spec.get("yaw_deg", 0)))[:2,:2].T+center
    elif spec["shape"] == "cylinder":
        radius = float(spec["radius_m"])
        if radius <= 0:
            raise ValueError(f"Invalid obstacle radius: {spec['id']}")
        theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
        polygon = center+radius*np.column_stack([np.cos(theta), np.sin(theta)])
    else:
        raise ValueError(f"Unsupported obstacle shape: {spec['shape']}")
    # 避开洞口、栏杆和桥面，保留这些区域的通行空间。
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    for feature in scene.get("floor_features", []):
        if feature["floor"] != spec["floor"]:
            continue
        a, b, c, d = feature["bounds"]
        if low[0] < c+.2 and high[0] > a-.2 and low[1] < d+.2 and high[1] > b-.2:
            raise ValueError(f"Obstacle {spec['id']} overlaps a floor opening or bridge")
    # 障碍物底座需贴合已有解析楼面及局部坡台。
    # 此处只约束摆放，不代替路线碰撞验收。
    heights = []
    for x, y in np.vstack([polygon, (polygon+np.roll(polygon,-1,axis=0))/2, center]):
        candidates = [z for z in support_heights(x, y, scene) if abs(z-floor["z"]) < .4]
        if not candidates:
            raise ValueError(f"Obstacle {spec['id']} footprint extends beyond its floor support")
        heights.append(min(candidates, key=lambda z: abs(z-floor["z"])))
    bottom, top = min(heights)-.02, max(heights)+height
    vertices = np.vstack([np.column_stack([polygon, np.full(len(polygon), bottom)]),
                          np.column_stack([polygon, np.full(len(polygon), top)])])
    mesh = trimesh.convex.convex_hull(vertices)
    item = dict(deepcopy(spec), kind="obstacle", color_rgb=COLORS[spec["category"]],
                vertices_xy=polygon.tolist(), bottom_plane=[0.,0.,bottom], top_plane=[0.,0.,top],
                lo=vertices.min(axis=0).tolist(), hi=vertices.max(axis=0).tolist(),
                support_z_range_m=[min(heights), max(heights)])
    mesh.vertices = to_world(mesh.vertices, scene)
    return item, mesh


def main():
    """读取源场景和障碍物配置，在新目录导出几何并按需重新建图/规划。

    保存来源指纹和旧 VST 调试图，不就地修改源目录。障碍物摆放约束不等于
    对所有重新生成路线完成了独立支撑、碰撞或动力学验收。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"outputs/upstream_f5580008/easy_00043_edit01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/obstacles_easy43.json")
    parser.add_argument("--geometry-only", action="store_true", help="Save geometry and previous debug only; no new map or paths")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        parser.error("Choose a new output directory outside the source scene")
    fingerprints = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    source_scene = read_json(source/"scene.json")
    scene, obstacle_config = deepcopy(source_scene), read_json(args.config)
    if obstacle_config["coordinate_frame"] != "building_local_xy_m":
        raise ValueError("Obstacle configuration must use building-local XY in meters")
    specs = obstacle_config["obstacles"]
    if not specs or len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Provide obstacles with unique IDs")
    items, meshes = [], [trimesh.load(source/"terrain.obj", force="mesh", process=False)]
    for spec in specs:
        item, mesh = obstacle_geometry(spec, source_scene)
        items.append(item); meshes.append(mesh)
    mesh = trimesh.boolean.union(meshes, engine="manifold")
    if not mesh.is_volume or not mesh.is_watertight:
        raise ValueError("Obstacle construction did not produce a closed positive-volume mesh")
    scene.update(name=output.name, source_scene=str(source), obstacles=scene.get("obstacles", [])+items,
                 bounds_world=mesh.bounds.tolist(), mesh_watertight=True)
    scene["solids"].extend(items)
    scene.setdefault("modifications", {})["added_obstacles"] = dict(
        config=str(args.config.resolve()), config_sha256=sha256(args.config), count=len(items),
        unchanged="Source floors, stairs, ramps, openings, building yaw and query endpoints")
    scene["config"].update(name=scene["name"], source_scene=str(source), modifications=scene["modifications"],
                           obstacle_layout=obstacle_config)
    output.mkdir(parents=True, exist_ok=False)
    if (source/"vst_debug").exists():
        shutil.copytree(source/"vst_debug", output/"vst_debug/baseline")
    mesh.export(output/"terrain.obj")
    points = sample_surface(mesh, scene["config"]["sample_spacing"])
    points = points[points[:,2] >= -1e-6]
    write_pcd(output/"terrain.pcd", points)
    scene.update(point_count=len(points), pcd_sha256=sha256(output/"terrain.pcd"))
    write_json(output/"scene.json", scene)
    write_json(output/"config.resolved.json", scene["config"])
    write_json(output/"obstacles.json", dict(config=obstacle_config, resolved=items))
    write_json(output/"source_snapshot.json", dict(source=str(source), sha256=fingerprints))
    status = dict(stage="obstacles_added", seed=scene["seed"], source=str(source), obstacle_count=len(items),
                  geometry_validation="disabled", success_criterion="gpmp_output")
    write_json(output/"run.json", status)
    if not args.geometry_only:
        from stair_nav.app.cli import worker
        try:
            print("Building new VST and native GPMP references for the obstacle scene", flush=True)
            worker("tomography", output); worker("plan", output)
            report = read_json(output/"paths.report.json")
            status.update(stage="complete", optimized_count=report["optimized_count"],
                          query_count=len(report["queries"]), accepted=report["optimized_count"]==len(report["queries"]))
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            status.update(stage="generation_failed", accepted=False, error=str(exc))
    from modify_easy_scene import preview
    preview(source, output, scene, mesh)
    if not args.geometry_only and (output/"paths.report.json").exists():
        from stair_nav.visualization.scene import build_scene_viewer
        if any(q.get("optimized") for q in read_json(output/"paths.report.json")["queries"]):
            build_scene_viewer(output)
            status["viewer"] = "viewer_data.json"
    for name, digest in fingerprints.items():
        if sha256(source/name) != digest:
            raise RuntimeError(f"Source artifact changed: {name}")
    status.update(source_unchanged=True, preview="preview.png",
                  baseline_debug_files=len(list((output/"vst_debug/baseline").glob("w_local_map*_binary.jpg"))),
                  current_debug_files=len(list((output/"vst_debug").glob("w_local_map*_binary.jpg"))))
    write_json(output/"run.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 2 if status.get("accepted") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
