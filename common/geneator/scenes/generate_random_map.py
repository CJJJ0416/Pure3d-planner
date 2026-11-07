#!/usr/bin/env python3
"""按新 seed 生成建筑、CUDA VST、规划路线与相机预览。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import secrets
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src")]
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT/".cache/cupy"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache/matplotlib"))
os.environ["PYTHONPATH"] = str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH", "")

import numpy as np
import trimesh
from stair_nav.scenes.generate import create_scene
from stair_nav.scenes.geometry import sample_surface, to_local
from stair_nav.core.io import read_json, sha256, write_json, write_pcd
from add_scene_obstacles import obstacle_geometry


def random_geometry(template, output, seed, avoid_family=None):
    """按 seed 重新生成真实建筑与障碍物，并写入 output 的几何产物。

    使用 random_building 配置，继承模板 VST 参数，可避开上一建筑形态。
    建筑和障碍物使用分离随机序列；障碍物避开查询端点、洞口/桥及已有障碍物。
    布尔合并后导出闭合网格、采样点云、scene/config/obstacles JSON，返回 scene。
    本函数只生成几何，不沿用模板的旧点云、VST 或曲线。
    """
    source = read_json(template/"scene.json")
    config = read_json(ROOT/"configs/random_building.json")
    config.update(name="random_map_current", seed=seed)
    if avoid_family:
        config["avoid_family"] = avoid_family
    config["vst"] = deepcopy(source["config"]["vst"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7821]))
    scene, mesh = create_scene(config)
    # 网页启用裸桥后，每张图至少要有一座实际必经桥；先核对再进行昂贵的建图/回放。
    features = scene.get("floor_feature_options", {})
    needs_bridge = (features.get("enabled") and features.get("bridges_per_floor", 1) > 0
                    and features.get("bridge_exposure", {}).get("probability", 0) > 0)
    if needs_bridge and not any(f.get("mandatory_passage") for f in scene.get("floor_features", [])):
        raise ValueError("本次地图缺少必经无栏杆窄桥，请检查生成配置")
    scene.update(name="random_map_current", source_scene=str(template), obstacles=[])
    scene["config"].update(name=scene["name"], source_scene=str(template))
    query_xy = to_local(np.array([q[f"{role}_support_xyz"] for q in scene["queries"]
                                 for role in ("start", "goal")]), scene)[:, :2]
    meshes, occupied = [mesh], []
    for floor in scene["floors"]:
        rooms = [room for room in scene["rooms"] if room["floor"] == floor["id"]]
        limits = config["random_obstacles_per_floor"]
        count = int(rng.integers(limits[0], limits[1]+1))
        placed = 0
        for _ in range(1000):
            if placed == count:
                break
            room = rooms[int(rng.integers(len(rooms)))]
            x0, y0, x1, y1 = room["bounds"]
            center = rng.uniform([x0+.85, y0+.85], [x1-.85, y1-.85])
            radius = float(rng.uniform(.24, .38))
            if np.min(np.linalg.norm(query_xy-center, axis=1)) < 1.0+radius:
                continue
            if any(f == floor["id"] and np.linalg.norm(center-xy) < r+radius+.7
                   for f, xy, r in occupied):
                continue
            shape = rng.choice(["box", "cylinder", "barrier"], p=[.45, .35, .20])
            spec = dict(id=f"F{floor['id']+1}_{placed+1}", floor=floor["id"],
                        center_xy=center.tolist(), height_m=float(rng.uniform(.65, 1.1)))
            if shape == "cylinder":
                spec.update(shape="cylinder", category="pillar", radius_m=radius)
            else:
                spec.update(shape="box", category="low_barrier" if shape == "barrier" else "crate",
                            size_xy=[float(rng.uniform(.45, .85)), float(rng.uniform(.35, .65))],
                            yaw_deg=float(rng.uniform(-60, 60)))
                if shape == "barrier":
                    spec["height_m"] = float(rng.uniform(.18, .3))
            try:
                item, obstacle = obstacle_geometry(spec, scene)
            except ValueError:
                continue
            scene["solids"].append(item)
            scene["obstacles"].append(item)
            meshes.append(obstacle)
            occupied.append((floor["id"], center, .5))
            placed += 1
        if placed != count:
            raise RuntimeError("本次布局无法放置所有障碍物，请重新随机")
    mesh = trimesh.boolean.union(meshes, engine="manifold")
    if not mesh.is_volume or not mesh.is_watertight:
        raise ValueError("随机地图未形成闭合几何体")
    scene.update(bounds_world=mesh.bounds.tolist(), mesh_watertight=True, pcd_ground_clip_z=0.0)
    scene["config"]["random_map"] = dict(seed=seed, architecture=scene["architecture"],
                                         obstacle_count=len(scene["obstacles"]))
    mesh.export(output/"terrain.obj")
    points = sample_surface(mesh, config["sample_spacing"])
    points = points[points[:, 2] >= -1e-6]
    write_pcd(output/"terrain.pcd", points)
    scene.update(point_count=len(points), pcd_sha256=sha256(output/"terrain.pcd"))
    write_json(output/"scene.json", scene)
    write_json(output/"config.resolved.json", scene["config"])
    write_json(output/"obstacles.json", dict(resolved=scene["obstacles"]))
    return scene


def generate(template, output, published, job_id, seed=None):
    """在任务目录完成新几何、GPU VST、所有固定 GPMP 路线和相机回放。

    template 为只读模板，output 为当前临时任务目录，published 是将来固定 current
    的位置，用于提前重写报告引用而非直接在该目录生成。最多尝试三组布局，必须
    全部固定路线产出成功后才渲染回放，结束时核对模板指纹并写 random_map.json。
    目录安装由上层管理器在进程退出后执行；生成失败不会主动覆盖正在服务的地图。
    """
    template, output, published = Path(template).resolve(), Path(output).resolve(), Path(published).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def progress(message, **extra):
        write_json(output/"status.json", dict(state="generating", message=message, **extra))

    # 新几何必须重新建图；CUDA 不可用时明确失败，不能混用旧 VST。
    progress("正在检查 GPU 并准备随机地图…")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("随机地图需要可用 NVIDIA GPU 来重新生成 VST；原地图已保留")
    inputs = [template/name for name in ("scene.json", "config.resolved.json", "camera_profile.json")]
    fingerprints = {str(p): sha256(p) for p in inputs if p.exists()}
    failures = []
    previous_family = None
    if (published/"scene.json").is_file():
        previous_family = read_json(published/"scene.json").get("architecture", {}).get("family")
    from stair_nav.app.cli import worker
    for attempt in range(1, 4):
        current_seed = (seed+attempt-1) if seed is not None else secrets.randbits(31)
        # 重试前只保留任务日志和进度文件，清除未完成的场景产物。
        for path in output.iterdir():
            if path.name in ("generation.log", "status.json"):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        try:
            progress(f"正在重新生成建筑架构和障碍物（第 {attempt} 次，seed {current_seed}）…")
            scene = random_geometry(template, output, current_seed, avoid_family=previous_family)
            progress("正在使用 GPU 重新生成 VST 分层地图…", seed=current_seed)
            worker("tomography", output)
            progress(f"正在生成新地图的 {len(scene['queries'])} 条 GPMP 路线…", seed=current_seed)
            worker("plan", output)
            report = read_json(output/"paths.report.json")
            if report["optimized_count"] != len(scene["queries"]):
                raise RuntimeError("部分路线未生成，正在重选地图布局")
            break
        except (ValueError, RuntimeError) as exc:
            failures.append(dict(seed=current_seed, error=str(exc)))
    else:
        raise RuntimeError("连续三次未生成完整地图和路线，请重新随机；上一张地图已保留")
    progress("正在渲染新地图的 RGB、深度回放和 VST Debug…", seed=current_seed)
    from stair_nav.visualization.scene import build_scene_viewer
    camera = template/"camera_profile.json"
    build_scene_viewer(output, camera_config=camera if camera.exists() else None,
                       compact=True, published_directory=published)
    for file, digest in fingerprints.items():
        if sha256(Path(file)) != digest:
            raise RuntimeError("生成过程中原地图配置发生变化，请重新尝试")
    write_json(output/"random_map.json", dict(job_id=job_id, seed=current_seed, attempts=attempt,
        failures=failures, source_scene=str(template), source_fingerprints=fingerprints,
        scene_sha256=sha256(output/"scene.json"), pcd_sha256=scene["pcd_sha256"],
        width_m=scene["width"], depth_m=scene["depth"], height_m=scene["height"],
        architecture=scene["architecture"], room_count=len(scene["rooms"]),
        doorway_count=len(scene["doorways"]), stair_count=len(scene["stair_flights"]),
        mandatory_bridge_count=sum(bool(f.get("mandatory_passage")) for f in scene.get("floor_features", [])),
        obstacle_count=len(scene["obstacles"]), storage="replace_current",
        success_criterion="gpmp_output", geometry_validation="disabled"))
    write_json(output/"status.json", dict(state="ready", message=f"新随机地图已生成 · seed {current_seed}", seed=current_seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        generate(args.template, args.output, args.published, args.job_id, args.seed)
    except Exception as exc:
        write_json(args.output/"status.json", dict(state="error", message=str(exc)))
        raise
