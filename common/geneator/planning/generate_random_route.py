#!/usr/bin/env python3
"""在已有 VST 地图中随机选点，生成原生 GPMP 路线及相机回放。"""
import argparse
import os
from pathlib import Path
import pickle
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src")]

import numpy as np
from stair_nav.core.io import read_json, sha256, write_json
from stair_nav.scenes.geometry import to_local
from stair_nav.planning.vst import (check_sources, endpoint_index, make_planner,
                           optimized_world, planner_config, save_path)


def floor_candidates(scene, data, center, resolution):
    """按实际建筑楼面从 VST 中筛选可采样的支撑点，返回 floor_id→Nx3 世界坐标。

    地面高度有限且代价不超过规划阈值才有效；同一 XY 的重复切片保留最低代价表示。
    只接收接近该层标称高度的点，并排除建筑轮廓外和实际洞口内的点。
    切片编号不等于楼层编号，因此不能直接按 data 的层索引对随机端点分楼。
    """
    ground, cost = data[3], data[0]
    valid = np.isfinite(ground) & np.isfinite(cost) & (cost >= 0)
    threshold = float(scene["config"].get("planner", {}).get("cost_threshold", planner_config().cost_threshold))
    valid &= cost <= threshold
    floors = scene.get("floors") or scene["floor_graph"]["nodes"]
    groups = {}
    for floor in floors:
        cells = np.argwhere(valid & (np.abs(ground-float(floor["z"])) <= .2))
        if not len(cells):
            continue
        # 同一地面可能出现在相邻 VST 切片中，每个 XY 仅保留
        # 代价最低的表示，与端点选层规则一致。
        order = np.argsort(cost[tuple(cells.T)], kind="stable")
        cells = cells[order]
        _, unique = np.unique(cells[:, 1:], axis=0, return_index=True)
        cells = cells[unique]
        xyz = np.column_stack([
            (cells[:, 1]-data.shape[2]//2)*resolution+center[0],
            (cells[:, 2]-data.shape[3]//2)*resolution+center[1],
            ground[tuple(cells.T)]])
        if "regions" in floor:
            local = to_local(xyz, scene)
            supported = np.zeros(len(local), dtype=bool)
            for x0, y0, x1, y1 in floor["regions"]:
                supported |= ((local[:, 0] >= x0) & (local[:, 0] <= x1)
                              & (local[:, 1] >= y0) & (local[:, 1] <= y1))
            for x0, y0, x1, y1 in floor["openings"]:
                supported &= ~((local[:, 0] > x0) & (local[:, 0] < x1)
                               & (local[:, 1] > y0) & (local[:, 1] < y1))
            xyz = xyz[supported]
        if not len(xyz):
            continue
        groups[int(floor["id"])] = xyz
    return groups


def sample_pair(groups, rng):
    """从候选楼层各抽一个端点，返回起终层号和两点世界 XYZ。

    有多个可用楼层时无放回选层，优先跨层；只有一层时允许同层抽样。
    本步骤不证明两点连通，后续仍需 A* 与 GPMP 实际生成路线。
    """
    ids = list(groups)
    if not ids:
        raise ValueError("没有可用于随机选点的 VST 地面")
    a, b = rng.choice(ids, size=2, replace=len(ids) < 2)
    start = groups[a][rng.integers(len(groups[a]))]
    goal = groups[b][rng.integers(len(groups[b]))]
    return int(a), int(b), start, goal


def generate(scene_dir, output, seed=None, max_attempts=30, max_frames=None, job_id=None):
    """在已有 VST 中随机选点并生成一条可替换路线，返回请求记录。

    先核对点云/地图/配置/源码指纹，再最多尝试 max_attempts 组端点；A* 需到达目标
    XY，GPMP 输出还需核对两端 XY 与高度，防止把重叠楼层当成同一终点。
    原生曲线、request.json、route.json 和 status.json 仅写入 output；原场景保持只读。
    路线成功后调用共享相机生成器，max_frames 仅作显式快速采样上限。
    此脚本写 ready 表示产物已写完，服务端仍要等待进程退出再安装和发布最终 ready。
    """
    scene_dir, output = Path(scene_dir).resolve(), Path(output).resolve()
    job_id = job_id or (output.name if re.fullmatch(r"[0-9a-f]{32}", output.name) else uuid.uuid4().hex)
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid random route job ID")
    output.mkdir(parents=True, exist_ok=True)
    status_file = output/"status.json"

    def status(state, message, **extra):
        write_json(status_file, dict(state=state, message=message, **extra))

    status("planning", "正在读取地图并选择不同楼层的起终点…")
    scene = read_json(scene_dir/"scene.json")
    meta = read_json(scene_dir/"vst.meta.json")
    inputs = [scene_dir/name for name in
              ("scene.json", "terrain.obj", "terrain.pcd", "vst.pickle", "vst.meta.json")]
    fingerprints = {str(p): sha256(p) for p in inputs}
    if (fingerprints[str(scene_dir/"terrain.pcd")] != meta["pcd_sha256"]
            or fingerprints[str(scene_dir/"vst.pickle")] != meta["tomogram_sha256"]
            or meta["config"] != scene["config"]["vst"]):
        raise ValueError("场景与 VST 地图不一致，请重新生成地图")
    provenance = check_sources()
    if meta["source"]["files"] != provenance["files"]:
        raise ValueError("VST 地图与当前规划器版本不一致")
    with (scene_dir/"vst.pickle").open("rb") as stream:
        payload = pickle.load(stream)
    data = np.asarray(payload["data"], dtype=np.float64)
    center, resolution = np.asarray(payload["center"]), float(payload["resolution"])
    groups = floor_candidates(scene, data, center, resolution)
    threshold = float(scene["config"].get("planner", {}).get("cost_threshold", planner_config().cost_threshold))
    rng = np.random.default_rng(seed)
    failures = []
    original_cwd = Path.cwd()
    try:
        # 原生规划器使用固定日志名，需在任务目录内运行，避免覆盖已有日志。
        os.chdir(output)
        planner = make_planner(data, resolution, cost_barrier=meta["config"]["cost_barrier"], cost_threshold=threshold)
        for attempt in range(1, max_attempts+1):
            a, b, start_xyz, goal_xyz = sample_pair(groups, rng)
            if np.linalg.norm(start_xyz[:2]-goal_xyz[:2]) < 1.5:
                continue
            status("planning", f"正在规划第 {attempt} 组随机点：{a+1}F → {b+1}F…")
            try:
                start = endpoint_index(start_xyz, data, center, resolution, payload["slice_dh"], threshold)
                goal = endpoint_index(goal_xyz, data, center, resolution, payload["slice_dh"], threshold)
                if not planner.plan(start, goal, False):
                    raise RuntimeError("A* 未找到连通路线")
                indices = np.asarray(planner.get_path_finder().get_result_matrix(), dtype=int)
                if not len(indices) or not np.array_equal(indices[-1, 1:], goal[[2, 1]]):
                    raise RuntimeError("A* 未到达终点")
                if not planner.plan(start, goal, True):
                    raise RuntimeError("GPMP 优化失败")
                path = optimized_world(planner, data, center, resolution)
                if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
                        or not np.isfinite(path).all()):
                    raise ValueError("GPMP 未产生有效坐标")
                # 同时核对支撑高度与 XY 到达情况，避免将同 XY 的其他楼层当作终点。
                if (np.linalg.norm(path[0, :2]-start_xyz[:2]) > .3
                        or np.linalg.norm(path[-1, :2]-goal_xyz[:2]) > .3
                        or abs(path[0, 2]-start_xyz[2]) > .5
                        or abs(path[-1, 2]-goal_xyz[2]) > .5):
                    raise ValueError("GPMP 未连接所选起终点")
                save_path(path, output/"random.gpmp")
                break
            except (ValueError, RuntimeError, IndexError) as exc:
                failures.append(dict(attempt=attempt, error=str(exc)))
        else:
            raise RuntimeError(f"尝试 {max_attempts} 组点后仍未生成路线，请再次随机")
    finally:
        os.chdir(original_cwd)
    status("rendering", f"已生成 {a+1}F → {b+1}F 路线，正在渲染 RGB 和深度回放…")
    from stair_nav.visualization.motion import create_playback
    camera = scene_dir/"camera_profile.json"
    replay = create_playback(scene_dir, "random_"+job_id, output/"replay",
                             max_frames=max_frames, camera_config=camera if camera.exists() else None,
                             path_file=output/"random.gpmp.csv", export_artifacts=False)
    record = dict(start_floor=a, goal_floor=b, different_floors=a != b,
                  cost_threshold=threshold, cost_barrier=float(meta["config"]["cost_barrier"]),
                  start_xyz=start_xyz.tolist(), goal_xyz=goal_xyz.tolist(),
                  start_index=start.tolist(), goal_index=goal.tolist(), attempts=attempt,
                  candidate_counts={str(k): len(v) for k, v in groups.items()}, failures=failures,
                  source_commit=provenance["commit"], source_fingerprints=fingerprints,
                  success_criterion="gpmp_output", geometry_validation="disabled")
    replay["report"].update(record, label=f"当前随机路线 {a+1}F → {b+1}F", storage="replace_current")
    for file, digest in fingerprints.items():
        if sha256(Path(file)) != digest:
            raise RuntimeError(f"生成期间源场景发生变化：{file}")
    write_json(output/"request.json", record)
    write_json(output/"route.json", {k: replay[k] for k in ("report", "frames", "figure")})
    status("ready", f"随机路线已生成：{a+1}F → {b+1}F · {replay['report']['reference_length_m']:.1f} m")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    try:
        generate(args.scene, args.output, args.seed, job_id=args.job_id)
    except Exception as exc:
        write_json(args.output/"status.json", dict(state="error", message=str(exc)))
        raise
