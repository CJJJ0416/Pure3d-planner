"""一份建筑 PCD + 机器人条件 -> 原生归一化 Tomogram 代价点云。

教师侧流程：
1. 调用 VST 的 ConditionedTeacher 只建一次几何（ground/ceiling/切片）；
2. 对 robots.json 中的各组条件分别计算代价，并保存为 vst_<ID>.pickle；
3. 记录条件契约、参数与统计信息；
4. 调用 prepare.py 把 pickle 导出为 tomogram_<ID>.pcd。
教师条件顺序：[clearance_min, slope_max, step_max, interval_free, stand_ratio]。
"""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from common.pointcloud_io import read_pcd
from utils.pointcloud import CONDITION_NAMES, ROOT, read_json, write_json
from Traversability.data.prepare import export_tomograms

# 固定 CuPy / Matplotlib 缓存目录，避免写用户主目录
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "datasets/cache/cupy"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))
sys.path.insert(0, str(ROOT / "common/vst_global_planner/src"))
from vst_global_planner.conditioned_teacher import ConditionedTeacher, DEFAULT_TEACHER_CONFIG


def render_slices(folder, scene, teacher, profiles, pcd_name):
    """预览作为中间数据保留的原生 VST 代价切片。

    每个条件一列、每个楼层附近一行，灰色表示该切片没有有效 ground。
    图像只用于检查教师输出，原始的完整切片与代价仍保存在 pickle 中。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    floors = scene["config"]["floor_z_m"]
    fig, axes = plt.subplots(len(floors), len(profiles), figsize=(21, 4.2 * len(floors)),
                             squeeze=False, facecolor="#f7f9fc")
    # 绿 -> 黄 -> 红表示代价从低到高；无效区域设为浅灰
    cmap = LinearSegmentedColormap.from_list("vst_cost", ["#16a34a", "#fde047", "#ef4444"])
    cmap.set_bad("#e7ebf1")
    nx, ny = teacher.shape_xy
    resolution = teacher.resolution
    center = teacher.center
    # 将 VST 网格的整数索引范围换算成世界坐标范围
    extent = [center[0] + (-(nx // 2) - .5) * resolution,
              center[0] + (nx - nx // 2 - .5) * resolution,
              center[1] + (-(ny // 2) - .5) * resolution,
              center[1] + (ny - ny // 2 - .5) * resolution]
    for column, profile in enumerate(profiles):
        with (folder / profile["vst"]).open("rb") as stream:
            costs = pickle.load(stream)["data"][0]
        for row, z in enumerate(floors):
            # 选择覆盖该楼层高度的原生水平切片
            layer = int(np.ceil((z + .5 - teacher.slice_h0) / teacher.slice_dh))
            slice_z = teacher.slice_h0 + layer * teacher.slice_dh
            valid = np.isfinite(teacher.ground[layer])
            raw = costs[layer].astype(np.float32)
            axis = axes[row, column]
            # 代价除以 cost_scale 后映射到 [0,1]，并掩蔽无有效 ground 的单元
            image = axis.imshow(np.ma.array(raw.T / teacher.cost_scale, mask=~valid.T),
                                origin="lower", extent=extent, vmin=0, vmax=1, cmap=cmap)
            axis.set_aspect("equal")
            if row == 0:
                values = ", ".join(f"{v:g}" for v in profile["condition"])
                axis.set_title(f"{profile['id']}\n[{values}]", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"Floor {row} vicinity | slice z={slice_z:.2f} m\nWorld Y (m)")
            if row == len(floors) - 1:
                axis.set_xlabel("World X (m)")
    fig.suptitle(f"Robot-conditioned VST teachers | seed {scene['seed']}\n"
                 f"{pcd_name} | [clearance_min, slope_max, step_max, interval_free, stand_ratio]",
                 fontsize=18, y=.98)
    fig.subplots_adjust(left=.06, right=.92, top=.88, bottom=.08, wspace=.14, hspace=.23)
    bar = fig.colorbar(image, cax=fig.add_axes([.94, .2, .012, .56]))
    bar.set_label(f"Cost / {teacher.cost_scale:g}")
    fig.text(.49, .018, f"Native horizontal slices only. Grey: no finite ground. Pickles retain all slices and raw costs [0, {teacher.cost_scale:g}].",
             ha="center", color="#526277", fontsize=11)
    fig.savefig(folder / "teacher_overview.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    """构建一次 VST 几何，对各组机器人条件分别计算并保存教师代价。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "datasets/wfc_capability_dr_stand_v3/scene_20261109")
    parser.add_argument("--pcd", help="输入 PCD 文件名；默认使用教师配置中的 pcd.file_name。")
    parser.add_argument("--robots", type=Path, default=ROOT / "Traversability/configs/robots.json")
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--resolution", type=float, help="覆盖教师配置中的 XY 分辨率")
    parser.add_argument("--slice-dh", type=float, help="覆盖教师配置中的切片厚度")
    parser.add_argument("--ground-h", type=float, help="覆盖 ground 高度阈值")
    args = parser.parse_args()
    folder = args.scene.resolve()
    settings = read_json(args.teacher_config)
    pcd_path = folder / (args.pcd or settings["pcd"]["file_name"])
    coord, normal = read_pcd(pcd_path)
    del normal  # 教师只使用 XYZ，法线不参与建图
    scene = read_json(folder / "scene.json")
    robots = read_json(args.robots)
    # 深拷贝 profiles，避免把运行期字段写回 robots.json
    profiles = deepcopy(robots["profiles"])
    baseline = np.asarray(profiles[0]["condition"])
    # 按场景 seed 分层采样联合条件，训练/验证建筑使用各自的组合。
    count = robots["random_profiles"]
    rng = np.random.default_rng(scene["seed"])
    ranges = np.asarray(robots["random_ranges"])
    samples = np.column_stack([(rng.permutation(count) + rng.random(count)) / count for _ in range(4)])
    for index, sample in enumerate(samples):
        condition = baseline.copy()
        for column, value in zip((0, 1, 2, 4), sample):
            condition[column] = ranges[column, 0] + value * (ranges[column, 1] - ranges[column, 0])
        profiles.append(dict(id=f"joint_random_{index:02d}", condition=condition.tolist()))
    # 在同一个随机基准上，每次只改变一个学生参数，明确监督条件响应。
    anchors = profiles[-count:] if count else []
    for index, anchor in enumerate(anchors[:robots.get("paired_anchors", 0)]):
        for column in (0, 1, 2, 4):
            condition = np.asarray(anchor["condition"]).copy()
            lower, upper = ranges[column]
            unit = (condition[column] - lower) / (upper - lower)
            condition[column] = lower + ((unit + .5) % 1.) * (upper - lower)
            profiles.append(dict(id=f"paired_{index:02d}_{CONDITION_NAMES[column]}",
                                 condition=condition.tolist(), paired_with=anchor["id"],
                                 paired_parameter=CONDITION_NAMES[column]))
    started = time.perf_counter()
    # 几何只构建一次，之后每组条件只重新计算代价
    teacher = ConditionedTeacher(coord, resolution=args.resolution, slice_dh=args.slice_dh,
                                 ground_h=args.ground_h, teacher_config=args.teacher_config)
    geometry_s = time.perf_counter() - started
    baseline_parameters = teacher.parameters(baseline)
    baseline_parameters["pcd"] = dict(file_name=pcd_path.name)
    print(f"PCD={pcd_path.name} points={len(coord):,} ground_h={teacher.ground_h:g}", flush=True)
    print(f"Geometry built once: shape={teacher.ground.shape} time={geometry_s:.2f}s", flush=True)
    for profile in profiles:
        # 记录该 profile 相对 nominal 改变了哪些条件，便于解释差异
        changed = np.flatnonzero(np.asarray(profile["condition"]) != baseline)
        profile["changed_parameters"] = [CONDITION_NAMES[index] for index in changed]
        profile["vst"] = f"vst_{profile['id']}.pickle"
        profile["teacher_parameters"] = teacher.parameters(profile["condition"])
    # PCD 哈希用于确认教师与输入几何版本一致
    source_hash = hashlib.sha256(pcd_path.read_bytes()).hexdigest()
    # 条件契约固定维度顺序、单位、缩放与教师配置，供学生模型对齐
    contract = dict(
        condition_names=CONDITION_NAMES, condition_units=["m", "rad", "m", "m", "ratio"],
        condition_scale=robots["condition_scale"], baseline_id=profiles[0]["id"],
        profiles=profiles, pcd=pcd_path.name, pcd_sha256=source_hash, points=len(coord),
        map=baseline_parameters["map"], cost_scale=teacher.cost_scale,
        teacher_parameters=baseline_parameters, teacher_config=str(args.teacher_config.resolve()),
        fixed={key: value for key, value in teacher.trav.items()
               if key not in ["safe_margin", "slope_max", "step_max", "interval_free", "standable_ratio"]},
    )
    write_json(folder / "robot_conditions.json", contract)
    # 有效 ground 掩码在整个场景内固定，用于统计与差异比较
    valid = np.isfinite(teacher.ground)
    summary = []
    for profile in profiles:
        started = time.perf_counter()
        raw = teacher.cost(profile["condition"])
        teacher_s = time.perf_counter() - started
        # payload 保存完整切片、ground/ceiling 与原始代价（未归一化）
        payload = teacher.payload(profile["condition"], raw)
        payload.update(condition_names=CONDITION_NAMES, pcd=pcd_path.name, pcd_sha256=source_hash,
                       ground_h=teacher.ground_h, units="m", frame_id=scene["frame_id"])
        payload["teacher_parameters"]["pcd"] = dict(file_name=pcd_path.name)
        with (folder / profile["vst"]).open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        # 第一组条件作为 nominal，后续比较相对变化
        if profile["id"] == profiles[0]["id"]:
            nominal = raw.copy()
        difference = raw[valid] - nominal[valid]
        summary.append(dict(
            **profile, teacher_s=teacher_s, size_bytes=(folder / profile["vst"]).stat().st_size,
            finite_ground_cells=int(valid.sum()), cost_min=float(raw[valid].min()),
            cost_max=float(raw[valid].max()), cost_mean=float(raw[valid].mean()),
            barrier_fraction=float(np.mean(raw[valid] >= teacher.cost_scale - .001)),
            changed_vs_nominal_fraction=float(np.mean(np.abs(difference) > .001)),
            mean_absolute_cost_change=float(np.mean(np.abs(difference))),
        ))
        print(f"{profile['id']}: saved {profile['vst']} time={teacher_s:.2f}s "
              f"changed={summary[-1]['changed_vs_nominal_fraction']:.2%}", flush=True)
    # teacher.json 汇总教师元数据、各组统计与固定参数
    metadata = dict(**teacher.metadata(), condition_names=CONDITION_NAMES, pcd=pcd_path.name,
                    pcd_sha256=source_hash, points=len(coord), geometry_s=geometry_s, profiles=summary)
    metadata["overview_profiles"] = robots["overview_profiles"]
    metadata["teacher_parameters"] = baseline_parameters
    write_json(folder / "teacher.json", metadata)
    # 生成切片预览图，并更新 scene.json 的教师阶段信息
    render_slices(folder, scene, teacher,
                  [p for p in profiles if p["id"] in robots["overview_profiles"]], pcd_path.name)
    scene["pointcloud"]["active"] = pcd_path.name
    scene["teacher"] = dict(pcd=pcd_path.name, conditions="robot_conditions.json", metadata="teacher.json",
                            overview="teacher_overview.png", profiles=[p["vst"] for p in profiles])
    scene["stage"] = "robot-conditioned VST maps ready for native Tomogram export"
    write_json(folder / "scene.json", scene)
    # 直接从 pickle 导出全部归一化 Tomogram 代价点云
    export_tomograms(folder)
    print(f"Done: {folder / 'teacher.json'}", flush=True)


if __name__ == "__main__":
    main()
