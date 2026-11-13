"""批量执行 WFC -> PCD -> 原生 VST/Tomogram -> 几何/查询空间分块。

每个 seed 依次运行：多层 WFC 场景网格生成、网格转完整 PCD、5 cm 体素降采样、
多组机器人条件的 VST 教师与 Tomogram 导出，最后统一生成独立分块。
整个流程 headless 顺序执行，不进行任何网络训练。
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.pointcloud import ROOT, CONDITION_NAMES, read_json, workspace_path, write_json


def run_stage(command, log_path):
    """以子进程运行一个生成阶段，实时写日志并返回耗时（秒）。"""
    started = time.perf_counter()
    with log_path.open("w") as log:
        # 统一使用当前 Python 解释器，保证与调用方环境一致
        process = subprocess.Popen([sys.executable, *map(str, command)], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            # 同时写入日志文件与终端，便于长任务观察进度
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        # 任一阶段失败立即抛出异常，避免生成不完整的场景
        raise subprocess.CalledProcessError(status, process.args)
    return time.perf_counter() - started


def collect_scene(folder, batch, timings):
    """汇总单个场景的几何、教师与质量检查信息，形成 manifest 行。"""
    scene = read_json(folder / "scene.json")
    teacher = read_json(folder / "teacher.json")
    pcd = read_json(folder / "pointcloud.report.json")
    downsampled = read_json(folder / "terrain_downsampled.report.json")
    landings = read_json(folder / "landings.report.json")["ports"]
    # 落地平台高度误差、楼梯井重叠单元等用于检查场景几何质量
    errors = [error for port in landings for error in port["height_error_m"]]
    occupied = [set() for _ in scene["floors"]]
    overlap_cells = 0
    for stair in scene["stairs"]:
        # 统计楼梯在楼层平面上的占用格，检测不同楼梯是否重叠
        cells = {(r, c) for r in range(*stair["bay_rows"]) for c in range(*stair["bay_columns"])}
        for floor in (stair["from_floor"], stair["to_floor"]):
            overlap_cells += len(occupied[floor] & cells)
            occupied[floor] |= cells
    # 检查楼梯两端 landing 是否落在楼梯井的占用区域内
    ports_in_wells = sum(tuple(s[side + "_landing_cell"]) in occupied[s[key]]
                         for s in scene["stairs"]
                         for side, key in [("lower", "from_floor"), ("upper", "to_floor")])
    return dict(
        id=folder.name, seed=scene["seed"], batch=batch["id"], split=batch["split"],
        folder=folder.name, overview=f"{folder.name}/overview.png", footprint=scene["footprint"],
        floors=len(scene["floors"]), connections=len(scene["stairs"]),
        connection_kinds=sorted({s["kind"] for s in scene["stairs"]}),
        railings={"with":sum(s["railings"] for s in scene["stairs"]),
                  "without":sum(not s["railings"] for s in scene["stairs"])},
        threshold_features=len(scene["capability_features"]), mesh_faces=scene["mesh_faces"],
        points=pcd["points"], input_points=downsampled["output_points"],
        pcd=f"{folder.name}/terrain_downsampled.pcd", pcd_sha256=teacher["pcd_sha256"],
        mesh_coverage_p99_m=pcd["mesh_to_pcd"]["p99_m"],
        landing_probes=len(errors), landing_hits=sum(p["hits"] for p in landings),
        landing_max_height_error_m=max(errors),
        well_overlap_cells=overlap_cells, landing_ports_in_wells=ports_in_wells,
        voxel_coverage_p99_m=downsampled["mesh_to_pcd"]["p99_m"],
        voxel_railing_coverage_p99_max_m=max(
            p["railing_to_pcd"]["p99_m"] for p in downsampled["connections"] if p["railings"]),
        # 各组 profile 的机器人条件、VST pickle 与 Tomogram 点云文件名
        profiles=[dict(id=p["id"], condition=p["condition"], vst=p["vst"],
                       tomogram=p["tomogram"], tomogram_points=p["tomogram_points"])
                  for p in teacher["profiles"]],
        timings_s=timings, stage="native Tomogram targets ready, no neural training",
    )


def render_contact_sheet(output, scenes):
    """用每个 seed 的实际总览图拼成合集，并附中层布局示意图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = (len(scenes) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.6 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, scene in zip(axes.flat, scenes):
        with Image.open(output / scene["overview"]) as source:
            w, h = source.size
            # 合集只裁剪总览图的中部；每个场景自己的 overview.png 保留完整视图。
            axis.imshow(source.crop((0, int(h * .13), w, int(h * .53))))
        area = scene["footprint"]["area_m2"]
        axis.set_title(f"{scene['batch']} / {scene['split']} | seed {scene['seed']}\n"
                       f"{area:.0f} m2 | {scene['floors']} floors | {scene['connections']} connections | {scene['threshold_features']} threshold cases",
                       fontsize=12)
    fig.suptitle("Compact WFC dataset | 40% of reference footprint | robot-scale geometry", fontsize=18)
    fig.subplots_adjust(top=.93, bottom=.01, left=.01, right=.99, hspace=.12, wspace=.03)
    fig.savefig(output / "overview_all.png", dpi=150, facecolor="#f7f9fc")
    plt.close(fig)
    sys.path.insert(0, str(ROOT / "common/geneator"))
    from visualize_wfc import draw_layout
    fig, axes = plt.subplots(rows, 2, figsize=(13, 6 * rows), squeeze=False, facecolor="#f7f9fc")
    for axis in axes.flat:
        axis.axis("off")
    layouts = []
    for axis, scene in zip(axes.flat, scenes):
        cfg = read_json(output / scene["folder"] / "config.resolved.json")
        floor = len(cfg["floor_z_m"]) // 2
        axis.axis("on")
        draw_layout(axis, cfg, floor)
        axis.set_title(f"seed {scene['seed']} | floor {floor} | {scene['footprint']['area_m2']:.0f} m2")
        layouts.append(dict(seed=scene["seed"], shape=cfg["shape"], footprint=cfg["footprint"],
                            stairs=cfg["stairs"], well_overlap_cells=scene["well_overlap_cells"],
                            landing_ports_in_wells=scene["landing_ports_in_wells"]))
    fig.suptitle("Actual compact scene layouts | middle floor | flat landing ports in green", fontsize=17)
    fig.subplots_adjust(top=.94, bottom=.05, left=.07, right=.98, hspace=.28, wspace=.2)
    fig.savefig(output / "layout_samples.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    write_json(output / "layout.report.json", dict(layout_samples=layouts,
                                                 actual_scenes=[s["id"] for s in scenes]))


def main():
    """按 dataset.json 的批次与 seed 依次生成场景、教师与分块。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/dataset.json")
    parser.add_argument("--batch", nargs="+", help="只生成这些批次，并替换其旧场景。")
    args = parser.parse_args()
    cfg = read_json(args.config)
    output = workspace_path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "runs/wfc_data_generation"
    logs.mkdir(parents=True, exist_ok=True)
    # 固定 Matplotlib 缓存目录，保证 headless 环境可复现
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))
    template = read_json(workspace_path(cfg["scene_config"]))
    robots = read_json(workspace_path(cfg["robots"]))
    teacher_settings = read_json(workspace_path(cfg["teacher_config"]))
    # manifest 记录条件定义、代价尺度、教师参数与场景模板哈希
    manifest = dict(version="compact_wfc_native_tomogram_stand_v2", condition_names=CONDITION_NAMES,
                    condition_scale=robots["condition_scale"], cost_scale=50., scenes=[],
                    area_ratio=template["domain_randomization"]["area_ratio"],
                    teacher_parameters=teacher_settings, generation_config=cfg,
                    scene_template_sha256=hashlib.sha256(workspace_path(cfg["scene_config"]).read_bytes()).hexdigest())
    if args.batch and (output / "manifest.json").exists():
        # 增量模式：保留未指定批次的既有场景
        manifest["scenes"] = [s for s in read_json(output / "manifest.json")["scenes"] if s["batch"] not in args.batch]
    for batch in cfg["batches"]:
        if args.batch and batch["id"] not in args.batch:
            continue
        for seed in batch["seeds"]:
            folder = output / f"scene_{seed}"
            print(f"\n{batch['id']} {batch['split']} seed={seed}", flush=True)
            # 每个阶段独立成子进程，失败时便于从日志定位
            stages = [
                ("mesh", ["common/geneator/generate_wfc_multifloor.py", "--config", workspace_path(cfg["scene_config"]), "--seed", seed, "--output", folder]),
                ("pcd", ["common/geneator/mesh_to_pcd.py", "--scene", folder, "--spacing", cfg["surface_spacing_m"]]),
                ("voxel", ["common/geneator/downsample_pcd.py", "--scene", folder, "--voxel", cfg["voxel_m"]]),
                ("teacher", ["-m", "Traversability.data.teacher", "--scene", folder,
                             "--robots", workspace_path(cfg["robots"]), "--teacher-config", workspace_path(cfg["teacher_config"])])]
            timings = {name:run_stage(command, logs / f"{batch['id']}_{seed}_{name}.log") for name,command in stages}
            # 场景完成后汇总质量指标并写入 generation.json 与总 manifest
            row = collect_scene(folder, batch, timings)
            write_json(folder / "generation.json", row)
            manifest["scenes"].append(row)
            manifest["scenes"].sort(key=lambda s:s["seed"])
            write_json(output / "manifest.json", manifest)
            # 同步更新每个批次的平铺索引文件
            for batch_id in {b["id"] for b in cfg["batches"]}:
                write_json(output / f"{batch_id}.json", dict(
                    id=batch_id, scenes=[s for s in manifest["scenes"] if s["batch"] == batch_id]))
            render_contact_sheet(output, manifest["scenes"])
            print(f"COMPLETED seed={seed} area={row['footprint']['area_m2']:.1f}m2 "
                  f"input={row['input_points']:,} targets={row['profiles'][0]['tomogram_points']:,}", flush=True)
    # 所有场景生成完毕后统一建立独立分块与查询近邻表
    from Traversability.data.blocks import prepare_blocks
    prepare_blocks(output, cfg["blocks"], args.batch)
    print(output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
