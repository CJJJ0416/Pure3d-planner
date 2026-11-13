"""从已有 VST 教师导出归一化的原生 Tomogram 代价点云。

prepare.py 不重新建图，只读取 teacher.py 产出的 pickle，调用原生
tomogram_points() 把合并后的 ground 点云导出为 [x,y,z,trav_cost]。
"""
import argparse
import os
from pathlib import Path
import pickle
import sys

import numpy as np

from common.pointcloud_io import write_pcd
from utils.pointcloud import ROOT, read_json, write_json

# Matplotlib 与 VST 原生代码需要固定的缓存目录，避免写用户主目录
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))
sys.path.insert(0, str(ROOT / "common/vst_global_planner/src"))
from vst_global_planner.tomogram_cloud import tomogram_points


def render_tomograms(folder, scene, profiles, clouds):
    """绘制各组原生 Tomogram 代价点云的三维对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FormatStrFormatter

    fig = plt.figure(figsize=(5 * len(profiles), 7), facecolor="#f7f9fc")
    # 绿 -> 黄 -> 红：低代价到高代价
    cmap = LinearSegmentedColormap.from_list("trav_cost", ["#16a34a", "#fde047", "#ef4444"])
    norm = Normalize(0, 1)
    # 所有子图共用第一组点云的世界坐标范围
    lower, upper = clouds[0][:, :3].min(0), clouds[0][:, :3].max(0)
    span = upper - lower
    rng = np.random.default_rng(scene["seed"])
    for index, (profile, cloud) in enumerate(zip(profiles, clouds)):
        # 每幅图最多抽样 20 万点，仅影响显示，不影响导出文件
        chosen = rng.choice(len(cloud), min(200000, len(cloud)), replace=False)
        preview = cloud[chosen]
        axis = fig.add_subplot(1, len(profiles), index + 1, projection="3d")
        axis.scatter(*preview[:, :3].T, c=preview[:, 3], cmap=cmap, norm=norm,
                     s=.35, linewidths=0, depthshade=False, rasterized=True)
        axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                 zlim=(lower[2], upper[2]), xlabel="World X (m)", ylabel="World Y (m)")
        # Z 轴刻度对齐楼层高度，便于检查层间代价
        axis.set_zticks(scene["config"]["floor_z_m"])
        axis.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.zaxis.set_tick_params(labelsize=8, pad=0)
        axis.set_box_aspect((span[0], span[1], span[2] * 2.5))
        axis.set_proj_type("ortho")
        axis.view_init(elev=25, azim=-55)
        values = ", ".join(f"{value:g}" for value in profile["condition"])
        axis.set_title(f"{profile['id']}\n[{values}]\n{len(cloud):,} Tomogram points", fontsize=12)
    fig.suptitle(f"Native merged Tomogram cost clouds | seed {scene['seed']}\n"
                 "[clearance_min, slope_max, step_max, interval_free, stand_ratio]", fontsize=20)
    fig.text(.49, .055, "All finite native ground points are saved, including zero and barrier costs. "
             "Plot: up to 200,000 samples per condition. Z display scale x2.5.\n"
             "World XYZ in metres, Z up. Colours: normalized trav_cost. No input-PCD label lookup.",
             ha="center", fontsize=11, color="#526277")
    fig.subplots_adjust(left=.01, right=.93, top=.79, bottom=.18, wspace=.07)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=fig.add_axes([.96, .24, .009, .45]))
    bar.set_label("trav_cost [0, 1]")
    fig.savefig(folder / "tomogram_overview.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def export_tomograms(folder):
    """读取 teacher.json 与 robot_conditions.json，逐组导出 PCD 并更新元数据。"""
    folder = Path(folder)
    metadata = read_json(folder / "teacher.json")
    conditions = read_json(folder / "robot_conditions.json")
    scene = read_json(folder / "scene.json")
    clouds = []
    for profile, condition in zip(metadata["profiles"], conditions["profiles"]):
        with (folder / profile["vst"]).open("rb") as stream:
            payload = pickle.load(stream)
        # 原生 Tomogram 点云字段为 [x, y, z, cost]；cost 已在教师侧归一化
        cloud = tomogram_points(payload)
        filename = f"tomogram_{profile['id']}.pcd"
        write_pcd(folder / filename, cloud[:, :3], cloud[:, 3])
        # 记录点数、代价范围以及零代价/障碍代价点数，便于核对监督分布
        row = dict(tomogram=filename, tomogram_points=len(cloud),
                   tomogram_cost_min=float(cloud[:, 3].min()), tomogram_cost_max=float(cloud[:, 3].max()),
                   zero_cost_points=int(np.sum(cloud[:, 3] == 0)),
                   barrier_cost_points=int(np.sum(cloud[:, 3] == 1)))
        profile.update(row)
        condition.update(tomogram=filename, tomogram_points=len(cloud))
        clouds.append(cloud)
        print(f"{profile['id']}: {filename}, points={len(cloud):,}, "
              f"cost=[{row['tomogram_cost_min']:g}, {row['tomogram_cost_max']:g}]", flush=True)
    # 描述监督数据的表示方式，写入元数据供训练侧与可视化侧共同读取
    representation = dict(
        representation="native_merged_ground_tomogram", fields=["x", "y", "z", "trav_cost"],
        coordinate_frame=scene["frame_id"], units="m", cost_range=[0, 1],
        coordinates="native integer-centred XY grid, Z=ground",
        merge="adjacent ground height difference < slice_dh, propagate minimum cost",
        source_pcd=metadata["pcd"], source_pcd_sha256=metadata["pcd_sha256"],
    )
    metadata["supervision"] = representation
    metadata["tomogram_overview"] = "tomogram_overview.png"
    conditions["supervision"] = representation
    write_json(folder / "teacher.json", metadata)
    write_json(folder / "robot_conditions.json", conditions)
    scene["teacher"]["supervision"] = representation["representation"]
    scene["teacher"]["tomograms"] = [profile["tomogram"] for profile in metadata["profiles"]]
    scene["teacher"]["tomogram_overview"] = "tomogram_overview.png"
    scene["stage"] = "native normalized Tomogram cost clouds ready for training"
    write_json(folder / "scene.json", scene)
    shown = metadata.get("overview_profiles", [p["id"] for p in metadata["profiles"]])
    selected = [(p, c) for p, c in zip(metadata["profiles"], clouds) if p["id"] in shown]
    render_tomograms(folder, scene, [p for p, _ in selected], [c for _, c in selected])
    return metadata


def main():
    """命令行入口：对指定场景目录执行 Tomogram 导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "datasets/wfc_capability_dr_stand_v3/scene_20261109")
    args = parser.parse_args()
    export_tomograms(args.scene.resolve())


if __name__ == "__main__":
    main()
