"""Voxel-downsample a WFC scene PCD and show the retained stairs and railings."""
import argparse
from pathlib import Path
import time

from mesh_to_pcd import ROOT, inspect_density, scatter

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import trimesh
from common.pointcloud_io import read_pcd, write_pcd
from utils.pointcloud import read_json, voxel_downsample, write_json


def connection_crop(points, record, scene, folder):
    part = trimesh.load_mesh(folder / f"{record['id']}.obj", process=False)
    low, high = part.bounds
    points = points[((points >= low - .00001) & (points <= high + .00001)).all(1)]
    rotation = trimesh.transformations.rotation_matrix(
        -np.deg2rad(scene["config"]["yaw_deg"]), [0, 0, 1],
    )[:3, :3]
    local = points @ rotation.T
    local[:, :2] -= record["center_local_xy"]
    width, depth = record["footprint_m"]
    base = scene["config"]["floor_z_m"][record["from_floor"]]
    top = scene["config"]["floor_z_m"][record["to_floor"]] + record["railing_height_m"] + .15
    keep = ((np.abs(local[:, 0]) <= width / 2 + .001)
            & (np.abs(local[:, 1]) <= depth / 2 + .001)
            & (local[:, 2] >= base - .3) & (local[:, 2] <= top))
    bounds = np.array([[-width / 2, -depth / 2, base - .2], [width / 2, depth / 2, top]])
    return local[keep], bounds


def render_comparison(folder, source, result, scene, mesh, voxel_size,
                      output_name="pointcloud_downsampled.png", source_label="Original"):
    fig = plt.figure(figsize=(19, 13), facecolor="#f7f9fc")
    norm = Normalize(mesh.bounds[0, 2], mesh.bounds[1, 2])
    records = [next(r for r in scene["stairs"] if r["kind"] == kind and r["railings"])
               for kind in ["straight", "spiral"]]
    rng = np.random.default_rng(scene["seed"])
    for row, (label, points) in enumerate([
        (source_label, source), (f"Voxel {voxel_size * 100:g} cm", result),
    ]):
        preview = points[rng.choice(len(points), min(200000, len(points)), replace=False)]
        axis = fig.add_subplot(2, 3, row * 3 + 1, projection="3d")
        scatter(axis, preview, mesh.bounds, norm)
        span = mesh.bounds[1] - mesh.bounds[0]
        axis.set_box_aspect((span[0], span[1], span[2] * 2.2))
        axis.set_zticks(scene["config"]["floor_z_m"])
        axis.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_zlabel("")
        axis.set_title(f"{label} | {len(points):,} points\n{len(preview):,} display samples", fontsize=13)
        for column, record in enumerate(records, start=1):
            local, bounds = connection_crop(points, record, scene, folder)
            axis = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            scatter(axis, local, bounds, norm, elev=40, azim=-135, size=.2)
            axis.set(xlabel="Local X (m)", ylabel="Local Y (m)", zlabel="")
            axis.set_title(f"{label} | {record['kind']} + rails\n{len(local):,} points, no display thinning", fontsize=13)
    fig.suptitle(f"PCD voxel downsampling | seed {scene['seed']}\n"
                 f"{len(source):,} -> {len(result):,} points | {100 * (1 - len(result) / len(source)):.1f}% fewer points",
                 fontsize=20)
    fig.text(.49, .025, "Left: whole-scene preview, Z scale x2.2. Centre/right: all points in each crop.\n"
             "One centroid per occupied voxel. World frame and metre units retained.",
             ha="center", fontsize=11, color="#526277")
    fig.subplots_adjust(left=.015, right=.91, top=.89, bottom=.09, hspace=.14, wspace=.04)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                       cax=fig.add_axes([.943, .24, .014, .52]))
    bar.set_label("World Z (m)")
    fig.savefig(folder / output_name, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=ROOT / "datasets/wfc_capability_dr_v2/scene_20261109")
    parser.add_argument("--voxel", type=float, default=.05, help="Voxel side length in metres.")
    parser.add_argument("--output", type=Path, default=Path("terrain_downsampled.pcd"))
    parser.add_argument("--compare", type=Path, help="Existing PCD to show in the first comparison row.")
    args = parser.parse_args()
    folder = args.scene.resolve()
    output = folder / args.output
    report_name = f"{output.stem}.report.json"
    overview_name = f"{output.stem}.png"
    scene = read_json(folder / "scene.json")
    source, normals = read_pcd(folder / "terrain.pcd")
    del normals
    print(f"Downsampling {len(source):,} points with {args.voxel:g} m voxels ...", flush=True)
    started = time.perf_counter()
    result = voxel_downsample(source, args.voxel)
    downsample_s = time.perf_counter() - started
    write_pcd(output, result)
    print(f"Saved {len(result):,} points in {downsample_s:.2f} s. Inspecting geometry ...", flush=True)
    mesh = trimesh.load_mesh(folder / "terrain.obj", process=False)
    inspection = inspect_density(mesh, result, scene, folder)
    report = dict(
        seed=scene["seed"], input_pcd="terrain.pcd", output_pcd=output.name,
        method="centroid per occupied world-aligned voxel", voxel_size_m=args.voxel,
        input_points=len(source), output_points=len(result),
        reduction_percent=100 * (1 - len(result) / len(source)),
        frame_id=scene["frame_id"], units="m", z_axis="up",
        bounds_world=[result.min(0).tolist(), result.max(0).tolist()],
        pcd_size_bytes=output.stat().st_size,
        downsample_s=downsample_s, overview=overview_name, **inspection,
    )
    write_json(folder / report_name, report)
    source_label = "Original"
    if args.compare:
        del source
        source, normals = read_pcd(folder / args.compare)
        del normals
        source_label = args.compare.stem
    render_comparison(folder, source, result, scene, mesh, args.voxel,
                      output_name=overview_name, source_label=source_label)
    record = dict(
        file=output.name, points=len(result), voxel_size_m=args.voxel,
        report=report_name, overview=overview_name,
    )
    scene["pointcloud"].setdefault("downsampled_versions", {})[output.name] = record
    if output.name == "terrain_downsampled.pcd":
        scene["pointcloud"]["downsampled"] = record
    write_json(folder / "scene.json", scene)
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
