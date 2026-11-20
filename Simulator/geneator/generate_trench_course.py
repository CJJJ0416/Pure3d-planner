"""Generate a straight terrain-generator trench course for gap-width tests."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE / "terrain-generator"), str(HERE), str(ROOT)]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terrain_generator.trimesh_tiles.mesh_parts.basic_parts import create_box_mesh
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import BoxMeshPartsCfg
from terrain_generator.utils import merge_meshes

from common.pointcloud_io import write_pcd
from utils.mesh_sampling import sample_surface
from visualize_wfc import draw


def box(x_bounds, course_width, z_bounds):
    size = (x_bounds[1] - x_bounds[0], course_width, z_bounds[1] - z_bounds[0])
    pose = np.eye(4)
    pose[:3, 3] = (sum(x_bounds) / 2, 0, sum(z_bounds) / 2)
    cfg = BoxMeshPartsCfg(
        dim=(size[0], size[1], 0.0), add_floor=False, minimal_triangles=False,
        box_dims=(size,), transformations=(pose,),
    )
    return create_box_mesh(cfg)


def layout(widths, platform_length, approach_length):
    platform_lengths = [approach_length] + [platform_length] * (len(widths) - 1) + [approach_length]
    length = sum(platform_lengths) + sum(widths)
    x = -length / 2
    platforms, trenches = [], []
    for i, span in enumerate(platform_lengths):
        platforms.append([x, x + span])
        x += span
        if i < len(widths):
            trenches.append([x, x + widths[i]])
            x += widths[i]
    return platforms, trenches, length


def overview(mesh, trenches, top, bottom, output):
    preview = mesh.copy()
    normals = preview.face_normals
    colors = np.tile([94, 115, 135, 255], (len(mesh.faces), 1))
    up = normals[:, 2] > .9
    colors[up & (preview.triangles_center[:, 2] > (top + bottom) / 2)] = [201, 215, 226, 255]
    colors[up & (preview.triangles_center[:, 2] <= (top + bottom) / 2)] = [163, 120, 84, 255]
    preview.visual.face_colors = colors
    fig = plt.figure(figsize=(12, 7), facecolor="#f7f9fc")
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=.28)
    ax = fig.add_subplot(grid[0], projection="3d")
    draw(ax, [(preview, None)], mesh.bounds, elev=37, azim=-63)
    ax.set_box_aspect([*mesh.extents[:2], mesh.extents[2] + .3], zoom=1.45)
    for collection in ax.collections:
        collection.set_clip_on(False)
    ax.set_axis_off()
    for i, interval in enumerate(trenches, 1):
        ax.text(np.mean(interval), -.27 * mesh.extents[1], top + .10, f"T{i}",
                ha="center", color="#24384b", fontsize=10, weight="bold",
                bbox=dict(fc="#f7f9fc", ec="none", boxstyle="round,pad=.2", alpha=.9))
    profile = fig.add_subplot(grid[1])
    x, z = [mesh.bounds[0, 0]], [top]
    for lo, hi in trenches:
        x.extend([lo, lo, hi, hi])
        z.extend([top, bottom, bottom, top])
    x.append(mesh.bounds[1, 0])
    z.append(top)
    profile.fill_between(x, z, 0, color="#dbe4ec")
    profile.plot(x, z, color="#304a61", linewidth=1.8)
    for i, (lo, hi) in enumerate(trenches, 1):
        profile.annotate(f"T{i}  {hi - lo:.2f} m", ((lo + hi) / 2, bottom),
                         xytext=((lo + hi) / 2, top + .22), ha="center",
                         fontsize=9, color="#304a61",
                         arrowprops=dict(arrowstyle="-", lw=.7, color="#526e87"))
    profile.set(xlim=(x[0] - .19, x[-1] + .19), ylim=(0, top + .46),
                xlabel="X (m), crossing direction", ylabel="Z (m)", yticks=[0, bottom, top])
    profile.spines[["top", "right"]].set_visible(False)
    profile.set_facecolor(fig.get_facecolor())
    profile.grid(axis="y", alpha=.24)
    fig.suptitle("Straight trench course", fontsize=20, fontweight="bold", color="#23374a")
    fig.savefig(output / "overview.png", dpi=165, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/simulation/gap_max_straight_trenches")
    parser.add_argument("--widths", nargs="+", default=[.10, .20, .40, .90], type=float)
    parser.add_argument("--depth", type=float, default=.60)
    parser.add_argument("--course-width", type=float, default=4.0)
    parser.add_argument("--platform-length", type=float, default=1.50)
    parser.add_argument("--approach-length", type=float, default=1.80)
    parser.add_argument("--spacing", type=float, default=.025)
    args = parser.parse_args()
    dimensions = [*args.widths, args.depth, args.course_width, args.platform_length,
                  args.approach_length, args.spacing]
    if not all(np.isfinite(dimensions)) or min(dimensions) <= 0:
        parser.error("Geometry dimensions and sampling spacing must be finite and positive")

    platforms, trenches, length = layout(args.widths, args.platform_length, args.approach_length)
    trench_floor = .20
    top = trench_floor + args.depth
    base = box([-length / 2, length / 2], args.course_width, [0, trench_floor])
    supports = [box(span, args.course_width, [trench_floor, top]) for span in platforms]
    mesh = merge_meshes([base, *supports], minimal_triangles=True, engine="manifold")
    expected_volume = args.course_width * (trench_floor * length + args.depth * (length - sum(args.widths)))
    if not mesh.is_volume or mesh.body_count != 1 or not np.isclose(mesh.volume, expected_volume):
        raise RuntimeError("The trench course must form a single watertight solid with the expected volume")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh.export(output / "terrain.obj")
    points, sampling = sample_surface(mesh, args.spacing)
    write_pcd(output / "terrain.pcd", points)
    revision = json.loads((HERE / "TERRAIN_GENERATOR_PROVENANCE.json").read_text())["revision"]
    manifest = dict(
        generator="terrain_generator.trimesh_tiles.mesh_parts.basic_parts.create_box_mesh",
        terrain_generator_revision=revision, frame_id="map", units="m", up_axis="Z",
        bounds_m=mesh.bounds.tolist(), footprint_m=[length, args.course_width], top_z_m=top,
        floor_z_m=trench_floor, depth_m=args.depth, platform_x_spans_m=platforms,
        trenches=[
            dict(id=f"T{i + 1}", width_m=width, x_span_m=span,
                 y_span_m=[-args.course_width / 2, args.course_width / 2],
                 start_support_xyz=[sum(platforms[i]) / 2, 0, top],
                 goal_support_xyz=[sum(platforms[i + 1]) / 2, 0, top])
            for i, (width, span) in enumerate(zip(args.widths, trenches))
        ],
        mesh=dict(faces=len(mesh.faces), watertight=mesh.is_watertight, bodies=mesh.body_count, volume_m3=mesh.volume),
        pcd=dict(points=len(points), spacing_m=args.spacing, **sampling),
    )
    (output / "scene.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    overview(mesh, trenches, top, trench_floor, output)
    print(f"{output}: {len(trenches)} trenches, {len(mesh.faces):,} faces, {len(points):,} PCD points")


if __name__ == "__main__":
    main()
