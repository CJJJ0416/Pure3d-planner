"""Step 2: complete WFC mesh -> world-frame surface PCD and density inspection."""
import argparse
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "datasets/cache/wfc_runtime"), str(ROOT)]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from utils.mesh_sampling import sample_surface
from common.pointcloud_io import write_pcd
from utils.pointcloud import read_json, write_json
from visualize_wfc import draw


def distance_summary(values):
    return {name: float(value) for name, value in zip(
        ["median_m", "p95_m", "p99_m", "max_m"], np.percentile(values, [50, 95, 99, 100]),
    )}


def inspect_density(mesh, points, scene, folder):
    """Measure saved-sample spacing and coverage on independent mesh probes."""
    tree = cKDTree(points)
    rng = np.random.default_rng(scene["seed"])
    chosen = rng.choice(len(points), 20000, replace=False)
    neighbours = tree.query(points[chosen], k=2)[0][:, 1]
    probes, _ = trimesh.sample.sample_surface(mesh, 20000, seed=scene["seed"])
    coverage = tree.query(probes)[0]
    connections = []
    for record in scene["stairs"]:
        part = trimesh.load_mesh(folder / f"{record['id']}.obj", process=False)
        probes, _ = trimesh.sample.sample_surface(part, 2000, seed=scene["seed"])
        row = dict(id=record["id"], kind=record["kind"], railings=record["railings"],
                   mesh_to_pcd=distance_summary(tree.query(probes)[0]))
        if record["railings"]:
            rail_faces = np.all(part.visual.face_colors[:, :3] == [65, 76, 89], axis=1)
            rails = part.submesh([rail_faces], append=True)
            probes, _ = trimesh.sample.sample_surface(rails, 1000, seed=scene["seed"])
            row["railing_to_pcd"] = distance_summary(tree.query(probes)[0])
        connections.append(row)
    return dict(
        nearest_neighbour=dict(probe_points=len(chosen), **distance_summary(neighbours)),
        mesh_to_pcd=dict(probe_points=20000, **distance_summary(coverage)),
        connections=connections,
    )


def scatter(axis, points, bounds, norm, elev=30, azim=-55, size=.35):
    axis.scatter(*points.T, c=points[:, 2], cmap="viridis", norm=norm,
                 s=size, linewidths=0, depthshade=False, rasterized=True)
    low, high = bounds
    span = high - low
    axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
             zlim=(low[2], high[2] + .2), xlabel="World X (m)",
             ylabel="World Y (m)", zlabel="World Z (m)")
    axis.set_box_aspect((span[0], span[1], span[2] + .2))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elev, azim=azim)
    axis.set_facecolor("#f7f9fc")


def render_pointcloud(folder, mesh, points, scene, spacing):
    """Use only display thinning. All samples remain in terrain.pcd."""
    rng = np.random.default_rng(scene["seed"])
    preview = points[rng.choice(len(points), min(300000, len(points)), replace=False)]
    bounds = mesh.bounds
    norm = Normalize(bounds[0, 2], bounds[1, 2])
    fig = plt.figure(figsize=(19, 13), facecolor="#f7f9fc")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.12, 1])
    axis = fig.add_subplot(grid[0, 0], projection="3d")
    draw(axis, [(mesh, "#91a6bb")], bounds, elev=30)
    axis.set_title("Source mesh | full geometry")
    axis = fig.add_subplot(grid[0, 1], projection="3d")
    scatter(axis, preview, bounds, norm)
    axis.set_title(f"Full PCD | {len(preview):,} display samples")
    axis = fig.add_subplot(grid[0, 2], projection="3d")
    section = preview[(preview[:, 1] >= 0) | (preview[:, 2] < scene["config"]["floor_z_m"][1] - .25)]
    scatter(axis, section, bounds, norm, elev=23)
    axis.set_title("PCD section | front upper points hidden")
    for axis in fig.axes:
        span = bounds[1] - bounds[0]
        axis.set_box_aspect((span[0], span[1], span[2] * 2.2))
        axis.set_zticks(scene["config"]["floor_z_m"])
        axis.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Draw actual output points in three connection bays, with no display thinning.
    details = [next(r for r in scene["stairs"] if r["kind"] == kind and r["railings"])
               for kind in ["straight", "switchback", "spiral"]]
    rotation = trimesh.transformations.rotation_matrix(-np.deg2rad(scene["config"]["yaw_deg"]), [0, 0, 1])[:3, :3]
    for column, record in enumerate(details):
        part = trimesh.load_mesh(folder / f"{record['id']}.obj", process=False)
        low, high = part.bounds
        crop = points[((points >= low - .00001) & (points <= high + .00001)).all(1)]
        local = crop @ rotation.T
        local[:, :2] -= record["center_local_xy"]
        width, depth = record["footprint_m"]
        base = scene["config"]["floor_z_m"][record["from_floor"]]
        top = scene["config"]["floor_z_m"][record["to_floor"]] + record["railing_height_m"] + .15
        inside = ((np.abs(local[:, 0]) <= width / 2 + .001)
                  & (np.abs(local[:, 1]) <= depth / 2 + .001)
                  & (local[:, 2] >= base - .3) & (local[:, 2] <= top))
        local = local[inside]
        local_bounds = np.array([[-width / 2, -depth / 2, base - .2], [width / 2, depth / 2, top]])
        axis = fig.add_subplot(grid[1, column], projection="3d")
        scatter(axis, local, local_bounds, norm, elev=40, azim=-135, size=.13)
        axis.set(xlabel="Local X (m)", ylabel="Local Y (m)")
        axis.set_title(f"{record['kind'].title()} + rails | {len(local):,} points\nAll PCD samples in this bay")
    fig.suptitle(
        f"Mesh -> complete surface PCD | seed {scene['seed']}\n"
        f"{len(points):,} points | target spacing {spacing * 100:g} cm | map frame, Z up, metres",
        fontsize=20,
    )
    fig.text(.49, .024, "Top row is thinned for display. Bottom row retains every point in each crop.\n"
             "Top row Z scale x2.2 for visibility. Colours show world Z. PCD retains undersides and contact faces.",
             ha="center", fontsize=11, color="#526277")
    fig.subplots_adjust(left=.015, right=.91, top=.89, bottom=.09, hspace=.12, wspace=.04)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                       cax=fig.add_axes([.943, .24, .014, .52]))
    bar.set_label("World Z (m)")
    fig.savefig(folder / "pointcloud_overview.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=ROOT / "datasets/wfc_capability_dr_v2/scene_20261109")
    parser.add_argument("--spacing", type=float, default=.03, help="Surface row/segment spacing in metres.")
    args = parser.parse_args()
    folder = args.scene.resolve()
    scene = read_json(folder / "scene.json")
    mesh = trimesh.load_mesh(folder / "terrain.obj", process=False)
    started = time.perf_counter()
    print(f"Sampling {len(mesh.faces):,} triangles at {args.spacing:g} m ...", flush=True)
    points, stats = sample_surface(mesh, args.spacing)
    sampling_s = time.perf_counter() - started
    print(f"Sampled {len(points):,} unique points in {sampling_s:.2f} s", flush=True)
    started = time.perf_counter()
    write_pcd(folder / "terrain.pcd", points)
    write_s = time.perf_counter() - started
    print(f"Saved terrain.pcd in {write_s:.2f} s. Inspecting coverage ...", flush=True)
    started = time.perf_counter()
    inspection = inspect_density(mesh, points, scene, folder)
    inspection_s = time.perf_counter() - started
    report = dict(
        seed=scene["seed"], input_mesh="terrain.obj", output_pcd="terrain.pcd",
        frame_id=scene["frame_id"], units="m", z_axis="up", fields=["x", "y", "z"],
        method="deterministic triangle rows, including boundaries and vertices",
        source_surface="all source triangles, including contact/interior faces of concatenated solids",
        spacing_m=args.spacing, deduplication_decimals=6, points=len(points),
        mesh_faces=len(mesh.faces), mesh_area_m2=float(mesh.area),
        bounds_world=[points.min(0).tolist(), points.max(0).tolist()],
        pcd_size_bytes=(folder / "terrain.pcd").stat().st_size,
        sampling_s=sampling_s, write_s=write_s, inspection_s=inspection_s,
        overview="pointcloud_overview.png", **stats, **inspection,
    )
    write_json(folder / "pointcloud.report.json", report)
    print("Rendering pointcloud_overview.png ...", flush=True)
    render_pointcloud(folder, mesh, points, scene, args.spacing)
    scene["pointcloud"] = dict(file="terrain.pcd", points=len(points), spacing_m=args.spacing,
                               report="pointcloud.report.json", overview="pointcloud_overview.png")
    scene["stage"] = "mesh and surface PCD ready, no VST labels or neural-network training"
    write_json(folder / "scene.json", scene)
    print(f"Done: {folder / 'terrain.pcd'}", flush=True)


if __name__ == "__main__":
    main()
