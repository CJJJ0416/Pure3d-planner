"""Targeted native terrain-generator stair families, with actual metric tread changes."""
import sys
from pathlib import Path

import numpy as np
import trimesh

from utils.pointcloud import ROOT, voxel_downsample, write_json
from utils.mesh_sampling import sample_surface
from common.pointcloud_io import write_pcd
from common.geneator.wfc_connections import build_connection, box, run_for_tread

sys.path[:0] = [str(ROOT / "datasets/cache/wfc_runtime"),
                str(ROOT / "common/geneator/terrain-generator")]


def generate_family_scene(folder, family_seed, variant, kind, split):
    """Three variants share risers, width, rails and yaw, changing realised tread depth."""
    rng = np.random.default_rng(family_seed)
    width = float(rng.uniform(.4, 1.6))
    target_riser = float(rng.uniform(.08, .38))
    target_rise = float(rng.uniform(2.1, 2.8))
    flights = 2 if kind == "switchback" else 1
    n = int(np.ceil(target_rise / (flights * target_riser)))
    rise = flights * n * target_riser
    depth = float([.14, .29, .47][variant] + rng.uniform(-.015, .015))
    railings = bool(rng.integers(0, 2))
    yaw = float(rng.uniform(-np.pi, np.pi))
    phase = rng.uniform(-.05, .05, 2)
    thickness = .12
    run = run_for_tread(n, depth)
    inner = .25
    if kind == "straight":
        size = [width + 1.2, run + .9]
        turns = 1.
    elif kind == "switchback":
        size = [2 * width + 1.15, run + width + .9]
        turns = 1.
    else:
        size = [2 * (inner + width) + 1.2] * 2
        turns = depth * n / ((inner + width / 2) * 2 * np.pi)
    spec = dict(id="stair_0", kind=kind, from_floor=0, to_floor=1,
                width_m=width, target_riser_m=target_riser + 1e-8,
                tread_depth_m=depth, exact_tread_depth=True, railings=railings,
                railing_height_m=float(rng.uniform(.8, 1.1)), turns=turns,
                threshold_band="random", bay_rows=[0, 1], bay_columns=[0, 1],
                rotation_deg=0, ramp_layout="straight")
    if kind == "spiral":
        spec["inner_radius_m"] = inner
    cfg = dict(shape=[1, 1], tile_dim_m=[*size, 2.], floor_z_m=[0., rise],
               floor_thickness_m=thickness, reference_vst=dict(step_max=.3, slope_max=.4))
    connection, record = build_connection(cfg, spec)
    # Continuous upper/lower landings surround the native stair well.
    margin = 1.25
    parts = [connection, box([size[0] + 2 * margin, size[1] + 2 * margin, thickness],
                             [0, 0, -thickness / 2])]
    for sign in [-1, 1]:
        parts.append(box([size[0] + 2 * margin, margin, thickness],
                         [0, sign * (size[1] + margin) / 2, rise - thickness / 2]))
        parts.append(box([margin, size[1], thickness],
                         [sign * (size[0] + margin) / 2, 0, rise - thickness / 2]))
    if family_seed % 3 == 0:
        parts.append(box([.12, size[1] + 2 * margin, rise + .8],
                         [size[0] / 2 + margin, 0, (rise + .8) / 2]))
    mesh = trimesh.util.concatenate(parts)
    transform = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    transform[:2, 3] = phase
    mesh.apply_transform(transform)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(folder / "terrain.ply")
    points, sampled = sample_surface(mesh, .03)
    write_pcd(folder / "terrain.pcd", points)
    down = voxel_downsample(points, .05)
    write_pcd(folder / "terrain_downsampled.pcd", down)
    record.update(yaw_rad=yaw, translation_xy=phase.tolist(), family_seed=family_seed)
    scene = dict(seed=family_seed * 10 + variant, family_id=f"family_{family_seed}",
                 split=split, variant=variant, kind=kind, config=cfg, stairs=[record],
                 mesh_faces=len(mesh.faces), input_points=len(down),
                 generator="native terrain-generator stair parts with metric family variations",
                 footprint=dict(area_m2=float(np.prod(np.array(size) + 2 * margin))),
                 sampling=sampled, transform=transform.tolist(), bounds=mesh.bounds.tolist())
    write_json(folder / "scene.json", scene)
    render_overview(folder, points, scene)
    return scene


def stair_membership(points, scene):
    """A geometric ROI for sampling/reporting. It never removes supervision labels."""
    r = scene["stairs"][0]
    yaw = r["yaw_rad"]
    c, s = np.cos(yaw), np.sin(yaw)
    xy = (points[:, :2] - r["translation_xy"]) @ np.array([[c, -s], [s, c]])
    width, length = r["unrotated_footprint_m"]
    base = r.get("base_z_m", 0.)
    if scene.get('augmentation', {}).get('template_id'):
        height = (points[:, 2] > base + .03) & (points[:, 2] < base + r['rise_m'] - .1)
        if r['kind'] == 'spiral':
            radius = np.linalg.norm(xy, axis=1)
            return height & (radius > r['inner_radius_m'] - .05) & (radius < r['outer_radius_m'] + .1)
        half_width = (r['width_m'] / 2 if r['kind'] == 'straight' else r['width_m'] + .125) + .1
        return height & (abs(xy[:, 0]) < half_width) & (abs(xy[:, 1]) < length / 2 - .45)
    return ((abs(xy[:, 0]) < width / 2 - .4) & (abs(xy[:, 1]) < length / 2 - .35)
            & (points[:, 2] > base + .04) & (points[:, 2] < base + r["rise_m"] + .06))


def generate_augmented_stair_scene(folder, spec):
    """独立 seed 的直梯、折返梯、旋转梯及两层/三层局部建筑上下文。"""
    rng = np.random.default_rng(spec["seed"])
    kind = spec.get('kind', 'straight')
    width, riser, depth = [spec[k] for k in ["width_m", "riser_m", "tread_depth_m"]]
    requested_rise = spec.get('rise_m', float(rng.uniform(2.7, 3.9)))
    flights = 2 if kind == 'switchback' else 1
    n = int(np.rint(requested_rise / (flights * riser)))
    rise, run = flights*n*riser, run_for_tread(n, depth)
    base = float(rng.uniform(2.5, 3.8)) if spec["three_floors"] else 0.
    thickness, margin = float(rng.uniform(.1, .18)), float(rng.uniform(1.1, 1.7))
    if kind == 'straight':
        size = [width + 2*spec['side_gap_m'], run + .9]
    elif kind == 'switchback':
        size = [2*width + .25 + 2*spec['side_gap_m'], run + width + .9]
    else:
        size = [2*(spec['inner_radius_m'] + width + spec['side_gap_m'])] * 2
    floors = [0., base, base+rise] if base else [0., rise]
    target = dict(id="target_stair", kind=kind, from_floor=int(base > 0),
                  to_floor=len(floors)-1, width_m=width, target_riser_m=riser+1e-8,
                  tread_depth_m=depth, exact_tread_depth=True, railings=spec["railings"],
                  railing_height_m=float(rng.uniform(.8, 1.15)), turns=1.,
                  threshold_band="random", bay_rows=[0, 1], bay_columns=[0, 1], rotation_deg=0)
    if kind == 'spiral':
        target.update(inner_radius_m=spec['inner_radius_m'], turns=spec['turns'])
    cfg = dict(shape=[1, 1], tile_dim_m=[*size, 2.], floor_z_m=floors,
               floor_thickness_m=thickness, reference_vst=dict(step_max=.4, slope_max=.6))
    connection, record = build_connection(cfg, target)
    parts = [connection]
    # 顶层外平台与原生井口周边相连，下方背景层只提供局部几何上下文。
    for sign in [-1, 1]:
        parts.append(box([size[0]+2*margin, margin, thickness],
                         [0, sign*(size[1]+margin)/2, base+rise-thickness/2]))
        parts.append(box([margin, size[1], thickness],
                         [sign*(size[0]+margin)/2, 0, base+rise-thickness/2]))
    parts.append(box([size[0]+2*margin, size[1]+2*margin, thickness],
                     [0, 0, base-thickness/2]))
    if base:
        parts.append(box([size[0]+2*margin, size[1]+2*margin, thickness], [0, 0, -thickness/2]))
    if spec["wall"]:
        parts.append(box([.12, size[1]+2*margin, rise+.8],
                         [size[0]/2+margin-.12, 0, base+(rise+.8)/2]))
    mesh = trimesh.util.concatenate(parts)
    yaw = float(rng.uniform(-np.pi, np.pi))
    transform = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    transform[:2, 3] = rng.uniform(-.15, .15, 2)
    if spec.get('mirror', False):
        transform[:3, :3] = transform[:3, :3] @ np.diag([-1., 1., 1.])
    mesh.apply_transform(transform)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(folder/"terrain.ply")
    points, sampled = sample_surface(mesh, .03)
    write_pcd(folder/"terrain.pcd", points)
    down = voxel_downsample(points, .05)
    write_pcd(folder/"terrain_downsampled.pcd", down)
    record.update(yaw_rad=yaw, translation_xy=transform[:2, 3].tolist(),
                  base_z_m=base, family_seed=spec["seed"], mirrored=spec.get('mirror', False))
    scene = dict(seed=spec["seed"], family_id=f"family_{spec['seed']}",
                 split=spec["split"], variant=0, kind=kind, config=cfg, stairs=[record],
                 mesh_faces=len(mesh.faces), input_points=len(down), generator="native terrain-generator stair parts",
                 footprint=dict(area_m2=float(np.prod(np.array(size)+2*margin))), sampling=sampled,
                 transform=transform.tolist(), bounds=mesh.bounds.tolist(), augmentation=spec)
    write_json(folder/"scene.json", scene)
    render_overview(folder, points, scene)
    return scene


def render_overview(folder, points, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = np.random.default_rng(0).choice(len(points), min(80000, len(points)), replace=False)
    xyz = points[chosen]
    fig = plt.figure(figsize=(14, 6), facecolor="white")
    for column, (elev, azim) in enumerate([(30, -55), (5, -85)], 1):
        ax = fig.add_subplot(1, 2, column, projection="3d")
        ax.scatter(*xyz.T, c=xyz[:, 2], cmap="terrain", s=.3, linewidths=0)
        ax.set_box_aspect(np.ptp(points, axis=0))
        ax.view_init(elev, azim)
        ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
    r = scene["stairs"][0]
    depth = r.get("tread_depth_m", r.get("tread_depth_at_mid_radius_m"))
    fig.suptitle(f"{scene['family_id']} / {scene['split']} / {scene['kind']} / variant {scene['variant']}\n"
                 f"actual riser={r['riser_m']:.3f} m | tread={depth:.3f} m | width={r['width_m']:.3f} m | rails={r['railings']}")
    fig.tight_layout()
    fig.savefig(folder / "overview.png", dpi=130)
    plt.close(fig)
