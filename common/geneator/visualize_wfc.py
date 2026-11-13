"""Headless overview of the actual WFC mesh, its floor layers and inter-floor stairs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from wfc_connections import COLORS as CONNECTION_COLORS
from wfc_randomization import floor_layout


COLORS = ["#779e82", "#b29a75", "#829bbd", "#a58cb9"]


def draw_layout(axis, cfg, floor):
    """Actual reserved bays in the assembly frame, before the scene-wide yaw."""
    rows, columns = cfg["shape"]
    dx, dy = cfg["tile_dim_m"][:2]
    extent = [-columns * dx / 2, columns * dx / 2, -rows * dy / 2, rows * dy / 2]
    axis.imshow(floor_layout(cfg, floor), origin="upper", extent=extent,
                cmap=ListedColormap(["#ffffff", "#dce4ed"]), vmin=0, vmax=1,
                interpolation="nearest")
    for record in cfg["stairs"]:
        if floor not in (record["from_floor"], record["to_floor"]):
            continue
        r0, r1 = record["bay_rows"]
        c0, c1 = record["bay_columns"]
        x, y = (c0 - columns / 2) * dx, (rows / 2 - r1) * dy
        width, height = (c1 - c0) * dx, (r1 - r0) * dy
        color = np.asarray(CONNECTION_COLORS[record["kind"]][:3]) / 255
        incoming = floor == record["to_floor"]
        axis.add_patch(Rectangle((x, y), width, height, facecolor=(*color, .38 if incoming else .8),
                                 edgecolor=color, linewidth=1.5, hatch="//" if incoming else None))
        cx, cy = x + width / 2, y + height / 2
        theta = np.deg2rad(record["rotation_deg"])
        direction = np.array([-np.sin(theta), np.cos(theta)]) * min(width, height) * .28
        axis.annotate("", xy=np.array([cx, cy]) + direction,
                      xytext=np.array([cx, cy]) - direction,
                      arrowprops=dict(arrowstyle="-|>", color="#182b40", linewidth=1.4))
        index = record["id"].rsplit("_", 1)[-1]
        axis.text(cx, y + height * .12,
                  f"F{record['from_floor']}-F{record['to_floor']} #{index}\n{record['rotation_deg']} deg",
                  ha="center", va="center", fontsize=7,
                  bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
        port_name = "upper_landing_cell" if incoming else "lower_landing_cell"
        if port_name in record:
            row, column = record[port_name]
            axis.plot((column - (columns - 1) / 2) * dx,
                      ((rows - 1) / 2 - row) * dy, marker="o", color="#163c28",
                      markersize=4, markeredgecolor="white")
    axis.set(xlim=extent[:2], ylim=extent[2:], xlabel="Assembly X (m)", ylabel="Assembly Y (m)")
    axis.set_aspect("equal")
    axis.set_title(f"Floor {floor} | Z = {cfg['floor_z_m'][floor]:.2f} m")
    axis.grid(alpha=.2)


def render_layout(output, cfg):
    count = len(cfg["floor_z_m"])
    fig, axes = plt.subplots(1, count, figsize=(5.7 * count, 6.5), facecolor="#f7f9fc")
    for floor, axis in enumerate(axes):
        draw_layout(axis, cfg, floor)
    handles = [Patch(facecolor=np.asarray(color[:3]) / 255, label=kind.title())
               for kind, color in CONNECTION_COLORS.items()]
    handles += [Patch(facecolor="#cbd4de", hatch="//", label="Arrives from lower floor")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .07))
    fig.suptitle(f"Whole-map connection placement | seed {cfg['seed']}\n"
                 "Independent XY positions and quarter-turn orientations", fontsize=19)
    fig.text(.5, .035, f"Assembly coordinates before global yaw {cfg['yaw_deg']:.1f} deg. "
             "Arrows show assembly +Y orientation. White: floor opening.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.045, right=.985, top=.83, bottom=.2, wspace=.22)
    fig.savefig(Path(output) / "layout_overview.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw(axis, parts, bounds, elev=34, azim=-55):
    camera = np.array([np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
                       np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)), np.sin(np.deg2rad(elev))])
    all_triangles, all_colors = [], []
    for mesh, color in parts:
        visible = mesh.face_normals @ camera > 0
        triangles = mesh.triangles[visible]
        light = np.array([-0.4, -0.5, 1.0])
        light /= np.linalg.norm(light)
        shade = 0.6 + 0.4 * np.maximum(mesh.face_normals[visible] @ light, 0)
        base = (mesh.visual.face_colors[visible, :3] / 255.0 if color is None
                else np.asarray(to_rgb(color))[None])
        colors = base * shade[:, None]
        all_triangles.append(triangles)
        all_colors.append(colors)
    # Sort faces across all floors together so a whole lower-floor collection
    # does not paint over stairs whose geometry crosses between collections.
    axis.add_collection3d(Poly3DCollection(
        np.concatenate(all_triangles), facecolors=np.concatenate(all_colors),
        edgecolors="none", linewidths=0, zsort="average",
    ))
    low, high = bounds
    span = high - low
    axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2] + 0.3),
             xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
    axis.set_box_aspect((span[0], span[1], span[2] + 0.3))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elev, azim=azim)
    axis.set_facecolor("#f7f9fc")


def render(output, cfg, floors, stairs):
    output = Path(output)
    bounds = np.array([
        np.vstack([m.bounds[0] for m in [*floors, *stairs]]).min(0),
        np.vstack([m.bounds[1] for m in [*floors, *stairs]]).max(0),
    ])
    fig = plt.figure(figsize=(19, 13), facecolor="#f7f9fc")
    columns = max(3, len(floors))
    grid = fig.add_gridspec(2, columns, height_ratios=[1.45, 1])
    axis = fig.add_subplot(grid[0, :columns - 1], projection="3d")
    connection_parts = [
        (mesh, None if "kind" in cfg["stairs"][i] else "#d27c44") for i, mesh in enumerate(stairs)
    ]
    parts = [(mesh, COLORS[i % len(COLORS)]) for i, mesh in enumerate(floors)] + connection_parts
    draw(axis, parts, bounds)
    axis.set_zticks(cfg["floor_z_m"])
    axis.set_title("Whole scene | overlapping floors, bridges and varied connections", fontsize=13)
    axis = fig.add_subplot(grid[0, columns - 1], projection="3d")
    # Hide the front half of upper-floor triangles in this diagnostic view only.
    section = []
    for i, mesh in enumerate(floors):
        mesh = mesh.copy()
        if i > 0:
            mesh.update_faces(mesh.triangles_center[:, 1] >= 0)
        section.append((mesh, COLORS[i % len(COLORS)]))
    section += connection_parts
    draw(axis, section, bounds, elev=23)
    axis.set_zticks(cfg["floor_z_m"])
    axis.set_title("Section view | front upper geometry hidden", fontsize=11)
    for i, mesh in enumerate(floors):
        axis = fig.add_subplot(grid[1, i], projection="3d")
        own_bounds = mesh.bounds.copy()
        own_bounds[1, 2] = max(own_bounds[1, 2], own_bounds[0, 2] + 2.0)
        draw(axis, [(mesh, COLORS[i % len(COLORS)])], own_bounds, elev=60)
        axis.set_zticks([cfg["floor_z_m"][i]])
        axis.set_title(f"Floor {i} | base height {cfg['floor_z_m'][i]:g} m", fontsize=13)
    fig.suptitle(
        f"WFC rough multi-floor terrain | seed {cfg['seed']}\n"
        f"{cfg['shape'][1]*cfg['tile_dim_m'][0]:.1f} x {cfg['shape'][0]*cfg['tile_dim_m'][1]:.1f} m | "
        f"{len(floors)} overlapping floors | {len(stairs)} inter-floor connections",
        fontsize=19, y=0.98,
    )
    caption = ("Orange: straight | Red: switchback | Purple: spiral | Teal: ramp | Dark grey: railings"
               if "domain_randomization" in cfg else
               "Native terrain-generator WFC tiles. Stacked floors and stair wells assembled explicitly. Metre coordinates.")
    fig.text(0.5, 0.015, caption,
             ha="center", fontsize=11, color="#526277")
    fig.subplots_adjust(left=0.01, right=0.94, top=0.91, bottom=0.065, hspace=0.10, wspace=0.03)
    fig.savefig(output / "overview.png", dpi=145, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def render_connections(output, cfg, meshes, records):
    columns = 5
    rows = (len(meshes) + columns - 1) // columns
    fig = plt.figure(figsize=(23, rows * 4.8), facecolor="#f7f9fc")
    for i, (mesh, record) in enumerate(zip(meshes, records)):
        axis = fig.add_subplot(rows, columns, i + 1, projection="3d")
        detail = mesh.copy()
        detail.apply_transform(trimesh.transformations.rotation_matrix(
            -np.deg2rad(cfg.get("yaw_deg", 0)), [0, 0, 1]
        ))
        detail.apply_translation([-record["center_local_xy"][0], -record["center_local_xy"][1], 0])
        draw(axis, [(detail, None)], detail.bounds, elev=43, azim=-135)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([cfg["floor_z_m"][record["from_floor"]], cfg["floor_z_m"][record["to_floor"]]])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_zlabel("")
        detail = (f"slope {np.rad2deg(record['slope_rad']):.1f} deg" if record["kind"] == "ramp"
                  else f"riser {record['riser_m']:.2f} m")
        axis.set_title(
            f"F{record['from_floor']} - F{record['to_floor']} | {record['kind']}\n"
            f"{'WITH rails' if record['railings'] else 'NO rails'} | {detail}", fontsize=11,
        )
    fig.suptitle(f"Inter-floor connections | seed {cfg['seed']} | local detail views of actual geometry", fontsize=18)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.02, top=0.88, wspace=.02, hspace=.28)
    fig.savefig(Path(output) / "connections_overview.png", dpi=135, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def render_features(output, cfg, parts_per_floor, records):
    meshes = [mesh for group in parts_per_floor for mesh in group]
    ordered = sorted(records, key=lambda record: record["floor"])
    by_id = {record["id"]: mesh for mesh, record in zip(meshes, ordered)}
    kinds = ["headroom", "slope", "step", "support", "clearance"]
    bands = ["below", "near", "above"]
    ordered = sorted(records, key=lambda record: (
        bands.index(record["threshold_band"]), kinds.index(record["kind"]), record["id"]
    ))
    rows = (len(meshes) + 4) // 5
    fig = plt.figure(figsize=(23, rows * 4.3), facecolor="#f7f9fc")
    for i, record in enumerate(ordered):
        mesh = by_id[record["id"]].copy()
        mesh.apply_translation(-np.asarray(record["center_world_xyz"]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            -np.deg2rad(cfg.get("yaw_deg", 0) + record["rotation_deg"]), [0, 0, 1]
        ))
        axis = fig.add_subplot(rows, 5, i + 1, projection="3d")
        draw(axis, [(mesh, "#8da4b9")], mesh.bounds, elev=30)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_zlabel("")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([mesh.bounds[0, 2], mesh.bounds[1, 2]])
        kind = record["kind"]
        detail = (
            f"gap {record['clearance_m']:.2f} m" if kind == "headroom" else
            f"slope {record['slope_rad']:.2f} rad" if kind == "slope" else
            f"riser {record['riser_m']:.2f} m" if kind == "step" else
            f"area proxy {record['support_fraction_proxy']:.3f}" if kind == "support" else
            f"wall distance {record['centre_to_wall_m']:.2f} m"
        )
        axis.set_title(f"{record['kind']} | {record['threshold_band']}\n{detail}", fontsize=12)
    fig.suptitle(f"Capability threshold cases | seed {cfg['seed']}\n"
                 "Local metre coordinates. Support is an area proxy, not a VST flat-cell measurement.", fontsize=17)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.02, top=0.88, wspace=.02, hspace=.28)
    fig.savefig(Path(output) / "capability_overview.png", dpi=135, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
