"""Headless mesh overviews shared by generation and existing-scene export."""
import os
from pathlib import Path

import numpy as np


FACILITIES = {
    "curved_bridge": ("Curved bridge", "#3478b8"),
    "ramp": ("Ramp", "#ed923c"),
    "open_stairs": ("Open stairs", "#49a66d"),
    "switchback_stairs": ("Switchback stairs", "#cf615d"),
    "spiral_stairs": ("Spiral stairs", "#9872b5"),
    "raised_platform": ("Raised platform", "#b58a60"),
    "underpass": ("Underpass", "#cf79ac"),
    "obstacles": ("Obstacles", "#5ba6b2"),
    "low_walls": ("Low walls", "#b2ac43"),
}


def _pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(__file__).resolve().parents[4] / "datasets/cache/matplotlib")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_overview(output, scene, mesh, split):
    """Save one overview.png from the actual metric mesh, without a display server."""
    plt = _pyplot()
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(output)
    centers = mesh.triangles_center
    face_colors = np.tile(to_rgb("#e2e8ed"), (len(mesh.faces), 1))
    labels, legend = [], []
    for number, (name, region) in enumerate(scene["training_regions"].items(), 1):
        title, color = FACILITIES[name]
        lower, upper = np.asarray(region["structure_bounds_xy"]).reshape(2, 2)
        inside = ((centers[:, :2] >= lower) & (centers[:, :2] <= upper)).all(1)
        structure = inside & (centers[:, 2] > 0.01)
        face_colors[structure] = to_rgb(color)
        labels.append((number, (lower + upper) / 2, centers[structure, 2].max() + 0.2))
        legend.append(Patch(facecolor=color, label=f"{number}. {title}"))
    light = np.array([-0.35, -0.45, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.65 + 0.35 * np.maximum(mesh.face_normals @ light, 0)
    face_colors *= shade[:, None]
    fig = plt.figure(figsize=(14, 10), facecolor="#f7f9fc")
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.set_facecolor("#f7f9fc")
    # Orthographic back-face culling avoids the floor underside being drawn
    # through its top by mplot3d's triangle-centroid painter ordering.
    camera = np.array([np.cos(np.pi / 4) * np.cos(-np.pi / 3),
                       np.cos(np.pi / 4) * np.sin(-np.pi / 3), np.sin(np.pi / 4)])
    visible = mesh.face_normals @ camera > 0
    floor = mesh.triangles[:, :, 2].max(1) <= 0.01
    for order, mask in enumerate((visible & floor, visible & ~floor)):
        ax.add_collection3d(Poly3DCollection(
            mesh.triangles[mask], facecolors=face_colors[mask],
            edgecolors=(0.1, 0.15, 0.2, 0.10), linewidths=0.12, zorder=order,
        ))
    for number, xy, height in labels:
        ax.text(
            *xy, height, str(number), ha="center", va="bottom", fontsize=10,
            color="#152337", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=3,
        )
    lower, upper = mesh.bounds
    span = upper - lower
    ax.set(
        xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
        zlim=(lower[2], upper[2] + 0.5),
        xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)",
    )
    ax.set_box_aspect((span[0], span[1], (span[2] + 0.5) * 2.2))
    ax.set_zticks([0, scene["height"]])
    ax.set_zticklabels(["0", f"{scene['height']:.2f}"])
    ax.set_proj_type("ortho")
    ax.view_init(elev=45, azim=-60)
    cfg = scene["config"]
    fig.suptitle(
        f"Terrain overview | seed {scene['seed']} | {split}\n"
        f"{scene['width']:g} x {scene['depth']:g} m | deck {scene['height']:.2f} m | "
        f"bridge width {cfg['bridge_width_m']:.2f} m | {len(labels)} facilities",
        fontsize=16, color="#152337", y=0.97,
    )
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.015, "Actual scene mesh. Visual Z scale 2.2x. Coordinates remain in metres.",
             ha="center", fontsize=9, color="#526277")
    fig.subplots_adjust(left=0, right=1, bottom=0.14, top=0.92)
    path = output / "overview.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_contact_sheet(output, scenes):
    """A single index image for the scene-specific overview files."""
    plt = _pyplot()
    columns = 3
    rows = (len(scenes) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 7, rows * 5.3), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, scene in zip(axes.flat, scenes):
        axis.imshow(plt.imread(Path(output) / scene["overview"]))
        axis.set_title(f"{scene['split']} | seed {scene['seed']}", fontsize=14)
    fig.suptitle("Generated terrain scenes", fontsize=20)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, hspace=0.03, wspace=0.02)
    path = Path(output) / "overview_all.png"
    fig.savefig(path, dpi=140, facecolor="#f7f9fc")
    plt.close(fig)
    return path
