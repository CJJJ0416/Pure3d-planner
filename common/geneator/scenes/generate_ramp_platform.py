#!/usr/bin/env python3
"""Generate one solid platform and a smooth connecting ramp using scene primitives."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import trimesh
from stair_nav.core.io import read_json, sha256, write_json, write_pcd
from stair_nav.scenes.geometry import box, sample_surface, support_heights
from stair_nav.scenes.stair_structure import add_prism, ramp_profile


def create_scene(cfg):
    width, depth = map(float, cfg["ground_size_m"])
    pw, pd = map(float, cfg["platform_size_m"])
    height, run, rw, slab, spacing, transition = map(float, (
        cfg["platform_height_m"], cfg["ramp_run_m"], cfg["ramp_width_m"],
        cfg["slab_thickness"], cfg["sample_spacing"], cfg["transition_length_m"]))
    dimensions = [width, depth, pw, pd, height, run, rw, slab, spacing, transition]
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        raise ValueError("All dimensions must be finite and positive")
    placement = cfg.get("ramp_placement", "front")
    if placement not in ("front", "right_side"):
        raise ValueError("ramp_placement must be front or right_side")
    side = placement == "right_side"
    landing = float(cfg.get("landing_depth_m", rw)) if side else 0.
    if side and (not np.isfinite(landing) or landing < rw or abs(run-pd) > 1e-8):
        raise ValueError("Side ramp requires run=platform depth and landing depth >= ramp width")
    if rw > pw or pw >= width or (not side and max(run+.8, pd) >= depth / 2):
        raise ValueError("Platform and ramp must fit inside the ground with a surrounding margin")
    if side and (pw/2+rw >= width/2 or run+landing >= depth/2):
        raise ValueError("Side ramp and its turning landing must fit inside the ground")
    if spacing > cfg["vst"]["resolution"] / 2:
        raise ValueError("Surface sampling must be at most half the VST resolution")

    count = max(2, int(np.ceil(run / .25)))
    ramp_x, ramp_y = (pw/2+rw/2, 0.) if side else (0., -run)
    flight = dict(id=0, x=ramp_x, y=ramp_y, end_y=ramp_y+run, direction=1,
                  width=rw, length=run, n_steps=count, step_depth=run/count,
                  riser=height/count, base_z=0., top_z=height, landing_depth=0.,
                  from_floor=0, to_floor=1, construction="ramp")
    q, z, blend, slope = ramp_profile(flight, transition)
    flight.update(profile_q_m=q.tolist(), profile_z_m=z.tolist(),
                  transition_length_m=blend, maximum_slope_rad=slope)
    scene = dict(schema_version=1, name=cfg["name"], seed=cfg["seed"],
                 layout="ramp_platform", architecture=dict(family="ramp_platform", label="侧边坡道平台" if side else "坡道平台"),
                 capabilities=dict(random_map=False),
                 frame_id="map", units="m", z_axis="up", yaw=0.,
                 width=width, depth=depth, height=height, slab_thickness=slab,
                 config=cfg, solids=[], support_surfaces=[], stair_flights=[flight],
                 stair_structure=cfg["stair_structure"], rooms=[], relief=[],
                 landings=[], floor_features=[],
                 floors=[dict(id=0, z=0., openings=[], regions=[[-width/2, -depth/2, width/2, depth/2]]),
                         dict(id=1, z=height, openings=[], regions=[[-pw/2, 0., pw/2, pd]])],
                 floor_graph=dict(nodes=[dict(id=0, z=0.), dict(id=1, z=height)],
                                  edges=[dict(**{"from": 0, "to": 1}, type="ramp", bidirectional=True)]),
                 geometry_backend="glocal2-nav box/add_prism + existing C1 ramp_profile")
    meshes = []

    def solid(lo, hi, kind):
        meshes.append(box(lo, hi))
        scene["solids"].append(dict(lo=lo, hi=hi, kind=kind))
        scene["support_surfaces"].append(dict(kind=kind,
            vertices_xy=[[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
            height_plane=[0., 0., hi[2]]))

    solid([-width/2, -depth/2, -slab], [width/2, depth/2, 0.], "lower_floor")
    solid([-pw/2, 0., -slab], [pw/2, pd, height], "platform")
    if side:
        solid([pw/2-rw, run, -slab], [pw/2+rw, run+landing, height], "turning_landing")
        scene["floors"][1]["regions"].append([pw/2-rw, run, pw/2+rw, run+landing])
        scene["landings"].append(dict(lo=[pw/2-rw, run, height-slab],
                                     hi=[pw/2+rw, run+landing, height], z=height))
    for index, (a, b, za, zb) in enumerate(zip(q[:-1], q[1:], z[:-1], z[1:])):
        ya, yb = ramp_y+a, ramp_y+b
        gradient = (zb-za)/(b-a)
        top = [0., float(gradient), float(za-gradient*ya)]
        polygon = [[ramp_x-rw/2, ya], [ramp_x+rw/2, ya],
                   [ramp_x+rw/2, yb], [ramp_x-rw/2, yb]]
        add_prism(scene, meshes, polygon, top, [0., 0., -slab], "navigation_ramp", segment=index)
        scene["support_surfaces"].append(dict(kind="navigation_ramp", vertices_xy=polygon, height_plane=top))

    start, goal = [ramp_x, ramp_y-.8, 0.], [0., pd/2, height]
    scene["queries"] = [dict(id="ascent", start_support_xyz=start, goal_support_xyz=goal,
                             start_yaw=float(np.pi/2), required_transition="ramp", start_floor=0, goal_floor=1),
                        dict(id="descent", start_support_xyz=goal, goal_support_xyz=start,
                             start_yaw=float(-np.pi/2), required_transition="ramp", start_floor=1, goal_floor=0)]
    mesh = trimesh.boolean.union(meshes, engine="manifold")
    if not mesh.is_watertight or not mesh.is_volume or len(mesh.split()) != 1:
        raise ValueError("Expected one connected, watertight positive-volume solid")
    return scene, mesh


def face_colors(mesh, cfg):
    centers = mesh.triangles_center
    if cfg.get("ramp_placement") == "right_side":
        ramp = ((centers[:,0] > cfg["platform_size_m"][0]/2+1e-7)
                & (centers[:,1] < cfg["ramp_run_m"]-1e-7) & (centers[:,2] > .001))
    else:
        ramp = (centers[:, 1] < 0) & (centers[:, 2] > .001)
    platform = ~ramp & (centers[:,2] > .001)
    colors = np.tile([.78, .81, .84, 1.], (len(centers), 1))
    colors[ramp] = [.93, .58, .19, 1.]
    colors[platform] = [.20, .48, .72, 1.]
    return colors


def previews(output, scene, mesh, cfg):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Rectangle
    import plotly.graph_objects as go

    colors = face_colors(mesh, cfg)
    run, height = cfg["ramp_run_m"], cfg["platform_height_m"]
    pw, pd = cfg["platform_size_m"]
    flight = scene["stair_flights"][0]
    y = np.asarray(flight["profile_q_m"])+flight["y"]
    z = np.asarray(flight["profile_z_m"])
    side_ramp = cfg.get("ramp_placement") == "right_side"
    xmin, xmax = -pw/2-1.1, (pw/2+cfg["ramp_width_m"] if side_ramp else pw/2)+1.1
    ymin, ymax = flight["y"]-1., (run+cfg.get("landing_depth_m", cfg["ramp_width_m"]) if side_ramp else pd)+.6
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="white")
    ax = fig.add_subplot(121, projection="3d")
    ax.computed_zorder = False
    # Matplotlib sorts whole polygons; draw a separate ground plane so large
    # ground triangles cannot cover the raised platform in the static preview.
    gx, gy = np.meshgrid([xmin, xmax], [ymin, ymax])
    ax.plot_surface(gx, gy, np.full_like(gx, -.005), color="#dce3e9", shade=False, zorder=0)
    raised = mesh.triangles[:,:,2].max(axis=1) > .001
    ax.add_collection3d(Poly3DCollection(mesh.triangles[raised], facecolors=colors[raised],
        edgecolors=(.2, .25, .3, .04), linewidths=.15, zorder=1))
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(-.18, height+.35),
           xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]", title="Solid platform + smooth ramp")
    ax.set_box_aspect((xmax-xmin, ymax-ymin, height+.53))
    ax.set_proj_type("ortho")
    ax.view_init(elev=26, azim=-52)
    side = fig.add_subplot(122)
    slab = cfg["slab_thickness"]
    side.fill_between(y, -slab, z, color=colors[np.argmax(colors[:, 0])], label="Ramp")
    side.add_patch(Rectangle((run if side_ramp else 0, -slab),
                            cfg.get("landing_depth_m", cfg["ramp_width_m"]) if side_ramp else pd,
                            height+slab, color="#347bb8", label="Turning landing" if side_ramp else "Platform"))
    side.axhline(0, color="#718096", lw=1)
    side.set(xlim=(ymin,ymax), ylim=(-.25, height+.55), xlabel="Y [m]", ylabel="Z [m]",
             title=f"Platform {pw:g} x {pd:g} m | height {height:g} m\nRamp run {run:g} m | width {cfg['ramp_width_m']:g} m | max slope {np.degrees(flight['maximum_slope_rad']):.2f} deg")
    side.set_aspect("equal")
    side.grid(alpha=.18)
    side.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output/"preview.png", dpi=160)
    plt.close(fig)

    rgb = [f"rgb({round(r*255)},{round(g*255)},{round(b*255)})" for r,g,b,_ in colors]
    vertices, faces = mesh.vertices, mesh.faces
    figure = go.Figure(go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2], facecolor=rgb, flatshading=True,
        lighting=dict(ambient=.65, diffuse=.8, specular=.08),
        hovertemplate="X %{x:.2f} m<br>Y %{y:.2f} m<br>Z %{z:.2f} m<extra></extra>"))
    figure.update_layout(title=f"{'侧边坡道平台' if side_ramp else '带斜坡的平台'} · 高 {height:g} m · 顶面 {pw:g} × {pd:g} m · 斜坡宽 {cfg['ramp_width_m']:g} m / 水平长 {run:g} m",
        paper_bgcolor="#f5f7fa", margin=dict(l=10,r=10,b=10,t=70),
        scene=dict(aspectmode="data", xaxis_title="X / m", yaxis_title="Y / m", zaxis_title="Z / m",
                   xaxis=dict(range=[xmin,xmax]), yaxis=dict(range=[ymin,ymax]),
                   zaxis=dict(range=[-.18,height+.5]), camera=dict(eye=dict(x=1.25,y=-1.65,z=1.15))))
    html = figure.to_html(include_plotlyjs=True, full_html=True,
                          config=dict(displaylogo=False, responsive=True, scrollZoom=True))
    html = html.replace("<head>", "<head><meta charset='utf-8'><title>带斜坡的平台</title>"
                        "<style>html,body{margin:0;width:100%;height:100%;overflow:hidden}</style>", 1)
    (output/"index.html").write_text(html, encoding="utf-8")


def generate(cfg, output):
    scene, mesh = create_scene(cfg)
    output.mkdir(parents=True, exist_ok=False)
    mesh.export(output/"terrain.obj")
    mesh.export(output/"terrain.stl")
    mesh.visual.face_colors = (face_colors(mesh, cfg)*255).astype(np.uint8)
    mesh.export(output/"terrain.glb")
    points = sample_surface(mesh, cfg["sample_spacing"])
    points = points[points[:,2] >= -1e-6]
    write_pcd(output/"terrain.pcd", points)
    scene.update(point_count=len(points), bounds_world=mesh.bounds.tolist(),
                 mesh_watertight=bool(mesh.is_watertight), pcd_ground_clip_z=0.,
                 pcd_sha256=sha256(output/"terrain.pcd"))
    write_json(output/"scene.json", scene)
    write_json(output/"config.resolved.json", cfg)

    loaded = trimesh.load(output/"terrain.obj", force="mesh", process=True)
    flight = scene["stair_flights"][0]
    side = cfg.get("ramp_placement") == "right_side"
    end_y = cfg["ramp_run_m"]+cfg.get("landing_depth_m", cfg["ramp_width_m"]) if side else cfg["platform_size_m"][1]
    ys = np.linspace(flight["y"]-.5, end_y-.2, 401)
    xy = np.column_stack([np.full(len(ys), flight["x"]), ys])
    if side:
        turning = np.column_stack([np.linspace(flight["x"], cfg["platform_size_m"][0]/2-cfg["ramp_width_m"]/2, 101),
                                   np.full(101,cfg["ramp_run_m"]+cfg["landing_depth_m"]/2)])
        entering = np.column_stack([np.full(101,turning[-1,0]),
                                    np.linspace(turning[-1,1],cfg["platform_size_m"][1]-.4,101)])
        xy = np.vstack([xy,turning,entering])
    expected = np.array([max(support_heights(x, y, scene)) for x,y in xy])
    # Reuse the existing triangle support query; no optional rtree dependency.
    sys.path.insert(0, str(ROOT/"isaac_training/src"))
    from isaac_nav.simulation.rom import MeshSupport
    support = MeshSupport(output/"terrain.obj")
    measured = np.array([max(support.heights(point)) for point in xy])
    error = float(np.max(np.abs(expected-measured)))
    if error > 1e-6:
        raise ValueError(f"Mesh/analytic support mismatch: {error}")
    report = dict(mesh_watertight=bool(loaded.is_watertight), positive_volume=bool(loaded.is_volume),
                  connected_components=len(loaded.split()), support_queries=len(xy), ramp_placement=cfg.get("ramp_placement","front"),
                  max_support_error_m=error, platform_height_m=cfg["platform_height_m"],
                  maximum_slope_deg=float(np.degrees(scene["stair_flights"][0]["maximum_slope_rad"])),
                  mesh_vertices=len(mesh.vertices), mesh_triangles=len(mesh.faces),
                  point_count=len(points), terrain_obj_sha256=sha256(output/"terrain.obj"),
                  scope="Geometry and mesh/support agreement; no robot or policy execution")
    write_json(output/"geometry_validation.json", report)
    previews(output, scene, mesh, cfg)
    preset = "ramp_platform_side.json" if side else "ramp_platform.json"
    placement_text = "斜坡沿平台右侧平行上升，坡顶经转接平台拐入主平台，形成 L 形通路。" if side else "斜坡从平台正面接入。"
    (output/"README.md").write_text(f"""# 带斜坡的平台

尺寸单位为米，Z 向上。平台顶面 {cfg['platform_size_m'][0]:g} × {cfg['platform_size_m'][1]:g}，高 {cfg['platform_height_m']:g}；斜坡水平长 {cfg['ramp_run_m']:g}、宽 {cfg['ramp_width_m']:g}，两端采用现有生成器的平滑过渡。平台和坡底均为实心，无栏杆。
{placement_text}

`terrain.obj` / `terrain.stl` / `terrain.glb` 是模型；`terrain.pcd` 可用于 VST 建图；`scene.json` 保存支撑面和上下坡查询；`index.html` 可离线旋转查看，`preview.png` 为预览。

复建：`.venv/bin/python common/geneator/scenes/generate_ramp_platform.py --config configs/{preset} --output outputs/新的目录`

几何检查见 `geometry_validation.json`。未运行机器人或训练策略。
""", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/ramp_platform.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(read_json(args.config), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
