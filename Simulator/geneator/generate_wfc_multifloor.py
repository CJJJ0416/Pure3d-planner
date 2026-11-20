"""Official terrain-generator WFC tiles -> rough, connected, stacked-floor geometry.

WFC and mesh primitives are imported from the recorded terrain-generator source snapshot.
This wrapper adds floor masks, atria, bridges and explicit stair/landing connections.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "datasets/cache/wfc_runtime"), str(HERE / "terrain-generator")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))

import numpy as np
import trimesh

import terrain_generator.utils as upstream_utils
import terrain_generator.utils.utils as upstream_core_utils

# Route the upstream configurable cache into datasets rather than the source checkout.
upstream_utils.CACHE_DIR = str(ROOT / "datasets/cache/wfc_tiles")
upstream_core_utils.CACHE_DIR = upstream_utils.CACHE_DIR
from terrain_generator.wfc.wfc import WFCSolver
from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile
from terrain_generator.trimesh_tiles.mesh_parts.indoor_parts import create_standard_stairs
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import WallPartsCfg, StairMeshPartsCfg
from terrain_generator.trimesh_tiles.mesh_parts.rough_parts import generate_perlin_tile_configs
from terrain_generator.trimesh_tiles.patterns.pattern_generator import (
    generate_platforms, generate_ramp_parts, generate_stair_parts,
)

from visualize_wfc import render, render_connections, render_features, render_layout
from wfc_randomization import sample_scene, floor_layout
from wfc_connections import build_connection
from wfc_capability_features import build_features


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tile_library(cfg):
    """Build native tile variants sequentially, using the same API as the official example."""
    dim = tuple(cfg["tile_dim_m"])
    relief = cfg["terrain_relief_m"]
    seed = cfg["seed"]
    weights = cfg.get("tile_weights", dict(floor=.4, terrace=.6, steps=1., ramp=.9, rough=2.2))
    parts = [WallPartsCfg(name="floor", dim=dim, wall_edges=(), weight=weights["floor"])]
    parts += generate_platforms("terrace", dim, max_h=relief, weight=weights["terrace"], enable_wall=False, seed=seed)
    parts += generate_stair_parts(
        "steps", dim, total_height=relief, step_height=cfg.get("small_step_height_m", .16),
        array_shape=[12, 12], depth_num=2, enable_wall=False, weight=weights["steps"], seed=seed,
    )
    parts += generate_ramp_parts(
        "ramp", dim, total_height=relief, array_shape=[16, 16], weight=weights["ramp"], seed=seed,
    )
    n = cfg["rough_grid"]
    parts += generate_perlin_tile_configs(
        "rough", dim, weight=weights["rough"], height=relief, shape=(n, n), res=(2, 2),
        base_weight=cfg["roughness_m"], fractal_weight=cfg["roughness_m"] * 0.5, seed=seed,
    )
    tiles = {}
    for index, part in enumerate(parts):
        # Concatenate the native solids. This path does not require Blender or Ray.
        part.minimal_triangles = False
        part.floor_thickness = cfg["floor_thickness_m"]
        if getattr(part, "wall", None) is not None:
            part.wall.floor_thickness = cfg["floor_thickness_m"]
            part.wall.minimal_triangles = False
        part.load_from_cache = cfg.get("load_tile_cache", True)
        part.use_generator = cfg.get("lazy_tile_meshes", True)
        if hasattr(part, "simplify"):
            part.simplify = False
        tile = create_mesh_tile(part)
        for variant in tile.get_all_tiles(rotations=part.rotations, flips=part.flips):
            tiles[variant.name] = variant
        if (index + 1) % 8 == 0 or index + 1 == len(parts):
            print(f"Native tile templates: {index + 1}/{len(parts)}", flush=True)
    print(f"Native WFC library: {len(parts)} templates, {len(tiles)} variants", flush=True)
    return tiles


def floor_constraints(cfg, floor):
    """Open wells and fixed flat landings are architectural inputs to generation."""
    rows, columns = cfg["shape"]
    mask = floor_layout(cfg, floor) if "domain_randomization" in cfg else np.ones((rows, columns), dtype=bool)
    anchors = set()
    if floor > 0 and "floor_openings" not in cfg:
        first_row, last_row = cfg["atrium_rows"]
        first_column, last_column = cfg["atrium_columns"]
        mask[first_row:last_row, first_column:last_column] = False
    if floor == len(cfg["floor_z_m"]) - 1:
        mask[rows - cfg["top_floor_front_open_rows"]:] = False
    for bridge in cfg["bridges"]:
        if bridge["floor"] == floor:
            first, last = bridge["columns"]
            for row in range(bridge["row"], bridge["row"] + bridge.get("width_cells", 1)):
                mask[row, first:last] = True
                anchors.update((row, column) for column in range(first - 1, last + 1))
    for stair in cfg["stairs"]:
        if floor in (stair["from_floor"], stair["to_floor"]):
            if "bay_rows" in stair:
                r0, r1 = stair["bay_rows"]
                c0, c1 = stair["bay_columns"]
                mask[r0:r1, c0:c1] = False
                if "lower_landing_cell" in stair:
                    port = stair["lower_landing_cell"] if floor == stair["from_floor"] else stair["upper_landing_cell"]
                    anchors.add(tuple(port))
                else:
                    anchors.update((row, column)
                                   for row in range(r0 - 1, r1 + 1)
                                   for column in range(c0 - 1, c1 + 1))
                continue
            first, last = stair["rows"]
            column = stair["column"]
            mask[first:last, column] = False
            anchors.update((row, column) for row in range(first - 1, last + 1))
    for feature in cfg.get("capability_features", []):
        if feature["floor"] == floor:
            cell = tuple(feature["cell"])
            anchors.add(cell)
            if feature["kind"] == "support":
                mask[cell] = False
    for corridor in cfg.get("corridors", []):
        if corridor["floor"] == floor:
            anchors.update(map(tuple, corridor["cells"]))
    return mask, sorted(anchors)


def world_xy(cfg, row, column):
    rows, columns = cfg["shape"]
    dx, dy = cfg["tile_dim_m"][:2]
    return [(column - (columns - 1) / 2) * dx, ((rows - 1) / 2 - row) * dy]


def inspect_landings(cfg, floors, yaw):
    """Measure the generated mesh at the actual flat entrance/exit platforms."""
    records = []
    for floor, mesh in enumerate(floors):
        ports = [(s["id"], side, s[f"{side}_landing_cell"])
                 for s in cfg["stairs"] for side, key in [("lower", "from_floor"), ("upper", "to_floor")]
                 if s[key] == floor and f"{side}_landing_cell" in s]
        origins = []
        for _, _, cell in ports:
            center = np.array([*world_xy(cfg, *cell), cfg["floor_z_m"][floor] + .1, 1.])
            for offset in [(0, 0), (-.15, 0), (.15, 0), (0, -.15), (0, .15)]:
                point = center.copy()
                point[:2] += offset
                origins.append((yaw @ point)[:3])
        if origins:
            origins = np.array(origins)
            locations, indices, _ = mesh.ray.intersects_location(
                origins, np.tile([0., 0., -1.], (len(origins), 1)), multiple_hits=False)
            errors = np.full(len(origins), np.nan)
            errors[indices] = np.abs(locations[:, 2] - cfg["floor_z_m"][floor])
            for index, (name, side, cell) in enumerate(ports):
                values = errors[index * 5:(index + 1) * 5]
                records.append(dict(connection=name, side=side, floor=floor, cell=cell,
                                    probes=5, hits=int(np.isfinite(values).sum()),
                                    height_error_m=values.tolist()))
    return dict(method="five downward mesh probes per flat landing, 15 cm centre offsets", ports=records)


def generate_floor(cfg, floor, tiles, *, return_parts=False):
    mask, anchors = floor_constraints(cfg, floor)
    solver = WFCSolver(shape=tuple(cfg["shape"]), dimensions=2, seed=cfg["seed"] + floor * 1009)
    for tile in tiles.values():
        solver.register_tile(*tile.get_dict_tile())
    wave = solver.run(init_tiles=[("floor", cell) for cell in anchors], max_steps=10000)
    meshes = []
    components = []
    histogram = Counter()
    for row, column in np.argwhere(mask):
        name = solver.names[int(wave[row, column])]
        part = tiles[name].get_mesh().copy()
        xy = world_xy(cfg, int(row), int(column))
        # Native tiles are centred at -dim_z/2 with floor top at floor_thickness.
        z = cfg["floor_z_m"][floor] + cfg["tile_dim_m"][2] / 2 - cfg["floor_thickness_m"]
        part.apply_translation([*xy, z])
        meshes.append(part)
        if return_parts:
            family = name.split("_")[0]
            semantic = "stair" if family == "steps" else "ramp" if family == "ramp" else "floor"
            components.append(dict(mesh=part, semantic=semantic, cell=[int(row), int(column)], tile_name=name))
        histogram[name.split("_")[0]] += 1
    if return_parts:
        return trimesh.util.concatenate(meshes), wave, mask, solver.names, dict(histogram), components
    return trimesh.util.concatenate(meshes), wave, mask, solver.names, dict(histogram)


def generate_stair(cfg, stair):
    first, last = stair["rows"]
    rise = cfg["floor_z_m"][stair["to_floor"]] - cfg["floor_z_m"][stair["from_floor"]]
    run = (last - first) * cfg["tile_dim_m"][1]
    landing = 0.3
    spec = StairMeshPartsCfg.Stair(
        dim=(cfg["tile_dim_m"][0], run, rise), total_height=rise,
        n_steps=stair["steps"], step_width=stair["width_m"],
        step_depth=(run - landing) / (stair["steps"] + 1),
        height_offset=cfg["floor_thickness_m"], floor_thickness=cfg["floor_thickness_m"],
        minimal_triangles=False, add_residual_side_up=True,
    )
    mesh = create_standard_stairs(spec)
    xy = world_xy(cfg, (first + last - 1) / 2, stair["column"])
    base = cfg["floor_z_m"][stair["from_floor"]]
    mesh.apply_translation([*xy, base + rise / 2 - cfg["floor_thickness_m"]])
    foundation = trimesh.creation.box(
        [cfg["tile_dim_m"][0], run, cfg["floor_thickness_m"]],
        trimesh.transformations.translation_matrix([*xy, base - cfg["floor_thickness_m"] / 2]),
    )
    mesh = trimesh.util.concatenate([foundation, mesh])
    return mesh, dict(
        **stair, bottom_z_m=base, top_z_m=base + rise, rise_m=rise, run_m=run,
        riser_m=rise / stair["steps"], tread_depth_m=spec.step_depth,
        lower_landing_cell=[last, stair["column"]], upper_landing_cell=[first - 1, stair["column"]],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "wfc_capability_dr.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Directory for this generation's WFC mesh cache.")
    args = parser.parse_args()
    if args.cache_dir is not None:
        upstream_utils.CACHE_DIR = str(args.cache_dir.resolve())
        upstream_core_utils.CACHE_DIR = upstream_utils.CACHE_DIR
        # WFC imports the same constant by value.
        import terrain_generator.wfc.wfc as upstream_wfc
        upstream_wfc.CACHE_DIR = upstream_utils.CACHE_DIR
    cfg = json.loads(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
        if "output_root" not in cfg:
            cfg["output"] = f"datasets/wfc_multifloor_v1/scene_{args.seed}"
    if "domain_randomization" in cfg:
        cfg = sample_scene(cfg)
    if args.output:
        cfg["output"] = str(args.output)
    output = (ROOT / cfg["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    floors, stairs, floor_records, connections = [], [], [], []
    feature_parts, feature_records = build_features(cfg) if "capability_features" in cfg else (None, [])
    with (output / "wfc.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log):
            tiles = tile_library(cfg)
        print(f"Native WFC library: {len(tiles)} variants", flush=True)
        for floor, height in enumerate(cfg["floor_z_m"]):
            with redirect_stdout(log):
                mesh, wave, mask, names, histogram = generate_floor(cfg, floor, tiles)
            floors.append(mesh)
            np.save(output / f"floor_{floor}_wave.npy", wave)
            np.save(output / f"floor_{floor}_mask.npy", mask)
            floor_records.append(dict(
                floor=floor, z_m=height, tiles=int(mask.sum()), families=histogram, tile_names=names,
            ))
            print(f"Floor {floor}: z={height:g} m, {int(mask.sum())} tiles, {histogram}", flush=True)
    for spec in cfg["stairs"]:
        mesh, record = build_connection(cfg, spec) if "kind" in spec else generate_stair(cfg, spec)
        stairs.append(mesh)
        connections.append(record)
    if feature_parts is not None:
        floors = [trimesh.util.concatenate([mesh, *feature_parts[i]]) for i, mesh in enumerate(floors)]
    yaw = trimesh.transformations.rotation_matrix(np.deg2rad(cfg.get("yaw_deg", 0)), [0, 0, 1])
    for mesh in [*floors, *stairs]:
        mesh.apply_transform(yaw)
    if feature_parts is not None:
        for group in feature_parts:
            for mesh in group:
                mesh.apply_transform(yaw)
    for i, mesh in enumerate(floors):
        mesh.export(output / f"floor_{i}.obj")
    for i, mesh in enumerate(stairs):
        name = cfg["stairs"][i].get("id", f"stairs_{cfg['stairs'][i]['from_floor']}_{cfg['stairs'][i]['to_floor']}")
        mesh.export(output / f"{name}.obj")
    for record in feature_records:
        point = np.array([*record["center_local_xyz"], 1.0])
        record["center_world_xyz"] = (yaw @ point)[:3].tolist()
    for record in connections:
        if "center_local_xy" in record:
            point = np.array([*record["center_local_xy"], cfg["floor_z_m"][record["from_floor"]], 1.0])
            record["center_world_xyz"] = (yaw @ point)[:3].tolist()
            record["rotation_world_deg"] = (cfg.get("yaw_deg", 0) + record.get("rotation_deg", 0)) % 360
    combined = trimesh.util.concatenate([*floors, *stairs])
    combined.export(output / "terrain.obj")
    landing_inspection = inspect_landings(cfg, floors, yaw)
    write_json(output / "landings.report.json", landing_inspection)
    provenance = json.loads((HERE / 'TERRAIN_GENERATOR_PROVENANCE.json').read_text())
    commits = {'terrain-generator': provenance['revision']}
    scene = dict(
        seed=cfg["seed"], units="m", frame_id="map", z_axis="up",
        generator="official terrain_generator.wfc.WFCSolver with native mesh tiles",
        wfc_dimensions=2, floor_assembly="stacked floor masks with explicit stair wells and landings",
        upstream_commits=commits, generator_source=provenance,
        config=cfg, floors=floor_records, stairs=connections,
        capability_features=feature_records, grid_frame="before yaw rotation, see config.yaw_deg",
        algorithm_reference="https://github.com/mxgmn/WaveFunctionCollapse",
        bounds_world=combined.bounds.tolist(), mesh_vertices=len(combined.vertices),
        footprint=cfg.get("footprint"), landing_inspection="landings.report.json",
        mesh_faces=len(combined.faces), overview="overview.png",
        stage="geometry preview only, no PCD, VST labels or neural-network training",
        generation_s=time.perf_counter() - started,
    )
    write_json(output / "scene.json", scene)
    write_json(output / "config.resolved.json", cfg)
    if feature_records:
        write_json(output / "capability_features.json", dict(
            reference_vst=cfg["reference_vst"], features=feature_records,
            support_ratio_note="Geometric area proxy only. VST standable_ratio is measured from discrete flat-cell counts.",
            cost_scale_note="cost_barrier is a label scale, not a sampled geometry parameter.",
        ))
    render(output, cfg, floors, stairs)
    if "domain_randomization" in cfg:
        scene["layout_overview"] = "layout_overview.png"
        write_json(output / "scene.json", scene)
        render_layout(output, cfg)
        render_connections(output, cfg, stairs, connections)
        render_features(output, cfg, feature_parts, feature_records)
    print(json.dumps(dict(output=str(output), faces=len(combined.faces),
                          total_s=time.perf_counter()-started)), flush=True)


if __name__ == "__main__":
    main()
