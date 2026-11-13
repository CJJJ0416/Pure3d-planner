"""Traversability WFC generation with identity labels and indoor corridors."""
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import trimesh

from common.geneator.filter_components import SEMANTICS, COLORS, surface_union
from common.geneator.wfc_connections import box, rod
from utils.pointcloud import ROOT, read_json, write_json, workspace_path


def reserve_corridors(cfg, specification, rng):
    """Choose supported straight runs, away from wells and capability cases."""
    from wfc_randomization import floor_layout
    corridors = []
    rows, columns = cfg["shape"]
    for floor, base in enumerate(cfg["floor_z_m"]):
        free = floor_layout(cfg, floor)
        for stair in cfg["stairs"]:
            if floor in (stair["from_floor"], stair["to_floor"]):
                r0, r1 = stair["bay_rows"]
                c0, c1 = stair["bay_columns"]
                free[r0:r1, c0:c1] = False
                port = stair["lower_landing_cell"] if floor == stair["from_floor"] else stair["upper_landing_cell"]
                free[tuple(port)] = False
        for feature in cfg["capability_features"]:
            if feature["floor"] == floor:
                free[tuple(feature["cell"])] = False
        for ordinal in range(specification["per_floor"]):
            candidates = []
            for length in range(*specification["length_cells_range"]):
                for row in range(rows):
                    for column in range(columns):
                        for direction in [(0, 1), (1, 0)]:
                            cells = [(row + direction[0] * i, column + direction[1] * i) for i in range(length)]
                            if cells[-1][0] < rows and cells[-1][1] < columns and all(free[cell] for cell in cells):
                                candidates.append((cells, direction))
            cells, direction = candidates[int(rng.integers(len(candidates)))]
            tile = cfg["tile_dim_m"][0]
            width = float(rng.uniform(*specification["width_m_range"]))
            height = float(rng.uniform(*specification["height_m_range"]))
            for cell in cells:
                free[cell] = False
            corridors.append(dict(id=f"corridor_F{floor}_{ordinal}", floor=floor,
                cells=[list(cell) for cell in cells], axis="x" if direction[1] else "y",
                length_m=len(cells) * tile, width_m=width, height_m=height,
                wall_thickness_m=specification["wall_thickness_m"],
                door_width_m=specification["door_width_m"], door_height_m=specification["door_height_m"],
                doorway_offset_m=float(rng.uniform(-.2, .2) * len(cells) * tile)))
    cfg["corridors"] = corridors
    return corridors


def corridor_meshes(cfg, corridor):
    """Open ends plus a side doorway in each wall and a full ceiling."""
    from generate_wfc_multifloor import world_xy
    points = np.array([world_xy(cfg, *cell) for cell in corridor["cells"]])
    center = points.mean(0)
    width, length, height = [corridor[name] for name in ["width_m", "length_m", "height_m"]]
    thickness, door_width, door_height, door_center = [corridor[name] for name in
        ["wall_thickness_m", "door_width_m", "door_height_m", "doorway_offset_m"]]
    frame = trimesh.transformations.rotation_matrix(np.pi / 2 if corridor["axis"] == "x" else 0, [0, 0, 1])
    frame[:3, 3] = [*center, cfg["floor_z_m"][corridor["floor"]]]
    components = []
    for side in [-1, 1]:
        x = side * (width + thickness) / 2
        for a, b in [(-length / 2, door_center - door_width / 2), (door_center + door_width / 2, length / 2)]:
            components.append(("wall", box([thickness, b - a, height], [x, (a + b) / 2, height / 2])))
        components.append(("wall", box([thickness, door_width, height - door_height],
                                       [x, door_center, (height + door_height) / 2])))
    components.append(("ceiling", box([width + 2 * thickness, length, thickness], [0, 0, height + thickness / 2])))
    for semantic, mesh in components:
        mesh.apply_transform(frame)
    return components


def object_proxies(center):
    """Small movable objects, with a separate identity from building geometry."""
    x, y, z = center
    torso = trimesh.creation.icosphere(subdivisions=2)
    torso.apply_scale([.25, .16, .4])
    torso.apply_translation([x, y, z + 1.12])
    head = trimesh.creation.icosphere(subdivisions=2, radius=.15)
    head.apply_translation([x, y, z + 1.68])
    human = [torso, head]
    for side in [-1, 1]:
        human.append(rod([x + side * .12, y, z + .07], [x + side * .1, y, z + .89], .075))
        human.append(rod([x + side * .22, y, z + 1.38], [x + side * .35, y + .05, z + .87], .055))
    chair = [box([.5, .5, .07], [x, y + 1., z + .48]),
             box([.5, .06, .5], [x, y + 1.22, z + .75])]
    for dx in [-.2, .2]:
        for dy in [-.2, .2]:
            chair.append(box([.05, .05, .45], [x + dx, y + 1. + dy, z + .225]))
    return [("person_proxy", trimesh.util.concatenate(human)),
            ("chair", trimesh.util.concatenate(chair)),
            ("movable_clutter", box([.28, .35, .3], [x - .35, y + .5, z + .15]))]


def build_wfc_terrain(seed, configuration, output):
    """Use the same native template/WFC/floor/stair functions as Traversability."""
    sys.path.insert(0, str(ROOT / "common/geneator"))
    from generate_wfc_multifloor import tile_library, generate_floor, world_xy
    from wfc_randomization import sample_scene
    from wfc_connections import build_connection
    from wfc_capability_features import build_features

    template = read_json(workspace_path(configuration["scene_config"]))
    template["seed"] = seed
    cfg = sample_scene(template)
    # OBJ cache loading welds touching boxes before component identities can
    # be retained. Build each native template once with its original topology.
    cfg.update(load_tile_cache=False, lazy_tile_meshes=False)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 69321]))
    corridors = reserve_corridors(cfg, configuration["corridors"], rng)
    parts, roles, floors, connections = [], [], [], []

    def add(name, semantic, mesh, target=True):
        instance = len(roles) + 1
        role = dict(instance_id=instance, name=name, semantic=semantic,
                    semantic_id=SEMANTICS[semantic], target_role=int(target),
                    color_rgb=COLORS[SEMANTICS[semantic]].tolist())
        roles.append(role)
        parts.append(dict(mesh=mesh, instance_id=instance, semantic_id=role["semantic_id"], target_role=int(target)))
        return instance

    with (output / "wfc.log").open("w") as log, redirect_stdout(log):
        from alive_progress import config_handler
        config_handler.set_global(file=log, force_tty=False)
        tiles = tile_library(cfg)
        for floor, height in enumerate(cfg["floor_z_m"]):
            _, wave, mask, names, histogram, components = generate_floor(cfg, floor, tiles, return_parts=True)
            np.save(output / f"floor_{floor}_wave.npy", wave)
            np.save(output / f"floor_{floor}_mask.npy", mask)
            for part in components:
                add(f"F{floor}_tile_{part['cell'][0]}_{part['cell'][1]}_{part['tile_name']}", part["semantic"], part["mesh"])
            floors.append(dict(floor=floor, z_m=height, tiles=int(mask.sum()), families=histogram, tile_names=names))
    for specification in cfg["stairs"]:
        _, record, components = build_connection(cfg, specification, return_parts=True)
        connections.append(record)
        for index, part in enumerate(components):
            add(f"{specification['id']}_{part['semantic']}_{index}", part["semantic"], part["mesh"])
    _, features, components = build_features(cfg, return_parts=True)
    for index, part in enumerate(components):
        add(f"{part['feature_id']}_{index}", part["semantic"], part["mesh"])
    for corridor in corridors:
        for index, (semantic, mesh) in enumerate(corridor_meshes(cfg, corridor)):
            add(f"{corridor['id']}_{semantic}_{index}", semantic, mesh)
        center = [*np.mean([world_xy(cfg, *cell) for cell in corridor["cells"]], axis=0),
                  cfg["floor_z_m"][corridor["floor"]]]
        object_frame = trimesh.transformations.rotation_matrix(np.pi / 2 if corridor["axis"] == "x" else 0, [0, 0, 1])
        object_frame[:3, 3] = center
        corridor["local_to_scene"] = object_frame.tolist()
        object_rng = np.random.default_rng(np.random.SeedSequence([seed, 7193, corridor["floor"]]))
        boundary = float(object_rng.uniform(-.15, .15) * corridor["length_m"])
        intervals = [[-corridor["length_m"] / 2, boundary], [boundary, corridor["length_m"] / 2]]
        static_zone = int(object_rng.integers(2))
        corridor["static_interval_m"] = intervals[static_zone]
        corridor["motion_interval_m"] = intervals[1 - static_zone]
        corridor["object_instances"] = {}
        slots = object_rng.permutation([.15, .5, .85])
        for ordinal, (semantic, mesh) in enumerate(object_proxies([0., 0., 0.])):
            base = np.r_[mesh.bounds[:, :2].mean(0), mesh.bounds[0, 2]]
            mesh.apply_translation(-base)
            scale = float(object_rng.uniform(.85, 1.1))
            mesh.apply_scale(scale)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(object_rng.uniform(-.4, .4), [0, 0, 1]))
            radius = np.linalg.norm(mesh.vertices[:, :2], axis=1).max()
            a, b = corridor["static_interval_m"]
            y = a + radius + .05 + slots[ordinal] * (b - a - 2 * radius - .1)
            x = object_rng.uniform(-1, 1) * max(0., corridor["width_m"] / 2 - radius - .05)
            mesh.apply_translation([x, y, 0.])
            mesh.apply_transform(object_frame)
            instance = add(f"{corridor['id']}_{semantic}", semantic, mesh)
            corridor["object_instances"][str(SEMANTICS[semantic])] = instance
    frame = trimesh.transformations.rotation_matrix(np.deg2rad(cfg["yaw_deg"]), [0, 0, 1])
    for part in parts:
        part["mesh"].apply_transform(frame)
    print(f"WFC | shape={cfg['shape']} | floors={len(floors)} | connections={len(connections)} | corridors={len(corridors)} | components={len(parts)}", flush=True)
    target, target_metadata = surface_union([part for part in parts if part["semantic_id"] < 8])
    other, other_metadata = surface_union([part for part in parts if part["semantic_id"] >= 8])
    combined = trimesh.util.concatenate([target, other])
    metadata = {key: np.concatenate([target_metadata[key], other_metadata[key]]) for key in target_metadata}
    metadata["face_id"] = np.arange(len(combined.faces), dtype=np.int64)
    for role in roles:
        role["faces"] = int((metadata["instance_id"] == role["instance_id"]).sum())
    write_json(output / "wfc_config.resolved.json", cfg)
    write_json(output / "corridors.json", dict(corridors=corridors, transform=frame.tolist()))
    record = dict(seed=seed, family="wfc_terrain", coordinate_frame="map", frame_id="map", units="m", z_axis="up",
        generator="Traversability native WFC multi-floor template plus reserved indoor corridors",
        scene_config=configuration["scene_config"], floors=floors, stairs=connections, corridors=corridors,
        capability_features=features, footprint=cfg["footprint"], transform=frame.tolist(),
        degenerate_faces_removed={"target": target.metadata["degenerate_faces_removed"],
                                  "static_objects": other.metadata["degenerate_faces_removed"]},
        building_faces=len(target.faces), static_object_faces=len(other.faces),
        target_faces=len(combined.faces), bounds_world=combined.bounds.tolist())
    return combined, target, other, metadata, roles, record
