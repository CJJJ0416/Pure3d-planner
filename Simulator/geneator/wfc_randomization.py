"""Seeded geometry sampling around VST ability thresholds."""
from copy import deepcopy

import numpy as np


def sample_scene(template):
    cfg = deepcopy(template)
    dr, ref = cfg["domain_randomization"], cfg["reference_vst"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 41027]))
    draw = lambda name: float(rng.uniform(*dr[name]))
    n_floors = int(rng.integers(dr["floor_count"][0], dr["floor_count"][1] + 1))
    rows = int(rng.integers(dr["grid_rows"][0], dr["grid_rows"][1] + 1))
    columns = int(rng.integers(dr["grid_columns"][0], dr["grid_columns"][1] + 1))
    tile = draw("tile_size_m")
    reference_shape, reference_tile = [rows, columns], tile
    area_ratio = dr.get("area_ratio", 1.0)
    rows, columns = np.rint(np.array([rows, columns]) * np.sqrt(area_ratio)).astype(int)
    # Keep square WFC modules and the exact requested footprint area. Stair
    # widths, risers, railings and robot thresholds remain in physical metres.
    tile *= np.sqrt(area_ratio * np.prod(reference_shape) / (rows * columns))
    rows, columns = int(rows), int(columns)
    spacing = rng.uniform(*dr["floor_spacing_m"], size=n_floors - 1)
    cfg.update(
        output=f"{cfg['output_root']}/scene_{cfg['seed']}", shape=[rows, columns],
        tile_dim_m=[tile, tile, 2.0], floor_z_m=[0.0, *np.cumsum(spacing).tolist()],
        floor_thickness_m=draw("floor_thickness_m"), roughness_m=draw("roughness_m"),
        terrain_relief_m=draw("terrain_relief_m"), yaw_deg=draw("yaw_deg"),
        small_step_height_m=float(ref["step_max"] * rng.uniform(0.6, 1.2)),
        tile_weights=dict(zip(
            ["floor", "terrace", "steps", "ramp", "rough"],
            map(float, rng.uniform([0.3, 0.35, 0.6, 0.6, 0.8], [1.1, 1.0, 1.8, 1.7, 2.7])),
        )),
    )
    cfg["footprint"] = dict(
        reference_shape=reference_shape, reference_tile_m=reference_tile,
        reference_area_m2=float(np.prod(reference_shape) * reference_tile ** 2),
        area_m2=float(rows * columns * tile ** 2), area_ratio=area_ratio,
        size_xy_m=[columns * tile, rows * tile],
    )
    cfg["floor_openings"], cfg["bridges"] = [], []
    cfg["top_floor_front_open_rows"] = 0
    cfg["stairs"] = []
    kinds = ["straight", "straight", "switchback", "spiral", "ramp"]
    for pair in range(n_floors - 1):
        for index, kind in enumerate(kinds):
            band = list(dr["bands"])[(pair + index) % len(dr["bands"])]
            factor = float(rng.uniform(*dr["bands"][band]))
            cfg["stairs"].append(dict(
                id=f"F{pair}_F{pair+1}_{kind}_{index}", kind=kind,
                from_floor=pair, to_floor=pair + 1,
                width_m=draw("stair_width_m"), target_riser_m=ref["step_max"] * factor,
                tread_depth_m=draw("tread_depth_m"),
                railings=(index == 1 if kind == "straight" else bool((pair + index + cfg["seed"]) % 2)),
                railing_height_m=draw("railing_height_m"),
                turns=float(rng.choice(dr["spiral_turns"])),
                ramp_layout="straight" if pair % 2 == 0 else "switchback",
                threshold_band=band,
            ))
    place_connections(cfg, rng)
    place_openings(cfg, rng)
    cfg["capability_features"] = sample_features(cfg, rng)
    cfg["randomization_seed"] = cfg["seed"]
    return cfg


def floor_layout(cfg, floor):
    rows, columns = cfg["shape"]
    mask = np.ones((rows, columns), dtype=bool)
    if "floor_openings" in cfg:
        for opening in cfg["floor_openings"]:
            if opening["floor"] == floor:
                r0, r1 = opening["rows"]
                c0, c1 = opening["columns"]
                mask[r0:r1, c0:c1] = False
    elif floor > 0:
        r0, r1 = cfg["atrium_rows"]
        c0, c1 = cfg["atrium_columns"]
        mask[r0:r1, c0:c1] = False
    if floor == len(cfg["floor_z_m"]) - 1:
        mask[rows - cfg["top_floor_front_open_rows"]:] = False
    for bridge in cfg["bridges"]:
        if bridge["floor"] == floor:
            c0, c1 = bridge["columns"]
            mask[bridge["row"]:bridge["row"] + bridge["width_cells"], c0:c1] = True
    return mask


def place_connections(cfg, rng):
    """Pack all requested wells and their flat entrance/exit cells jointly.

    Bit sets represent occupied cells. Ports can share landings, while a well
    cannot cover another well or a landing needed on the same floor.
    """
    layout = cfg["domain_randomization"]["connection_layout"]
    rows, columns = cfg["shape"]
    border = layout["edge_margin_cells"]
    rotations = layout["rotations_deg"]
    candidates = []
    for spec in cfg["stairs"]:
        height, width = layout["bay_cells_by_kind"][spec["kind"]]
        options = []
        for angle in rotations:
            h, w = (height, width) if angle % 180 == 0 else (width, height)
            radians = np.deg2rad(angle)
            rotate = np.array([[np.cos(radians), -np.sin(radians)],
                               [np.sin(radians), np.cos(radians)]])
            for r0 in range(border, rows - h - border + 1):
                for c0 in range(border, columns - w - border + 1):
                    r1, c1 = r0 + h, c0 + w
                    # Coordinates here are in tile units, with +Y toward -row.
                    half_run = height / 2 + .5
                    same_side = spec["kind"] == "switchback" or (
                        spec["kind"] == "ramp" and spec["ramp_layout"] == "switchback")
                    offset = (spec["width_m"] + .25) / (2 * cfg["tile_dim_m"][0])
                    local = [[-offset if same_side else 0, -half_run],
                             [offset if same_side else 0, -half_run if same_side else half_run]]
                    ports = []
                    for x, y in np.asarray(local) @ rotate.T:
                        row = int(np.floor((r0 + r1 - 1) / 2 - y + .5 + 1e-8))
                        column = int(np.floor((c0 + c1 - 1) / 2 + x + .5 + 1e-8))
                        ports.append([row, column])
                    bits = sum(1 << (r * columns + c) for r in range(r0, r1) for c in range(c0, c1))
                    options.append((r0, r1, c0, c1, angle, ports, bits,
                                    *[1 << (r * columns + c) for r, c in ports]))
        rng.shuffle(options)
        candidates.append(options)
    occupied = [0] * len(cfg["floor_z_m"])
    landings = [0] * len(occupied)
    placements = {}

    def pack(remaining):
        if not remaining:
            return True
        available = {}
        for index in remaining:
            pair = cfg["stairs"][index]["from_floor"]
            blocked = occupied[pair] | occupied[pair + 1] | landings[pair] | landings[pair + 1]
            available[index] = [item for item in candidates[index]
                                if not (item[6] & blocked or item[7] & occupied[pair]
                                        or item[8] & occupied[pair + 1])]
        index = min(available, key=lambda i: len(available[i]))
        pair = cfg["stairs"][index]["from_floor"]
        for item in available[index]:
            previous = occupied[pair:pair + 2], landings[pair:pair + 2]
            occupied[pair] |= item[6]
            occupied[pair + 1] |= item[6]
            landings[pair] |= item[7]
            landings[pair + 1] |= item[8]
            placements[index] = item
            if pack([i for i in remaining if i != index]):
                return True
            occupied[pair:pair + 2], landings[pair:pair + 2] = previous
        return False

    if not pack(list(range(len(cfg["stairs"])))):
        raise ValueError("Requested stair wells and landing ports do not fit this footprint.")
    for index, (r0, r1, c0, c1, angle, ports, *_) in placements.items():
        cfg["stairs"][index].update(bay_rows=[r0, r1], bay_columns=[c0, c1],
                                    rotation_deg=angle, lower_landing_cell=ports[0],
                                    upper_landing_cell=ports[1])
    cfg["connection_placement"] = "whole-map compact wells, independent rotation, explicit flat landing ports"


def place_openings(cfg, rng):
    """Add upper-floor voids and a one-cell bridge outside the reserved wells."""
    rows, columns = cfg["shape"]
    for floor in range(1, len(cfg["floor_z_m"])):
        free = np.ones((rows, columns), dtype=bool)
        for stair in cfg["stairs"]:
            if floor in (stair["from_floor"], stair["to_floor"]):
                r0, r1 = stair["bay_rows"]
                c0, c1 = stair["bay_columns"]
                free[r0:r1, c0:c1] = False
                port = stair["lower_landing_cell"] if floor == stair["from_floor"] else stair["upper_landing_cell"]
                free[tuple(port)] = False
        options = [(r, c) for r in range(1, rows - 2) for c in range(1, columns - 3)
                   if free[r:r + 2, c - 1:c + 4].all()]
        if options:
            row, column = options[int(rng.integers(len(options)))]
            cfg["floor_openings"].append(dict(floor=floor, rows=[row, row + 2], columns=[column, column + 3]))
            cfg["bridges"].append(dict(floor=floor, row=row, columns=[column, column + 3], width_cells=1))


def sample_features(cfg, rng):
    dr, ref = cfg["domain_randomization"], cfg["reference_vst"]
    candidates = []
    for floor in range(len(cfg["floor_z_m"])):
        free = floor_layout(cfg, floor)
        free[0] = free[-1] = False
        free[:, 0] = free[:, -1] = False
        for stair in cfg["stairs"]:
            if floor in (stair["from_floor"], stair["to_floor"]):
                r0, r1 = stair["bay_rows"]
                c0, c1 = stair["bay_columns"]
                free[r0:r1, c0:c1] = False
                port = stair["lower_landing_cell"] if floor == stair["from_floor"] else stair["upper_landing_cell"]
                free[tuple(port)] = False
        for bridge in cfg["bridges"]:
            if bridge["floor"] == floor:
                c0, c1 = bridge["columns"]
                free[bridge["row"]:bridge["row"] + bridge["width_cells"], c0 - 1:c1 + 1] = False
        candidates.extend((floor, int(row), int(column)) for row, column in np.argwhere(free))
    rng.shuffle(candidates)
    features = []
    kinds = ["headroom", "slope", "step", "support", "clearance"]
    window = cfg["support_reference"]["resolution_m"] * cfg["support_reference"]["kernel_cells"]
    for kind in kinds:
        for band, limits in dr["bands"].items():
            for repeat in range(dr.get("feature_cases_per_kind", {}).get(kind, dr["feature_cases_per_band"])):
                floor, row, column = candidates.pop()
                factor = float(rng.uniform(*limits))
                feature = dict(
                    id=f"{kind}_{band}_{repeat}", kind=kind, threshold_band=band,
                    floor=floor, cell=[row, column], factor=factor,
                    rotation_deg=int(rng.choice([0, 90, 180, 270])),
                )
                if kind == "headroom":
                    reference = ref["interval_min"] if band == "below" else ref["interval_free"]
                    feature.update(clearance_m=reference * factor, width_m=float(rng.uniform(0.85, 1.5)))
                elif kind == "slope":
                    feature.update(slope_rad=ref["slope_max"] * factor, width_m=float(rng.uniform(0.8, 1.6)))
                elif kind == "step":
                    feature.update(riser_m=ref["step_max"] * factor,
                                   tread_depth_m=float(rng.uniform(*dr["tread_depth_m"])), steps=3)
                elif kind == "support":
                    ratio = ref["standable_ratio"] * factor
                    feature.update(
                        support_fraction_proxy=ratio, reference_window_m=window,
                        pad_width_m=float(np.sqrt(ratio) * window),
                        pit_depth_m=float(rng.uniform(0.45, 0.9)),
                    )
                else:
                    feature.update(
                        centre_to_wall_m=ref["safe_margin"] * factor,
                        passage_width_m=2 * ref["safe_margin"] * factor,
                        wall_height_m=float(rng.uniform(0.8, 1.5)),
                    )
                features.append(feature)
    return features
