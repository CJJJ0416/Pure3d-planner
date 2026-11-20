"""Measured geometry cases around clearance, slope, step and support thresholds."""
import numpy as np
import trimesh

from wfc_connections import box, wedge, straight_segment


def build_features(cfg, *, return_parts=False):
    per_floor = [[] for _ in cfg["floor_z_m"]]
    records = []
    components = []
    rows, columns = cfg["shape"]
    dx, dy = cfg["tile_dim_m"][:2]
    thickness = cfg["floor_thickness_m"]
    for feature in cfg["capability_features"]:
        kind = feature["kind"]
        parts = []
        extent = .75 * dx
        if kind == "headroom":
            height, width = feature["clearance_m"], feature["width_m"]
            parts.append(box([width + .24, extent, .14], [0, 0, height + .07]))
            for side in [-1, 1]:
                parts.append(box([.12, extent, height + .14],
                                 [side * (width / 2 + .06), 0, (height + .14) / 2]))
        elif kind == "slope":
            rise = np.tan(feature["slope_rad"]) * extent
            parts.append(wedge(feature["width_m"], extent, rise, 0))
        elif kind == "step":
            rise = feature["riser_m"] * feature["steps"]
            run = feature["tread_depth_m"] * (feature["steps"] + 1) + .15
            mesh, depth = straight_segment(.65 * dx, run, rise, 0, 0, -run / 2, 1,
                                           feature["steps"], thickness)
            parts.append(mesh)
        elif kind == "clearance":
            height, width = feature["wall_height_m"], feature["passage_width_m"]
            for side in [-1, 1]:
                parts.append(box([.12, extent, height],
                                 [side * (width / 2 + .06), 0, height / 2]))
        else:
            depth, pad = feature["pit_depth_m"], feature["pad_width_m"]
            rim = .18 * dx
            parts.append(box([dx, dy, thickness], [0, 0, -depth - thickness / 2]))
            for side in [-1, 1]:
                parts.append(box([rim, dy, thickness],
                                 [side * (dx - rim) / 2, 0, -thickness / 2]))
                parts.append(box([dx - 2 * rim, rim, thickness],
                                 [0, side * (dy - rim) / 2, -thickness / 2]))
            parts.append(box([pad, pad, depth], [0, 0, -depth / 2]))
        mesh = trimesh.util.concatenate(parts)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.deg2rad(feature["rotation_deg"]), [0, 0, 1]
        ))
        row, column = feature["cell"]
        x, y = (column - (columns - 1) / 2) * dx, ((rows - 1) / 2 - row) * dy
        z = cfg["floor_z_m"][feature["floor"]]
        mesh.apply_translation([x, y, z])
        if return_parts:
            rotation = trimesh.transformations.rotation_matrix(np.deg2rad(feature["rotation_deg"]), [0, 0, 1])
            rotation[:3, 3] = [x, y, z]
            for index, part in enumerate(parts):
                component = part.copy()
                component.apply_transform(rotation)
                semantic = ({"slope": "ramp", "step": "stair", "clearance": "wall", "support": "floor"}.get(kind)
                            if kind != "headroom" else "ceiling" if index == 0 else "wall")
                components.append(dict(mesh=component, semantic=semantic, feature_id=feature["id"]))
        per_floor[feature["floor"]].append(mesh)
        record = dict(feature, center_local_xyz=[x, y, z])
        if kind == "slope":
            record.update(run_m=extent, rise_m=float(rise))
        elif kind == "step":
            record.update(actual_tread_depth_m=depth, total_rise_m=rise)
        records.append(record)
    if return_parts:
        return per_floor, records, components
    return per_floor, records
