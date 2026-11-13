"""Straight, switchback, spiral and ramp connections, with optional railings."""
import numpy as np
import trimesh


COLORS = {
    "straight": [210, 124, 68, 255], "switchback": [181, 86, 80, 255],
    "spiral": [150, 103, 179, 255], "ramp": [68, 151, 160, 255],
}


def box(size, center):
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))


def rod(start, end, radius=0.025):
    start, end = np.asarray(start), np.asarray(end)
    transform = trimesh.geometry.align_vectors([0, 0, 1], end - start)
    transform[:3, 3] = (start + end) / 2
    return trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(end - start),
                                    sections=8, transform=transform)


def wedge(width, length, rise, base, center_y=0.0):
    """Solid ramp along +Y, with exact slope atan(rise/length)."""
    x, y = width / 2, length / 2
    vertices = [[-x, -y, base - .1], [x, -y, base - .1],
                [x, y, base - .1], [-x, y, base - .1],
                [-x, -y, base], [x, -y, base],
                [x, y, base + rise], [-x, y, base + rise]]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.apply_translation([0, center_y, 0])
    return mesh


def straight_segment(width, run, rise, base, x, start_y, direction, n_steps, thickness):
    """Use the upstream standard-stair geometry for each straight flight."""
    from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import StairMeshPartsCfg
    from terrain_generator.trimesh_tiles.mesh_parts.indoor_parts import create_standard_stairs
    landing = min(.3, run * .1)
    spec = StairMeshPartsCfg.Stair(
        dim=(width, run, rise), total_height=rise, n_steps=n_steps,
        step_width=width, step_depth=(run - landing) / (n_steps + 1),
        height_offset=thickness, floor_thickness=thickness,
        minimal_triangles=False, add_residual_side_up=True,
    )
    mesh = create_standard_stairs(spec)
    if direction < 0:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    mesh.apply_translation([x, start_y + direction * run / 2, base + rise / 2 - thickness])
    return mesh, spec.step_depth


def run_for_tread(n_steps, tread):
    """Invert straight_segment's landing convention so the requested tread is realised."""
    span = (n_steps + 1) * tread
    return span / .9 if span < 2.7 else span + .3


def flight_rails(width, run, rise, base, x, start_y, direction, height):
    result = []
    for side in [-1, 1]:
        rail_x = x + side * width / 2
        bottom = [rail_x, start_y, base + height]
        top = [rail_x, start_y + direction * run, base + rise + height]
        result.append(rod(bottom, top))
        for t in np.linspace(0, 1, int(np.ceil(run / .6)) + 1):
            z = base + rise * t
            y = start_y + direction * run * t
            result.append(rod([rail_x, y, z], [rail_x, y, z + height], radius=.018))
    return result


def annular_tread(inner, outer, angle0, angle1, z, thickness):
    angles = np.linspace(angle0, angle1, 4)
    polygon = np.vstack((
        np.c_[outer * np.cos(angles), outer * np.sin(angles)],
        np.c_[inner * np.cos(angles[::-1]), inner * np.sin(angles[::-1])],
    ))
    n = len(polygon)
    vertices = np.vstack((np.c_[polygon, np.full(n, z - thickness)],
                          np.c_[polygon, np.full(n, z)]))
    faces = []
    m = len(angles)
    for i in range(m - 1):
        a, b, c, d = i, i + 1, n - 2 - i, n - 1 - i
        faces.extend([[a, c, b], [a, d, c],
                      [n + a, n + b, n + c], [n + a, n + c, n + d]])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n + j], [i, n + j, n + i]])
    return trimesh.Trimesh(vertices, faces, process=False)


def build_connection(cfg, spec, *, return_parts=False):
    r0, r1 = spec["bay_rows"]
    c0, c1 = spec["bay_columns"]
    rows, columns = cfg["shape"]
    dx, dy = cfg["tile_dim_m"][:2]
    center = [(0.5 * (c0 + c1 - 1) - (columns - 1) / 2) * dx,
              ((rows - 1) / 2 - 0.5 * (r0 + r1 - 1)) * dy]
    footprint = [(c1 - c0) * dx, (r1 - r0) * dy]
    rotation = spec.get("rotation_deg", 0)
    width, length = footprint if rotation % 180 == 0 else footprint[::-1]
    base = cfg["floor_z_m"][spec["from_floor"]]
    upper = cfg["floor_z_m"][spec["to_floor"]]
    rise, thickness = upper - base, cfg["floor_thickness_m"]
    landing = .45
    inner_length = length - 2 * landing
    stair_width = spec["width_m"]
    body = [box([width, length, thickness], [0, 0, base - thickness / 2])]
    # The upper perimeter ring joins the stair exit to neighbouring WFC floor cells.
    for y in [-1, 1]:
        body.append(box([width, landing, thickness],
                        [0, y * (length - landing) / 2, upper - thickness / 2]))
    for x in [-1, 1]:
        body.append(box([landing, length - 2 * landing, thickness],
                        [x * (width - landing) / 2, 0, upper - thickness / 2]))
    rails = []
    record = dict(spec, rise_m=rise, footprint_m=footprint,
                  unrotated_footprint_m=[width, length], center_local_xy=center,
                  traversal_directions=["up", "down"])
    record["sampled_tread_depth_m"] = record.pop("tread_depth_m")
    record["requested_threshold_band"] = spec["threshold_band"]
    kind = spec["kind"]
    if kind == "straight" or (kind == "ramp" and spec["ramp_layout"] == "straight"):
        start = -inner_length / 2
        if kind == "straight":
            n = int(np.ceil(rise / spec["target_riser_m"]))
            run = run_for_tread(n, spec["tread_depth_m"]) if spec.get("exact_tread_depth") else inner_length
            mesh, depth = straight_segment(stair_width, run, rise, base, 0, start, 1, n, thickness)
            if inner_length > run + 1e-6:
                body.append(box([stair_width, inner_length - run, thickness],
                                [0, start + (run + inner_length) / 2, upper - thickness / 2]))
            record.update(steps=n, riser_m=rise / n, tread_depth_m=depth)
        else:
            mesh = wedge(stair_width, inner_length, rise, base)
            record.update(slope_rad=float(np.arctan2(rise, inner_length)))
        body.append(mesh)
        if spec["railings"]:
            rails += flight_rails(stair_width, inner_length, rise, base, 0, start, 1, spec["railing_height_m"])
    elif kind == "switchback" or kind == "ramp":
        turn_depth = stair_width
        turn_y = length / 2 - landing - turn_depth
        available_run = inner_length - turn_depth
        half_rise = rise / 2
        n = int(np.ceil(half_rise / spec["target_riser_m"]))
        if kind == "ramp":
            run = available_run
        else:
            run = (run_for_tread(n, spec["tread_depth_m"]) if spec.get("exact_tread_depth")
                   else min(available_run, (n + 1) * spec["tread_depth_m"] + .3))
        gap = .25
        flight_x = (stair_width + gap) / 2
        body.append(box([2 * stair_width + gap, turn_depth, thickness],
                        [0, turn_y + turn_depth / 2, base + half_rise - thickness / 2]))
        for x, start_y, direction, floor_base in [
            (-flight_x, turn_y - run, 1, base),
            (flight_x, turn_y, -1, base + half_rise),
        ]:
            if kind == "ramp":
                mesh = wedge(stair_width, run, half_rise, floor_base)
                if direction < 0:
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
                mesh.apply_translation([x, start_y + direction * run / 2, 0])
            else:
                mesh, depth = straight_segment(stair_width, run, half_rise, floor_base,
                                               x, start_y, direction, n, thickness)
            body.append(mesh)
            if spec["railings"]:
                rails += flight_rails(stair_width, run, half_rise, floor_base, x, start_y,
                                      direction, spec["railing_height_m"])
        exit_y = turn_y - run
        bridge_length = exit_y + length / 2
        body.append(box([stair_width, bridge_length, thickness],
                        [flight_x, -length / 2 + bridge_length / 2, upper - thickness / 2]))
        if kind == "ramp":
            record.update(slope_rad=float(np.arctan2(half_rise, run)))
        else:
            record.update(steps=2 * n, riser_m=half_rise / n, tread_depth_m=depth)
    else:
        outer = width / 2 - landing - .1
        inner = outer - stair_width
        if "inner_radius_m" in spec:
            inner = spec["inner_radius_m"]
            outer = inner + stair_width
        n = int(np.ceil(rise / spec["target_riser_m"]))
        start_angle, total_angle = -np.pi / 2, spec["turns"] * 2 * np.pi
        angles = np.linspace(start_angle, start_angle + total_angle, n + 1)
        for i in range(n):
            body.append(annular_tread(inner, outer, angles[i] - .01, angles[i + 1] + .01,
                                       base + (i + 1) * rise / n, .08))
        post = trimesh.creation.cylinder(radius=inner + .02, height=rise, sections=24)
        post.apply_translation([0, 0, base + rise / 2])
        body.append(post)
        exit_angle = (angles[-1] + angles[-2]) / 2
        ux, uy = np.cos(exit_angle), np.sin(exit_angle)
        distance = min((width / 2 - landing / 2) / abs(ux),
                       (length / 2 - landing / 2) / abs(uy))
        begin = inner + stair_width * .3
        connector = box([distance - begin, stair_width, thickness],
                        [(distance + begin) / 2, 0, upper - thickness / 2])
        connector.apply_transform(trimesh.transformations.rotation_matrix(exit_angle, [0, 0, 1]))
        body.append(connector)
        if spec["railings"]:
            for radius in [inner, outer]:
                positions = []
                for i, angle in enumerate(angles[1:]):
                    z = base + (i + 1) * rise / n
                    point = [radius * np.cos(angle), radius * np.sin(angle), z]
                    positions.append([point[0], point[1], z + spec["railing_height_m"]])
                    if i % 2 == 0:
                        rails.append(rod(point, positions[-1], radius=.018))
                rails += [rod(a, b) for a, b in zip(positions[:-1], positions[1:])]
        record.update(steps=n, riser_m=rise / n, inner_radius_m=inner, outer_radius_m=outer,
                      total_turn_degrees=360 * spec["turns"],
                      tread_depth_at_mid_radius_m=(inner + outer) / 2 * total_angle / n)
    for mesh in body:
        mesh.visual.face_colors = COLORS[kind]
    for mesh in rails:
        mesh.visual.face_colors = [65, 76, 89, 255]
    combined = trimesh.util.concatenate([*body, *rails])
    combined.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(rotation), [0, 0, 1]))
    combined.apply_translation([*center, 0])
    record["railing_parts"] = len(rails)
    reference = cfg["reference_vst"]["slope_max" if kind == "ramp" else "step_max"]
    actual = record["slope_rad"] if kind == "ramp" else record["riser_m"]
    factor = actual / reference
    record["actual_threshold_factor"] = factor
    record["threshold_band"] = "below" if factor < .9 else "above" if factor > 1.1 else "near"
    if return_parts:
        transform = trimesh.transformations.rotation_matrix(np.deg2rad(rotation), [0, 0, 1])
        transform[:2, 3] = center
        components = []
        for index, part in enumerate([*body, *rails]):
            component = part.copy()
            component.apply_transform(transform)
            semantic = ("railing" if index >= len(body) else
                        "floor" if index < 5 else "ramp" if kind == "ramp" else "stair")
            components.append(dict(mesh=component, semantic=semantic))
        return combined, record, components
    return combined, record
