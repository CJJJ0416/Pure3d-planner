"""Offline fused-XYZ corruptions. Provenance is supervision, never model input."""
import numpy as np
from scipy.spatial.transform import Rotation

from common.geneator.filter_components import SEMANTICS

SOURCE_ROLES = {
    "observed_structure": 0, "jittered_structure": 1,
    "registration_fragment": 2,
    "static_object": 5, "registration_trail": 6, "dynamic_object_trail": 7,
}
SOURCE_COLORS = np.array([[55, 155, 125], [60, 140, 200], [235, 151, 35],
    [150, 100, 205], [207, 97, 45], [216, 65, 126], [182, 116, 34], [220, 45, 45]]) / 255.


def augment_cloud(clean, clean_attributes, config, seed, variant, scene):
    """Return shuffled XYZ and aligned attributes, plus replayable transforms.

    source_point_id indexes clean.pcd even for a moving object. instance_id is
    the observed actor, source_instance_id is its stationary template. Each
    transform maps that template's map-frame clean_xyz to the final observation.
    """
    streams = np.random.SeedSequence([seed, 2901]).spawn(3)
    registration_rng, dynamic_rng, shuffle_rng = [np.random.default_rng(s) for s in streams]
    base_rng = np.random.default_rng(np.random.SeedSequence([scene["seed"], config.get("suite_seed", 0), 1701]))
    records, source_arrays, transforms, chunks = {}, {}, [np.eye(4).tolist()], []
    low, high = clean.min(0), clean.max(0)

    def append(points, reference, role, *, fragment_id=-1, trail_id=-1, step=-1,
               transform=None, actor=None, disconnected=None, placement_id=-1):
        count = len(points)
        transform_id = 0
        if transform is not None:
            transforms.append(transform.tolist())
            transform_id = len(transforms) - 1
        src = np.asarray(reference, dtype=np.int64)
        dynamic = np.isscalar(role) and role == SOURCE_ROLES["dynamic_object_trail"]
        chunks.append(dict(
            coord=np.asarray(points, dtype=np.float32), source_point_id=src,
            source_face_id=clean_attributes["face_id"][src].astype(np.int64),
            source_instance_id=clean_attributes["instance_id"][src].astype(np.int32),
            instance_id=clean_attributes["instance_id"][src].astype(np.int32) if actor is None else np.full(count, actor, np.int32),
            semantic_id=clean_attributes["semantic_id"][src].astype(np.int32),
            target_role=np.full(count, 0 if dynamic else 1, np.uint8),
            source_role=np.broadcast_to(np.asarray(role, np.uint8), (count,)).copy(),
            fragment_id=np.full(count, fragment_id, np.int32), trail_id=np.full(count, trail_id, np.int32),
            trail_step=np.full(count, step, np.int32), transform_id=np.full(count, transform_id, np.int32),
            ghost_placement_id=np.full(count, placement_id, np.int32),
            is_disconnected_control=np.zeros(count, bool) if disconnected is None else disconnected,
            coordinate_residual=(np.asarray(points) - (clean[src] if transform is None else
                clean[src] @ transform[:3, :3].T + transform[:3, 3])).astype(np.float32),
            clean_xyz=clean[src].copy()))

    density = np.asarray(config["retention_probability_range"])
    probability = density[0] + np.diff(density)[0] * (clean[:, 0] - low[0]) / (high[0] - low[0])
    keep = base_rng.random(len(clean)) < probability
    stair_indices = np.flatnonzero(clean_attributes["semantic_id"] == SEMANTICS["stair"])
    holes = []
    for _ in range(config["hole_count"]):
        anchor = clean[base_rng.choice(stair_indices)]
        size = base_rng.uniform(*config["hole_size_m_range"], size=3)
        inside = (np.abs(clean - anchor) < size / 2).all(1)
        holes.append(dict(center=anchor.tolist(), size_m=size.tolist(), removed_points=int((keep & inside).sum())))
        keep &= ~inside
    anchor = clean[base_rng.choice(stair_indices)]
    axis = int(base_rng.integers(0, 2))
    width = float(base_rng.uniform(*config["stripe_width_m_range"]))
    stripe = (abs(clean[:, axis] - anchor[axis]) < width / 2) & (abs(clean[:, 2] - anchor[2]) < .6)
    keep &= ~stripe
    disconnected = np.zeros(len(clean), bool)
    controls = []
    for semantic in config["disconnected_controls"]["semantic_ids"]:
        candidates = np.flatnonzero(clean_attributes["semantic_id"] == semantic)
        center = clean[base_rng.choice(candidates)]
        radius = float(base_rng.uniform(*config["disconnected_controls"]["radius_m_range"]))
        distance = np.linalg.norm(clean - center, axis=1)
        inside = distance < radius
        shell = (distance >= radius) & (distance < radius + config["disconnected_controls"]["gap_m"])
        keep[shell] = False
        keep[inside] = True
        disconnected[inside] = True
        controls.append(dict(semantic_id=semantic, center=center.tolist(), radius_m=radius,
                             gap_m=config["disconnected_controls"]["gap_m"], points=int(inside.sum())))
    source = np.flatnonzero(keep)
    sigma = float(base_rng.uniform(*config["jitter_std_m_range"]))
    jitter = (base_rng.random(len(source)) < config["jitter_fraction"]) & ~disconnected[source]
    # Exact stationary exemplars are a positive control for every movable class.
    jitter &= clean_attributes["semantic_id"][source] < 8
    observed = clean[source].copy()
    observed[jitter] += base_rng.normal(0, sigma, (int(jitter.sum()), 3))
    role = np.where(clean_attributes["semantic_id"][source] >= 8, SOURCE_ROLES["static_object"],
                    np.where(jitter, SOURCE_ROLES["jittered_structure"], SOURCE_ROLES["observed_structure"]))
    append(observed, source, role, disconnected=disconnected[source])
    records.update(density=dict(retention_probability_range=density.tolist()), holes=holes,
        stripe=dict(center=anchor.tolist(), axis=axis, width_m=width), disconnected_controls=controls,
        jitter=dict(std_m=sigma, points=int(jitter.sum())), variant=variant)

    fragments, registration_trails = [], []
    if variant["registration"]:
        rng = registration_rng
        for fragment_id, semantic in enumerate(config["fragment_semantic_ids"]):
            candidates = np.flatnonzero(clean_attributes["semantic_id"] == semantic)
            anchor = clean[rng.choice(candidates)]
            radius = float(rng.uniform(*config["fragment_radius_m_range"]))
            src = candidates[np.linalg.norm(clean[candidates] - anchor, axis=1) < radius]
            src = rng.choice(src, min(len(src), config["fragment_max_points"]), replace=False)
            near = fragment_id == 0
            magnitude = rng.uniform(*(config["near_shift_m_range"] if near else config["ghost_shift_m_range"]))
            direction = rng.normal(size=3)
            direction[2] = abs(direction[2]) + .4
            shift = direction / np.linalg.norm(direction) * magnitude
            angle = rng.uniform(-config["fragment_rotation_deg"], config["fragment_rotation_deg"], 3)
            if near:
                angle *= .02
            rotation = Rotation.from_euler("xyz", angle, degrees=True).as_matrix()
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, anchor + shift - rotation @ anchor
            points = clean[src] @ rotation.T + transform[:3, 3]
            append(points, src, SOURCE_ROLES["registration_fragment"], fragment_id=fragment_id, transform=transform)
            key = f"fragment_{fragment_id:03d}_source_point_id"
            source_arrays[key] = src
            fragments.append(dict(id=fragment_id, semantic_id=semantic, anchor_xyz=anchor.tolist(),
                radius_m=radius, source_indices_key=key, transform_id=len(transforms) - 1, near_surface=near, points=len(src)))

        settings = config["registration_trails"]
        for trail_id, semantic in enumerate(settings["semantic_ids"]):
            candidates = np.flatnonzero(clean_attributes["semantic_id"] == semantic)
            anchor_index = int(rng.choice(candidates))
            anchor = clean[anchor_index]
            radius = float(rng.uniform(*settings["patch_radius_m_range"]))
            src = candidates[np.linalg.norm(clean[candidates] - anchor, axis=1) < radius]
            src = rng.choice(src, min(len(src), settings["max_patch_points"]), replace=False)
            steps = int(rng.integers(*settings["copies_range"]))
            direction = clean_attributes["surface_normal"][anchor_index] + rng.normal(0, .15, 3)
            shift = direction / np.linalg.norm(direction) * rng.uniform(*settings["total_drift_m_range"])
            angles = rng.uniform(-1, 1, 3) * rng.uniform(*settings["total_rotation_deg_range"])
            frames = []
            for step in range(1, steps + 1):
                fraction = step / steps
                rotation = Rotation.from_euler("xyz", angles * fraction, degrees=True).as_matrix()
                transform = np.eye(4)
                transform[:3, :3], transform[:3, 3] = rotation, anchor + fraction * shift - rotation @ anchor
                points = clean[src] @ rotation.T + transform[:3, 3]
                append(points, src, SOURCE_ROLES["registration_trail"], fragment_id=len(fragments) + trail_id,
                       trail_id=trail_id, step=step, transform=transform)
                frames.append(dict(step=step, transform_id=len(transforms) - 1))
            key = f"registration_{trail_id:03d}_source_point_id"
            source_arrays[key] = src
            registration_trails.append(dict(id=trail_id, semantic_id=semantic, anchor_xyz=anchor.tolist(),
                radius_m=radius, copies=steps, source_indices_key=key, frames=frames, points=len(src) * steps))

    floating_ghosts = []
    # Explicit destinations extend the same source-fragment augmentation beyond
    # the core and context. No box boundary is used to assign retention targets.
    for placement_id, setting in enumerate(config.get("floating_ghosts", [])):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3901, placement_id]))
        semantic = SEMANTICS[setting["semantic"]]
        candidates = np.flatnonzero(clean_attributes["semantic_id"] == semantic)
        if setting['kind']=='volume_scatter':
            # Independent floating returns occupy several air volumes, rather
            # than retaining the shape of a copied wall or stair patch.
            points=np.concatenate([rng.uniform(region['lower'],region['upper'],(region['points'],3))
                                   for region in setting['air_regions']])
            src=rng.choice(candidates,len(points),replace=False)
            frames=[]
            for point,source in zip(points,src):
                transform=np.eye(4);transform[:3,3]=point-clean[source]
                append(point[None],np.array([source]),SOURCE_ROLES['registration_fragment'],
                    fragment_id=len(fragments)+len(registration_trails)+placement_id,
                    transform=transform,placement_id=placement_id)
                frames.append(dict(transform_id=len(transforms)-1,points=1,destination_xyz=point.tolist()))
            key=f'floating_{placement_id:03d}_source_point_id'
            source_arrays[key]=src
            floating_ghosts.append(dict(id=placement_id,**setting,source_indices_key=key,
                points=len(points),frames=frames))
            continue
        anchor = clean[rng.choice(candidates)]
        src = candidates[np.linalg.norm(clean[candidates] - anchor, axis=1) < setting["radius_m"]]
        if setting["kind"] == "vertical_scatter":
            src = rng.choice(src, max(1, int(len(src) * setting["surface_fraction"])), replace=False)
        # Recenter the selected actual surface samples, including asymmetric
        # fragments at source edges. Their original face identities stay intact.
        pivot = clean[src].mean(0)
        steps = setting.get("copies", 1)
        frames = []
        for step in range(steps):
            # Sparse ghosts are sampled at their source before any copy is
            # emitted. A new subset per position avoids dense repeated sheets.
            chosen=(rng.choice(src,min(len(src),setting['points_per_position']),replace=False)
                    if 'points_per_position' in setting else src)
            fraction = step / max(steps - 1, 1)
            angles = np.asarray(setting.get("rotation_deg", [0., 0., 0.])) * fraction
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            destination = np.asarray(setting["destination_xyz"]) + fraction * np.asarray(setting.get("drift_xyz", [0., 0., 0.]))
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, destination - rotation @ pivot
            points = clean[chosen] @ rotation.T + transform[:3, 3]
            points += rng.normal(0., setting.get("jitter_std_m", .003), points.shape)
            role = SOURCE_ROLES["registration_trail" if steps > 1 else "registration_fragment"]
            append(points, chosen, role, fragment_id=len(fragments) + len(registration_trails) + placement_id,
                   trail_id=len(registration_trails) + placement_id if steps > 1 else -1,
                   step=step, transform=transform, placement_id=placement_id)
            frames.append(dict(step=step, transform_id=len(transforms) - 1, destination_xyz=destination.tolist(),
                               points=len(chosen)))
        key = f"floating_{placement_id:03d}_source_point_id"
        source_arrays[key] = src
        floating_ghosts.append(dict(id=placement_id, **setting, source_indices_key=key,
            source_anchor_xyz=anchor.tolist(), source_pivot_xyz=pivot.tolist(),
            points=sum(frame['points'] for frame in frames), frames=frames))

    dynamic_trails = []
    if variant["dynamic"]:
        rng = dynamic_rng
        settings = config["dynamic_trails"]
        world = np.asarray(scene["transform"])
        next_actor = int(clean_attributes["instance_id"].max()) + 1
        for corridor_index, corridor in enumerate(scene["corridors"]):
            frame = world @ np.asarray(corridor["local_to_scene"])
            length, width = corridor["length_m"], corridor["width_m"]
            for ordinal, semantic in enumerate(settings["semantic_ids"]):
                # Templates come from the same scene's exact stationary controls.
                candidates = np.flatnonzero(clean_attributes["semantic_id"] == semantic)
                template = corridor["object_instances"][str(semantic)]
                src = candidates[clean_attributes["instance_id"][candidates] == template]
                # Object-local coordinates use the corridor frame, including yaw.
                local_template = (clean[src] - frame[:3, 3]) @ frame[:3, :3]
                origin = np.r_[local_template[:, :2].mean(0), local_template[:, 2].min()]
                local_template -= origin
                radius = np.linalg.norm(local_template[:, :2], axis=1).max()
                available = max(0., width / 2 - radius - .05)
                steps = int(rng.integers(settings["positions_range"][0], settings["positions_range"][1] + 1))
                phase = np.linspace(0., 1., steps)
                curved = (ordinal + corridor_index) % 2 == 1
                amplitude = float(rng.uniform(.3, .9) * available) if curved else 0.
                a, b = corridor["motion_interval_m"]
                available_length = b - a - 2 * radius - .1
                first, last = np.sort(rng.uniform([0., .8], [.2, 1.]))
                y = a + radius + .05 + np.linspace(first, last, steps) * available_length
                if rng.random() < .5:
                    y = y[::-1]
                x = amplitude * np.sin(np.pi * phase)
                yaw = np.arctan2(-np.gradient(x), np.gradient(y))
                frames, trajectory_id = [], len(dynamic_trails)
                for step in range(steps):
                    rotation = Rotation.from_euler("z", yaw[step]).as_matrix()
                    local_position = np.array([x[step], y[step], 0.])
                    local_to_world = np.eye(4)
                    local_to_world[:3, :3] = rotation
                    local_to_world[:3, 3] = local_position
                    pose = frame @ local_to_world
                    template_to_local = np.eye(4)
                    template_to_local[:3, :3] = frame[:3, :3].T
                    template_to_local[:3, 3] = -frame[:3, :3].T @ frame[:3, 3] - origin
                    transform = pose @ template_to_local
                    fraction = float(rng.uniform(*settings["surface_fraction_range"]))
                    # A local cap of sampled surface, with its orientation varying
                    # per position. This is array sampling, not sensor visibility.
                    direction = rng.normal(size=3)
                    projection = local_template @ direction
                    mask = projection >= np.quantile(projection, 1 - fraction)
                    chosen = src[mask]
                    points = clean[chosen] @ transform[:3, :3].T + transform[:3, 3]
                    append(points, chosen, SOURCE_ROLES["dynamic_object_trail"], trail_id=trajectory_id,
                           step=step, transform=transform, actor=next_actor)
                    frames.append(dict(step=step, transform_id=len(transforms) - 1,
                        position_xyz=pose[:3, 3].tolist(), surface_fraction=fraction, points=len(chosen)))
                dynamic_trails.append(dict(id=trajectory_id, instance_id=next_actor, template_instance_id=template,
                    semantic_id=semantic, corridor_id=corridor["id"], kind="curved" if curved else "straight",
                    copies=steps, anchor_xyz=frames[len(frames) // 2]["position_xyz"], frames=frames,
                    points=sum(f["points"] for f in frames), corridor_width_m=width,
                    template_radius_m=float(radius), lateral_amplitude_m=amplitude))
                next_actor += 1

    result = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    order = shuffle_rng.permutation(len(result["coord"]))
    result = {name: values[order] for name, values in result.items()}
    result["target_relevance"] = result["target_role"].astype(np.float32)
    source_arrays["transforms"] = np.asarray(transforms, dtype=np.float64)
    records.update(seed=seed, fragments=fragments, registration_trails=registration_trails,
        floating_ghosts=floating_ghosts,
        dynamic_trails=dynamic_trails, shuffled=True, source_roles=SOURCE_ROLES, points=len(order),
        source_role_counts={name: int((result["source_role"] == code).sum()) for name, code in SOURCE_ROLES.items()},
        transform_table="augmentation_sources NPZ: transforms[transform_id]", labels_status="D2 pending")
    return result, records, source_arrays
