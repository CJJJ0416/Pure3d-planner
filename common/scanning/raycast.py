"""First-hit Warp queries with time-dependent sensor and actor transforms."""
import gzip
import hashlib
from pathlib import Path
import numpy as np
import warp as wp

from .trajectory import actor_position


@wp.kernel
def cast_kernel(static_mesh: wp.uint64, actor0: wp.uint64, actor1: wp.uint64,
                actors_enabled: int, origins: wp.array(dtype=wp.vec3),
                directions: wp.array(dtype=wp.vec3), actor_positions0: wp.array(dtype=wp.vec3),
                actor_positions1: wp.array(dtype=wp.vec3), max_range: float,
                ranges: wp.array(dtype=float), faces: wp.array(dtype=int), instances: wp.array(dtype=int)):
    i = wp.tid()
    hit = wp.mesh_query_ray(static_mesh, origins[i], directions[i], max_range)
    best = max_range
    face = int(-1)
    instance = int(0)
    if hit.result:
        best = float(hit.t)
        face = int(hit.face)
    if actors_enabled:
        a = wp.mesh_query_ray(actor0, origins[i] - actor_positions0[i], directions[i], max_range)
        if a.result and a.t < best:
            best = float(a.t)
            face = int(a.face)
            instance = 1001
        b = wp.mesh_query_ray(actor1, origins[i] - actor_positions1[i], directions[i], max_range)
        if b.result and b.t < best:
            best = float(b.t)
            face = int(b.face)
            instance = 1002
    ranges[i] = best
    faces[i] = face
    instances[i] = instance


class RollingRayCaster:
    def __init__(self, mesh, actors, config):
        self.config = config
        self.device = config['device']
        wp.config.kernel_cache_dir = str(__import__('pathlib').Path(config['output']).resolve() / '.warp_cache')
        wp.init()
        self.meshes = []
        for item in [mesh, *actors]:
            self.meshes.append(wp.Mesh(
                wp.array(np.asarray(item.vertices, np.float32), dtype=wp.vec3, device=self.device),
                wp.array(np.asarray(item.faces, np.int32).ravel(), dtype=int, device=self.device)))
        pattern = Path(config['pattern'])
        raw = gzip.decompress(pattern.read_bytes()) if pattern.suffix == '.gz' else pattern.read_bytes()
        table = np.loadtxt(__import__('io').BytesIO(raw), delimiter=',', skiprows=1, usecols=(1, 2))
        az = np.deg2rad(table[:, 0])
        el = np.deg2rad(90 - table[:, 1])
        self.directions = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)].astype(np.float32)
        self.pattern_sha256 = hashlib.sha256(raw).hexdigest()

    def scan(self, frame, trajectory, dynamic, edge, rng):
        c = self.config
        n = c['rays_per_scan']
        ray_id = np.arange(n, dtype=np.int32)
        offset_ns = np.rint((ray_id.astype(np.float64) + .5) / (n * c['scan_hz']) * 1e9).astype(np.uint32)
        time = frame / c['scan_hz'] + offset_ns * 1e-9
        origin, rotation = trajectory.evaluate(time)
        # Subsample the same angular-pattern interval without slowing its phase.
        # At 24,000/24,000 this reproduces the original sequential indices.
        span = c.get('pattern_samples_per_scan', n)
        pattern_offset = np.floor((ray_id.astype(np.float64)+.5)*span/n).astype(np.int64)
        directions = self.directions[(frame * span + pattern_offset) % len(self.directions)]
        beams = [directions]
        if edge:
            # Four finite-footprint samples in a tangent basis around the centre ray.
            axis = np.cross(directions, np.broadcast_to([0, 0, 1], directions.shape))
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            other = np.cross(directions, axis)
            angle = np.deg2rad(c['edge_angle_deg'])
            for vector in [axis, -axis, other, -other]:
                beams.append(directions * np.cos(angle) + vector * np.sin(angle))
        beam_count = len(beams)
        sensor_direction = np.concatenate(beams)
        world_direction = np.concatenate([np.einsum('nij,nj->ni', rotation, b) for b in beams])
        positions = [np.tile(actor_position(time, actor), (beam_count, 1)) for actor in range(2)]
        count = beam_count * n
        def gpu(value):
            return wp.array(np.asarray(value, np.float32), dtype=wp.vec3, device=self.device)
        ranges = wp.empty(count, dtype=float, device=self.device)
        faces = wp.empty(count, dtype=int, device=self.device)
        instances = wp.empty(count, dtype=int, device=self.device)
        wp.launch(cast_kernel, count, [*(m.id for m in self.meshes), int(dynamic),
            gpu(np.tile(origin, (beam_count, 1))), gpu(world_direction), gpu(positions[0]), gpu(positions[1]),
            float(c['max_range_m']), ranges, faces, instances], device=self.device)
        measured = ranges.numpy().reshape(beam_count, n)
        face = faces.numpy()[:n]
        instance = instances.numpy()[:n]
        elevations = np.rad2deg(np.arcsin(np.clip(sensor_direction[:, 2], -1, 1))).reshape(beam_count, n)
        fov_low, fov_high = c['vertical_fov_deg']
        in_fov = (elevations >= fov_low-1e-5) & (elevations <= fov_high+1e-5)
        near_range = max(c['min_range_m'], c['blind_zone_m'])
        valid_beam = (measured >= near_range) & (measured < c['max_range_m']) & in_fov
        valid = valid_beam[0] & (face >= 0)
        mixed = np.zeros(n, bool)
        distance = measured[0].copy()
        if edge:
            near = np.where(valid_beam, measured, np.inf).min(0)
            far = np.where(valid_beam, measured, -np.inf).max(0)
            candidate = valid & (valid_beam.sum(0) >= 2) & ((far - near) > c['edge_range_span_m'])
            mixed = candidate & (rng.random(n) < c['edge_probability'])
            weight = rng.uniform(.15, .85, mixed.sum())
            distance[mixed] = weight * near[mixed] + (1 - weight) * far[mixed]
            distance[valid] += rng.normal(0, c['range_noise_std_m'], valid.sum())
            valid &= (distance >= near_range) & (distance < c['max_range_m'])
        xyz = directions[valid] * distance[valid, None]
        truth_xyz = np.einsum('nij,nj->ni', rotation[valid], xyz) + origin[valid]
        return dict(coord=xyz.astype(np.float32), truth_coord=truth_xyz.astype(np.float32),
            ray_id=ray_id[valid], offset_ns=offset_ns[valid], range_m=distance,
            return_status=np.where(~in_fov[0], 3,
                np.where(valid, 1, np.where((face >= 0)&(distance < near_range), 2, 0))).astype(np.uint8),
            face_id=face[valid], dynamic_instance=instance[valid], edge_mixed=mixed[valid],
            reflectivity=(ray_id[valid] % 251 + 1).astype(np.uint8))
