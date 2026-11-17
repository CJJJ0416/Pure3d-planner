#!/usr/bin/env python3
"""Export a terrain-generator OBJ as an exact static triangle mesh for Classic.

Only the Python standard library is needed. The terrain is not simplified or
transformed. A separate ground plane below its lowest point catches falls.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = WORKSPACE / 'datasets/simulation/isaac_lio_20261117/scene'


def read_obj(path):
    vertices, faces = [], []
    with path.open() as stream:
        for line in stream:
            words = line.split()
            if not words:
                continue
            if words[0] == 'v':
                vertex = tuple(map(float, words[1:4]))
                if len(vertex) != 3 or not all(map(math.isfinite, vertex)):
                    raise ValueError('Nonfinite or malformed OBJ vertex')
                vertices.append(vertex)
            elif words[0] == 'f':
                indices = [int(word.split('/')[0]) for word in words[1:]]
                indices = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                for index in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[index], indices[index + 1]))
    if not vertices or not faces:
        raise ValueError(f'Empty terrain mesh: {path}')
    return vertices, faces


def export_stl(path, vertices, faces):
    with path.open('wb') as stream:
        stream.write(b'Pure3D terrain-generator; world coordinates, metres'.ljust(80, b'\0'))
        stream.write(struct.pack('<I', len(faces)))
        for face in faces:
            a, b, c = [vertices[index] for index in face]
            u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
            normal = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
            length = math.sqrt(sum(value*value for value in normal))
            normal = tuple(value / length for value in normal) if length else (0., 0., 0.)
            stream.write(struct.pack('<12fH', *normal, *a, *b, *c, 0))


def vertical_hits(vertices, faces, x, y):
    """All intersections of a vertical line with nonvertical mesh triangles."""
    hits = []
    for face in faces:
        a, b, c = [vertices[index] for index in face]
        if not min(a[0], b[0], c[0]) - 1e-8 <= x <= max(a[0], b[0], c[0]) + 1e-8:
            continue
        if not min(a[1], b[1], c[1]) - 1e-8 <= y <= max(a[1], b[1], c[1]) + 1e-8:
            continue
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-12:
            continue
        u = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / den
        v = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / den
        if u >= -1e-7 and v >= -1e-7 and u + v <= 1. + 1e-7:
            hits.append(u*a[2] + v*b[2] + (1-u-v)*c[2])
    return hits


def choose_spawn(scene, vertices, faces, floor, clearance):
    cfg = scene['config']
    z = cfg['floor_z_m'][floor]
    angle = math.radians(cfg.get('yaw_deg', 0.))
    ca, sa = math.cos(angle), math.sin(angle)
    rows, columns = cfg['shape']
    dx, dy = cfg['tile_dim_m'][:2]
    candidates = []
    for stair in scene['stairs']:
        for side, key in [('lower', 'from_floor'), ('upper', 'to_floor')]:
            if stair[key] != floor or side + '_landing_cell' not in stair:
                continue
            row, column = stair[side + '_landing_cell']
            local_x = (column - (columns-1)/2) * dx
            local_y = ((rows-1)/2 - row) * dy
            x, y = ca*local_x - sa*local_y, sa*local_x + ca*local_y
            center = stair['center_world_xyz']
            yaw = math.atan2(center[1] - y, center[0] - x)
            cy, sy = math.cos(yaw), math.sin(yaw)
            probes, supported = [], True
            # 0.8 x 0.6 m footprint, including the feet and body clearance.
            for px in (-.4, 0., .4):
                for py in (-.3, 0., .3):
                    hits = vertical_hits(vertices, faces, x+cy*px-sy*py, y+sy*px+cy*py)
                    ground = [h for h in hits if abs(h-z) < .025]
                    obstacles = [h for h in hits if z+.025 < h < z+.8]
                    probes.append({'offset_xy': [px, py], 'support': bool(ground), 'clear': not obstacles})
                    supported &= bool(ground) and not obstacles
            candidates.append({'connection': stair['id'], 'side': side, 'supported': supported})
            if supported:
                return dict(x=x, y=y, height=z+clearance, yaw=yaw, floor=floor,
                            ground_z=z, landing=stair['id'] + '/' + side,
                            probes=probes, candidates=candidates)
    raise ValueError(f'No clear, supported landing on floor {floor}: {candidates}')


def make_world(path, mesh, spawn, ground_z):
    sdf = ET.Element('sdf', version='1.7')
    world = ET.SubElement(sdf, 'world', name='pure3d_terrain')
    state = ET.SubElement(world, 'plugin', name='gazebo_ros_state', filename='libgazebo_ros_state.so')
    ET.SubElement(state, 'update_rate').text = '100'
    ET.SubElement(world, 'gravity').text = '0 0 -9.81'
    physics = ET.SubElement(world, 'physics', name='ode', type='ode')
    for key, value in [('max_step_size', '.001'), ('real_time_update_rate', '1000'), ('real_time_factor', '1')]:
        ET.SubElement(physics, key).text = value
    solver = ET.SubElement(ET.SubElement(physics, 'ode'), 'solver')
    ET.SubElement(solver, 'type').text = 'quick'
    ET.SubElement(solver, 'iters').text = '50'
    scene = ET.SubElement(world, 'scene')
    for key, value in [('ambient', '.6 .6 .6 1'), ('background', '.8 .85 .9 1'), ('shadows', 'true')]:
        ET.SubElement(scene, key).text = value
    sun = ET.SubElement(world, 'light', name='sun', type='directional')
    for key, value in [('pose', '0 0 20 0 0 0'), ('diffuse', '.8 .8 .8 1'), ('specular', '.1 .1 .1 1'), ('direction', '-.5 .2 -1')]:
        ET.SubElement(sun, key).text = value
    camera = ET.SubElement(ET.SubElement(world, 'gui', fullscreen='false'), 'camera', name='user_camera')
    ET.SubElement(camera, 'pose').text = f"{spawn['x']+5} {spawn['y']-8} {spawn['height']+6} 0 .5 2.1"
    ground = ET.SubElement(world, 'model', name='ground_plane')
    ET.SubElement(ground, 'static').text = 'true'
    ET.SubElement(ground, 'pose').text = f'0 0 {ground_z} 0 0 0'
    ground_link = ET.SubElement(ground, 'link', name='ground')
    for kind in ('collision', 'visual'):
        element = ET.SubElement(ground_link, kind, name='ground_' + kind)
        plane = ET.SubElement(ET.SubElement(element, 'geometry'), 'plane')
        ET.SubElement(plane, 'normal').text = '0 0 1'
        # ODE plane collision is infinite; the visible surface spans 200 m.
        ET.SubElement(plane, 'size').text = '200 200'
        if kind == 'collision':
            friction = ET.SubElement(ET.SubElement(ET.SubElement(element, 'surface'), 'friction'), 'ode')
            ET.SubElement(friction, 'mu').text = '1.0'
            ET.SubElement(friction, 'mu2').text = '1.0'
        else:
            material = ET.SubElement(element, 'material')
            ET.SubElement(material, 'ambient').text = '.35 .4 .3 1'
            ET.SubElement(material, 'diffuse').text = '.45 .5 .4 1'
    model = ET.SubElement(world, 'model', name='terrain_generator')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = '0 0 0 0 0 0'
    link = ET.SubElement(model, 'link', name='terrain')
    for kind in ('collision', 'visual'):
        element = ET.SubElement(link, kind, name='terrain_' + kind)
        geom = ET.SubElement(ET.SubElement(element, 'geometry'), 'mesh')
        ET.SubElement(geom, 'uri').text = mesh.as_uri()
        ET.SubElement(geom, 'scale').text = '1 1 1'
        if kind == 'collision':
            friction = ET.SubElement(ET.SubElement(ET.SubElement(element, 'surface'), 'friction'), 'ode')
            ET.SubElement(friction, 'mu').text = '1.0'
            ET.SubElement(friction, 'mu2').text = '1.0'
        else:
            material = ET.SubElement(element, 'material')
            ET.SubElement(material, 'ambient').text = '.55 .6 .65 1'
            ET.SubElement(material, 'diffuse').text = '.65 .7 .75 1'
    ET.indent(sdf)
    ET.ElementTree(sdf).write(path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'datasets/simulation/gazebo_terrain')
    parser.add_argument('--floor', type=int, default=0)
    parser.add_argument('--clearance', type=float, default=.4, help='Initial base height above the landing, m')
    args = parser.parse_args()
    scene_dir, output = args.scene.resolve(), args.output.resolve()
    scene = json.loads((scene_dir / 'scene.json').read_text())
    if scene.get('units') != 'm' or scene.get('frame_id') != 'map' or scene.get('z_axis') != 'up':
        raise ValueError('Expected metre, map, Z-up terrain-generator coordinates')
    if not 0 <= args.floor < len(scene['config']['floor_z_m']) or not .25 <= args.clearance <= .7:
        raise ValueError('Invalid floor or clearance (expected 0.25–0.7 m)')
    source = scene_dir / 'terrain.obj'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report_path = output / 'scene.report.json'
    signature = dict(source_obj=str(source), source_sha256=digest, floor=args.floor,
                     clearance=args.clearance, exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('signature') == signature and all((output/name).exists() for name in ('terrain.stl', 'terrain.world', 'spawn.json')):
            print(f'Using prepared world: {output / "terrain.world"}')
            return
    vertices, faces = read_obj(source)
    spawn = choose_spawn(scene, vertices, faces, args.floor, args.clearance)
    # Place the catcher at least 5 cm below the complete mesh, rounded down to
    # 5 cm. This also preserves pits and other source geometry below floor zero.
    ground_z = math.floor((min(vertex[2] for vertex in vertices) - .05) / .05) * .05
    output.mkdir(parents=True, exist_ok=True)
    mesh = output / 'terrain.stl'
    export_stl(mesh, vertices, faces)
    make_world(output / 'terrain.world', mesh, spawn, ground_z)
    (output / 'spawn.json').write_text(json.dumps(spawn, indent=2) + '\n')
    report = dict(signature=signature, seed=scene['seed'], vertices=len(vertices), triangles=len(faces),
                  world=str(output / 'terrain.world'), mesh=str(mesh), spawn=spawn,
                  geometry='All source triangles; STL float32 coordinates; no scaling or simplification',
                  ground_plane=True, ground_z_m=ground_z,
                  ground_visual_size_m=[200., 200.], ground_collision='infinite ODE plane',
                  odometry='Gazebo truth world->base; LIO world->map->camera_init->body')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {len(faces):,} terrain triangles: {output / "terrain.world"}')
    print(f'Spawn: {spawn["landing"]}, xyz=({spawn["x"]:.3f}, {spawn["y"]:.3f}, {spawn["height"]:.3f})')
    print(f'Ground plane: z={ground_z:.3f} m (below the terrain mesh)')


if __name__ == '__main__':
    main()
