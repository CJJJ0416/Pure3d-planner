"""Small connected corridor, native terrain-generator stairs and upper landing."""
import sys
from pathlib import Path

import numpy as np
import trimesh

from common.geneator.filter_components import SEMANTICS, surface_union
from common.geneator.filter_wfc import object_proxies
from common.geneator.wfc_connections import box, rod, straight_segment, flight_rails
from common.pointcloud_io import write_pcd
from utils.mesh_sampling import sample_surface
from utils.pointcloud import write_json


def build(folder, config):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'geneator/terrain-generator'))
    parts, roles = [], []

    def add(name, mesh):
        instance = len(roles) + 1
        semantic = SEMANTICS[name]
        parts.append(dict(mesh=mesh, instance_id=instance, semantic_id=semantic, target_role=1))
        roles.append(dict(instance_id=instance, semantic=name, semantic_id=semantic, static=True))

    add('floor', box([10, 8, .15], [0, 0, -.075]))
    # Open-ended lower corridor, a turn, and a stairwell with no ceiling over it.
    add('wall', box([5.8, .15, 2.6], [-1.8, -3.0, 1.3]))
    add('wall', box([4.6, .15, 2.6], [-2.4, -.95, 1.3]))
    add('wall', box([.15, 3.1, 2.6], [2.15, -1.55, 1.3]))
    add('ceiling', box([5.8, 2.2, .15], [-1.8, -2, 2.675]))
    # End-wall, columns and a doorway supply nonparallel constraints for LIO.
    add('wall', box([.15, 2.2, 2.6], [-4.7, -2, 1.3]))
    add('column_beam', box([.25, .25, 2.5], [-.6, -2.75, 1.25]))
    stairs, tread = straight_segment(1.8, 3.6, 1.2, 0, 1., 0., 1, 8, .15)
    add('stair', stairs)
    add('floor', box([3.2, 1.4, .16], [1., 4.15, 1.12]))
    for mesh in flight_rails(1.8, 3.6, 1.2, 0, 1., 0., 1, .9):
        add('railing', mesh)
    for x in [-.6, 2.6]:
        for y in [3.55, 4.75]:
            add('column_beam', rod([x, y, 0], [x, y, 1.04], .06))
        add('railing', rod([x, 3.55, 2.1], [x, 4.75, 2.1]))
    templates = dict(object_proxies([0, 0, 0]))
    for name, position in [('person_proxy', [-3.7, -2.55, 0]), ('movable_clutter', [-.1, -2.6, 0]),
                           ('chair', [2.15, 4.25, 1.2])]:
        mesh = templates[name].copy()
        mesh.apply_translation(position)
        add(name, mesh)
    mesh, attributes = surface_union(parts)
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(folder / 'static_mesh.ply')
    np.savez(folder / 'mesh_faces.npz', **attributes)
    clean, face, report = sample_surface(mesh, .03, return_face_ids=True)
    write_pcd(folder / 'clean.pcd', clean)
    np.savez(folder / 'clean_attributes.npz', face_id=face,
             **{key: value[face] for key, value in attributes.items()})
    for name in ['person_proxy', 'movable_clutter']:
        templates[name].export(folder / f'dynamic_{name}.ply')
    write_json(folder / 'scene.json', dict(seed=config['seed'], frame_id='map', units='m',
        geometry_group_id=f"fasterlio_{config['seed']}", split='example',
        bounds=mesh.bounds.tolist(), roles=roles, stair_tread_m=tread,
        upper_height_m=1.2, clean_points=len(clean), sampling=report,
        dynamic_instances=[dict(instance_id=1001, template='person_proxy', max_speed_m_s=.6),
                           dict(instance_id=1002, template='movable_clutter', max_speed_m_s=.25)]))
    return mesh
