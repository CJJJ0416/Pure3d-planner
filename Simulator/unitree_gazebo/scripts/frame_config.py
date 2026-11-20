"""Rigid transform helpers shared by the LIO frame interface and configuration."""
from pathlib import Path
import math

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT.parents[1] / 'Entrance/config/sim/frames.yaml'
DEFAULT_LIO_CONFIG = PACKAGE_ROOT.parents[1] / 'Entrance/config/sim/faster_lio.yaml'


def quaternion_product(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    q = np.array([w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
                  w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z])
    return q / np.linalg.norm(q)


def rotate(q, v):
    return v + 2.*np.cross(q[:3], np.cross(q[:3], v) + q[3]*v)


def rotation_matrix(q):
    return np.column_stack([rotate(q, axis) for axis in np.eye(3)])


def transform_from_config(value):
    t = np.asarray(value['translation'], dtype=float)
    rpy = np.asarray(value['rpy'], dtype=float)
    if t.shape != (3,) or rpy.shape != (3,) or not np.isfinite([t, rpy]).all():
        raise ValueError('Expected three finite translation values and three finite rpy angles')
    r, p, y = rpy / 2
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    q = np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                  cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])
    return t, q


def inverse(transform):
    t, q = transform
    qi = q * np.array([-1., -1., -1., 1.])
    return -rotate(qi, t), qi


def compose(a, b):
    ta, qa = a
    tb, qb = b
    return ta + rotate(qa, tb), quaternion_product(qa, qb)


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['mount']['source'] not in ('urdf', 'parameters') or cfg['anchor']['source'] not in ('gazebo', 'fixed', 'initial_body'):
        raise ValueError('Invalid mount.source or anchor.source')
    names = [cfg['frames'][key] for key in ('map', 'camera', 'body')]
    if 'world' in cfg['frames']:
        names.append(cfg['frames']['world'])
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError('world, map, camera and body frame names must be distinct')
    for value in (cfg['mount']['body_to_lidar'], cfg['mount']['lidar_to_imu'], cfg['anchor']):
        transform_from_config(value)
    return cfg


def anchor_edges(frames, transform):
    """Keep legacy map->camera configs; split simulation alignment above map."""
    if 'world' in frames:
        identity = (np.zeros(3), np.array([0., 0., 0., 1.]))
        return [(frames['world'], frames['map'], transform),
                (frames['map'], frames['camera'], identity)]
    return [(frames['map'], frames['camera'], transform)]


def prepare_lio_config(frame_path, output, lio_path=DEFAULT_LIO_CONFIG, profile=None):
    cfg = load_config(frame_path)
    params = yaml.safe_load(Path(lio_path).read_text())
    p = params['/**']['ros__parameters']
    if profile == 'real' and (cfg['anchor']['source'] == 'gazebo' or p.get('use_sim_time', False)):
        raise ValueError('real requires an anchor without Gazebo truth and use_sim_time=false')
    if profile == 'sim' and not p.get('use_sim_time', False):
        raise ValueError('sim requires use_sim_time=true')
    # Faster-LIO expects IMU <- LiDAR, the inverse of the configured LiDAR <- IMU.
    t, q = inverse(transform_from_config(cfg['mount']['lidar_to_imu']))
    p['mapping']['extrinsic_T'] = t.tolist()
    p['mapping']['extrinsic_R'] = rotation_matrix(q).ravel().tolist()
    p['publish']['tf_world_frame'] = cfg['frames']['camera']
    Path(output).write_text(yaml.safe_dump(params))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lio-config', type=Path, default=DEFAULT_LIO_CONFIG)
    parser.add_argument('--profile', choices=['sim', 'real'])
    args = parser.parse_args()
    prepare_lio_config(args.config, args.output, args.lio_config, args.profile)
