#!/usr/bin/env python3
"""Create a verified, source-free archive of a playground scene and its native routes."""
import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
import tarfile
import tempfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=PROJECT / 'outputs/playground_rotated_seed20261104')
    parser.add_argument('--routes', type=Path, default=PROJECT / 'outputs/random_routes_seed20261104')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene, routes, output = args.scene.resolve(), args.routes.resolve(), args.output.resolve()
    if output.exists() or output.with_name(output.name + '.sha256').exists():
        raise FileExistsError('Choose a new archive and checksum filename')
    scene_json = json.loads((scene / 'scene.json').read_text())
    vst_meta = json.loads((scene / 'vst.meta.json').read_text())
    if digest(scene / 'terrain.pcd') != scene_json['pcd_sha256'] or vst_meta['pcd_sha256'] != scene_json['pcd_sha256']:
        raise ValueError('Scene and VST point-cloud fingerprints differ')
    if digest(scene / 'vst.pickle') != vst_meta['tomogram_sha256']:
        raise ValueError('VST fingerprint differs from metadata')
    if digest(scene / 'terrain.obj') != scene_json['mesh_sha256']:
        raise ValueError('Mesh fingerprint differs from scene metadata')
    report = json.loads((routes / 'routes.report.json').read_text())
    records = report['queries']
    viewer = json.loads((routes / 'viewer_manifest.json').read_text())
    if viewer['scene_seed'] != scene_json['seed']:
        raise ValueError('Route bundle and scene seeds differ')
    source_hashes = viewer['input_sha256']
    for name in ['terrain.obj', 'terrain.pcd', 'vst.pickle', 'vst.meta.json']:
        if source_hashes.get(name) != digest(scene / name):
            raise ValueError('Route bundle was generated from a different scene: ' + name)
    if len(records) != 18 or len({r['id'] for r in records}) != 18:
        raise ValueError('Expected exactly 18 distinct baseline routes')
    region_counts = {}
    for record in records:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', record['id']):
            raise ValueError('Invalid route ID: ' + record['id'])
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', record['region_id']):
            raise ValueError('Invalid region ID: ' + record['region_id'])
        if not (record['optimized'] and record['accepted'] and record['geometry_validation']['passed']):
            raise ValueError('Route has not passed planning and geometry checks: ' + record['id'])
        region = record['region_id']
        region_counts[region] = region_counts.get(region, 0) + 1
        if record['path_csv'] != f"paths/{record['id']}.gpmp.csv":
            raise ValueError('Unexpected route path: ' + record['path_csv'])
        csv_path = routes / record['path_csv']
        points = np.atleast_2d(np.loadtxt(csv_path, delimiter=',', skiprows=1))
        if points.shape[1] != 5 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Invalid route CSV: ' + str(csv_path))
        for extension in ['csv', 'npy']:
            relative = 'paths/' + record['id'] + '.gpmp.' + extension
            path = routes / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if source_hashes.get(relative) != digest(path):
                raise ValueError('Route file differs from validated bundle: ' + relative)
    if len(region_counts) != 9 or set(region_counts.values()) != {2}:
        raise ValueError('Expected two routes in each of nine regions')
    files = []
    scene_names = ['scene.json', 'terrain.obj', 'terrain.pcd', 'vst.pickle', 'vst.meta.json',
                   'config.resolved.json', 'geometry_paths.json', 'training_regions.json',
                   'geometry_validation.json', 'label_validation.json', 'supervision_summary.json']
    for name in scene_names:
        path = scene / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append((path, f'third_party/vst_global_planner/outputs/{scene.name}/{name}'))
    region_root = scene / 'regions'
    if {p.name for p in region_root.iterdir() if p.is_dir()} != set(region_counts):
        raise ValueError('Scene region folders and planned regions differ')
    for region in sorted(region_counts):
        for name in ['vst.pickle', 'tomography.npz', 'queries.json', 'planning_region.json']:
            path = region_root / region / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((path, f'third_party/vst_global_planner/outputs/{scene.name}/regions/{region}/{name}'))
    for name in ['routes.report.json', 'viewer_manifest.json']:
        files.append((routes / name, f'third_party/vst_global_planner/outputs/{routes.name}/{name}'))
    for record in sorted(records, key=lambda row: row['id']):
        for extension in ['csv', 'npy']:
            name = record['id'] + '.gpmp.' + extension
            files.append((routes / 'paths' / name, f'third_party/vst_global_planner/outputs/{routes.name}/paths/{name}'))
    hashes = {relative: digest(path) for path, relative in files}
    if len(hashes) != len(files):
        raise ValueError('Duplicate archive path')
    manifest = {'version': 'playground_scene_assets_v1', 'scene_seed': scene_json['seed'],
                'scene_directory': scene.name, 'route_directory': routes.name,
                'region_counts': region_counts, 'files': hashes,
                'scope': 'Geometry, VST and validated reference routes. No LiDAR captures, paired local training labels, or model weights.'}
    checksums = ''.join(f'{sha}  {name}\n' for name, sha in sorted(hashes.items())).encode()
    manifest_dir = f"third_party/vst_global_planner/outputs/playground_seed{scene_json['seed']}_manifest"
    extra = {
        f'{manifest_dir}/manifest.json':
            (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode(),
        f'{manifest_dir}/SHA256SUMS': checksums,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.pending', delete=False) as stream:
            temporary = Path(stream.name)
        with tarfile.open(temporary, 'w:gz', dereference=True) as archive:
            for path, relative in files:
                archive.add(path, arcname=relative, recursive=False)
            for name, data in extra.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
        with tarfile.open(temporary, 'r:gz') as archive:
            members = archive.getmembers()
            if len(members) != len(files) + len(extra) or any(not m.isfile() for m in members):
                raise ValueError('Unexpected archive entries or links')
            for relative, expected in hashes.items():
                stream = archive.extractfile(relative)
                sha = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
                if sha.hexdigest() != expected:
                    raise ValueError('Archive checksum mismatch: ' + relative)
        os.replace(temporary, output)
        temporary = None
        output.with_name(output.name + '.sha256').write_text(digest(output) + '  ' + output.name + '\n')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({'archive': str(output), 'files': len(files), 'regions': len(region_counts),
                      'routes': len(records), 'sha256': digest(output), 'sensor_training_data_included': False},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
