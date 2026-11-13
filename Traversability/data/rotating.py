"""One replaceable four-building pool and persistent lightweight scene provenance."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from Traversability.data.sliced import prepare, progress
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def disk_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def generation_seeds(config, generation):
    count = config['rotation']['scenes_per_generation']
    first = config['rotation']['seed_start'] + generation * count
    return list(range(first, first + count))


def fixed_manifest(config):
    root = workspace_path(config['data'])
    root.mkdir(parents=True, exist_ok=True)
    fixed = workspace_path(config['fixed_data'])
    metadata = read_json(fixed / 'manifest.json')
    scenes = [s for s in metadata['scenes'] if s['split'] in ['val', 'test']]
    for scene in scenes:
        link = root / scene['id']
        if not link.is_symlink():
            link.symlink_to(fixed / scene['id'], target_is_directory=True)
    return dict(architecture=config['architecture'], config=config, scenes=scenes,
                blocks=[b for b in metadata['blocks'] if b['split'] in ['val', 'test']], complete=True)


def prepare_generation(config, generation):
    """Called in the native CuPy geometry environment, with WFC in the torch environment."""
    import trimesh
    from common.pointcloud_io import write_pcd
    from utils.mesh_sampling import sample_surface

    root, output = workspace_path(config['data']), workspace_path(config['output'])
    active = root / 'active'
    previous_path = root / 'generation.json'
    previous = read_json(previous_path) if previous_path.exists() else None
    seeds = generation_seeds(config, generation)
    started = time.perf_counter()
    if previous is not None and previous['index'] != generation:
        old_scenes = previous['seeds']
        old_bytes = disk_bytes(active)
        for seed in old_scenes:
            (root / f'scene_{seed}').unlink()
        shutil.rmtree(active)
        event = dict(from_generation=previous['index'], to_generation=generation,
                     removed_seeds=old_scenes, new_seeds=seeds, reclaimed_bytes=old_bytes,
                     previous_active_removed=not active.exists(),
                     previous_wfc_cache_removed=not (active / 'wfc_cache').exists())
        write_json(output / 'rotations' / f'{generation:02d}.json', event)
        previous_path.unlink()
    active.mkdir(parents=True, exist_ok=True)
    source = active / 'source'
    source.mkdir(exist_ok=True)
    progress(config, 'scene_generation', generation=generation, generation_seeds=seeds,
             scenes_completed=0, scenes_total=len(seeds))
    rows = []
    for number, seed in enumerate(seeds):
        name = f'scene_{seed}'
        folder = source / name
        archive = output / 'scene_archive' / name
        archive.mkdir(parents=True, exist_ok=True)
        tick = time.perf_counter()
        if not (folder / 'geometry_ready.json').exists():
            command = [config['generator_python'], '-u', str(ROOT / 'common/geneator/generate_wfc_multifloor.py'),
                       '--config', str(workspace_path(config['generator_config'])), '--seed', str(seed),
                       '--output', str(folder), '--cache-dir', str(active / 'wfc_cache')]
            env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(config['cpu_threads']),
                       OPENBLAS_NUM_THREADS=str(config['cpu_threads']))
            with (archive / 'generation.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(folder / 'geometry_ready.json', dict(seed=seed, generation_s=time.perf_counter()-tick))
        scene = read_json(folder / 'scene.json')
        if not (folder / 'sampling_ready.json').exists():
            mesh = trimesh.load_mesh(folder / 'terrain.obj', process=False)
            points, sampling = sample_surface(mesh, config['rotation']['surface_spacing_m'])
            write_pcd(folder / 'terrain.pcd', points)
            record = dict(seed=seed, points=len(points), spacing_m=config['rotation']['surface_spacing_m'],
                          sampling_s=time.perf_counter()-tick, **sampling)
            write_json(folder / 'sampling_ready.json', record)
            del points, mesh
        for filename in ['scene.json', 'config.resolved.json', 'overview.png', 'layout_overview.png',
                         'geometry_ready.json', 'sampling_ready.json', 'landings.report.json']:
            shutil.copy2(folder / filename, archive / filename)
        rows.append(dict(id=name, seed=seed, split='train', folder=name))
        progress(config, 'scene_generation', generation=generation, generation_seeds=seeds,
                 scenes_completed=number+1, scenes_total=len(seeds), active_scene=name,
                 temporary_bytes=disk_bytes(active))
        print(f'GEOMETRY READY {number+1}/{len(seeds)} seed={seed}', flush=True)
    write_json(source / 'manifest.json', dict(scenes=rows))
    local = deepcopy(config)
    local.update(source_data=str(source), data=str(active / 'prepared'))
    prepare(local)
    prepared = read_json(active / 'prepared/manifest.json')
    metadata = fixed_manifest(config)
    metadata['scenes'] = prepared['scenes'] + metadata['scenes']
    metadata['blocks'] = prepared['blocks'] + metadata['blocks']
    metadata['generation'] = generation
    for scene in prepared['scenes']:
        name = scene['id']
        link = root / name
        if not link.is_symlink():
            link.symlink_to(active / 'prepared' / name, target_is_directory=True)
        archive = output / 'scene_archive' / name
        write_json(archive / 'supervision.json', scene)
        write_json(archive / 'conditions.json', read_json(active / 'prepared' / name / 'ready.json')['scene']['profiles'])
        values = np.load(active / 'prepared' / name / 'cost.npy', mmap_mode='r')
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        coord = np.load(active / 'prepared' / name / 'coord.npy', mmap_mode='r')
        write_json(archive / 'data_summary.json', dict(points=len(coord), profiles=values.shape[1],
                   label_range=[float(values.min()), float(values.max())], all_labels_finite=True,
                   xyz_sha256=hashlib.sha256(coord.tobytes()).hexdigest(),
                   blocks=sum(b['scene'] == name for b in prepared['blocks']), shared_xyz_verified=scene['shared_xyz_verified']))
        del values, coord
    write_json(root / 'manifest.json', metadata)
    state = dict(index=generation, seeds=seeds, prepared=True, encoded=False,
                 preparation_s=time.perf_counter()-started, temporary_bytes=disk_bytes(active))
    write_json(previous_path, state)
    print(f'GENERATION {generation} PREPARED {state}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    args = parser.parse_args()
    prepare_generation(read_json(args.config), args.generation)


if __name__ == '__main__':
    main()
