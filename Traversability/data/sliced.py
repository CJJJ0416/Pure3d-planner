"""Shared native geometry and exactly aligned slice-point teacher labels."""
import argparse
import hashlib
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from common.pointcloud_io import read_pcd_xyz, write_pcd
from utils.pointcloud import ROOT, read_json, spatial_blocks, workspace_path, write_json


def geometry_cloud(ground, center, resolution, slice_dh):
    """Use the native float16 export precision and geometry-only ground merge."""
    ground = ground.astype(np.float16).astype(np.float32)
    for layer in range(len(ground) - 1):
        merge = ground[layer + 1] - ground[layer] < slice_dh
        ground[layer, merge] = np.nan
    layer, x, y = np.nonzero(np.isfinite(ground))
    coord = np.column_stack(((x - ground.shape[1] // 2) * resolution + center[0],
                             (y - ground.shape[2] // 2) * resolution + center[1],
                             ground[layer, x, y])).astype(np.float32)
    return coord, np.column_stack((layer, x, y)).astype(np.int32)


def progress(config, stage, **values):
    values = {**config.get('progress_context', {}), **values}
    path = workspace_path(config['output']) / 'progress.json'
    target = config.get('stop_epoch', config['epochs'])
    payload = dict(status='complete' if stage == 'complete' else 'running', stage=stage,
                   updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), epochs_total=target, **values)
    if target != config['epochs']:
        payload['scheduler_budget_epochs'] = config['epochs']
    completed = values.get('completed_epochs', 0)
    filled = int(20 * completed / target)
    payload['training_bar'] = '[' + '█' * filled + '░' * (20-filled) + f"] {completed}/{target} epochs"
    if 'last_validation' in values:
        payload['validation_cost_mae_raw'] = values['last_validation']['cost_mae_raw']
        payload['parameter_response'] = values['last_validation']['parameter_response']
    temporary = path.with_suffix('.json.tmp')
    write_json(temporary, payload)
    temporary.replace(path)


def make_teacher(config, pcd):
    sys.path.insert(0, str(ROOT / 'common/vst_global_planner/src'))
    from vst_global_planner.conditioned_teacher import ConditionedTeacher
    return ConditionedTeacher(read_pcd_xyz(pcd), teacher_config=workspace_path(config['teacher_config']))


def save_blocks(folder, coord, scene_id, split, config):
    rows = []
    for number, (indices, core, lower) in enumerate(spatial_blocks(
            coord, config['window']['core_size_m'], config['window']['context_m'])):
        name = f'block_{number:05d}'
        np.savez(folder / (name + '.npz'), context_index=indices,
                 core_local=np.flatnonzero(core), point_index=indices[core])
        rows.append(dict(id=name, scene=scene_id, split=split,
                         points=int(core.sum()), context_points=len(indices),
                         lower=lower.tolist()))
    return rows


def prepare_inference(config, pcd, output):
    output.mkdir(parents=True, exist_ok=True)
    teacher = make_teacher(config, pcd)
    coord, layer_cell = geometry_cloud(teacher.ground, teacher.center, teacher.resolution, teacher.slice_dh)
    np.save(output / 'coord.npy', coord)
    np.save(output / 'layer_cell.npy', layer_cell)
    blocks = save_blocks(output, coord, 'inference', 'inference', config)
    write_json(output / 'geometry.json', dict(source_pcd=str(pcd), points=len(coord),
               blocks=blocks, teacher_geometry=teacher.metadata(), cost_computed=False))


def prepare(config):
    # This environment supplies CuPy, SciPy and the native teacher, without torch.
    from Traversability.experiments.decoupling_data import condition_banks
    from Traversability.experiments.stair_data import legacy_stair_mask
    from vst_global_planner.tomogram_cloud import tomogram_points

    root = workspace_path(config['data'])
    root.mkdir(parents=True, exist_ok=True)
    source = workspace_path(config['source_data'])
    original = read_json(source / 'manifest.json')
    metadata = dict(architecture=config['architecture'], config=config, scenes=[], blocks=[])
    started = time.perf_counter()
    for number, old in enumerate(original['scenes']):
        folder = root / old['id']
        folder.mkdir(exist_ok=True)
        if (folder / 'ready.json').exists():
            ready = read_json(folder / 'ready.json')
            metadata['scenes'].append(ready['scene'])
            metadata['blocks'].extend(ready['blocks'])
            continue
        source_folder = source / old['folder']
        pcd = source_folder / 'terrain.pcd'
        scene = read_json(source_folder / 'scene.json')
        tick = time.perf_counter()
        teacher = make_teacher(config, pcd)
        coord, layer_cell = geometry_cloud(teacher.ground, teacher.center, teacher.resolution, teacher.slice_dh)
        profiles, banks, pairs = condition_banks(
            teacher, old['seed'], np.asarray(config['condition_ranges']),
            np.ones_like(teacher.ground, dtype=bool), old['split'] == 'train')
        cost = np.lib.format.open_memmap(folder / 'cost.npy', mode='w+', dtype=np.float16,
                                        shape=(len(coord), len(profiles)))
        same_xyz = True
        for index, profile in enumerate(profiles):
            progress(config, 'teacher', scenes_completed=number, scenes_total=len(original['scenes']),
                     scene=old['id'], profile=index + 1, profiles_total=len(profiles))
            values = teacher.cost(profile['condition'])
            payload = teacher.payload(profile['condition'], values)
            cloud = tomogram_points(payload)
            same_xyz &= np.array_equal(cloud[:, :3], coord)
            cost[:, index] = cloud[:, 3]
            if index in [0, 3]:
                with (folder / f"vst_{profile['id']}.pickle").open('wb') as stream:
                    pickle.dump(payload, stream, pickle.HIGHEST_PROTOCOL)
            if index == 0:
                write_pcd(folder / 'ground_truth.pcd', coord, cloud[:, 3])
            if (index + 1) % 10 == 0:
                print(f"TEACHER {old['id']} {index+1}/{len(profiles)}", flush=True)
        cost.flush()
        assert same_xyz, 'All condition labels must use the shared native slice XYZ.'
        np.save(folder / 'coord.npy', coord)
        np.save(folder / 'layer_cell.npy', layer_cell)
        np.save(folder / 'conditions.npy', np.asarray([p['condition'] for p in profiles], np.float32)[:, [0, 1, 2, 4]])
        facility = legacy_stair_mask(coord, scene)
        np.save(folder / 'facility.npy', facility)
        rows = save_blocks(folder, coord, old['id'], old['split'], config)
        record = dict(id=old['id'], split=old['split'], seed=old['seed'],
                      source_pcd=str(pcd.resolve()), source_sha256=hashlib.sha256(pcd.read_bytes()).hexdigest(),
                      points=len(coord), profiles=profiles, banks=banks, pairs=pairs,
                      shared_xyz_verified=same_xyz, facility_points=int(facility.sum()),
                      teacher=teacher.metadata(), preparation_s=time.perf_counter() - tick)
        write_json(folder / 'ready.json', dict(scene=record, blocks=rows))
        metadata['scenes'].append(record)
        metadata['blocks'].extend(rows)
        write_json(root / 'manifest.json', metadata)
        print(f"SCENE READY {number+1}/{len(original['scenes'])} {old['id']} {len(coord):,} points {len(rows)} blocks", flush=True)
        del teacher, cost
    metadata.update(complete=True, preparation_s=time.perf_counter() - started)
    write_json(root / 'manifest.json', metadata)
    progress(config, 'data_ready', scenes_completed=len(metadata['scenes']), scenes_total=len(metadata['scenes']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pcd', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.pcd:
        prepare_inference(config, workspace_path(args.pcd), workspace_path(args.output))
    else:
        prepare(config)


if __name__ == '__main__':
    main()
