"""Encode complete 3-D contexts once and retain only supervised query features."""
import hashlib
import time

import numpy as np
import point_transformer_v3 as ptv3
import torch

from common.pointcloud_io import read_pcd_xyz
from Traversability.data.sliced import progress
from Traversability.data.voxel_hash import representatives, VoxelHash, verify_nearest
from Traversability.training.sliced import load_encoder
from utils.pointcloud import read_json, workspace_path, write_json


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def encode_context(encoder, points, source_local, config):
    zero = torch.zeros_like(points)
    spec = config['encoder']
    point, inverse = ptv3.prepare(points, zero, zero, preset=spec['name'],
        grid_size=spec['grid_size'], scale=spec['scale'], pick='first', device=config['device'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feature = ptv3.upcast(encoder(point)).feat
    query_inverse = inverse[source_local]
    used, back = torch.unique(query_inverse, sorted=True, return_inverse=True)
    return feature[used].half(), back.int(), query_inverse.int()


@torch.inference_mode()
def cache_scene(encoder, config, root, scene, rows, verify=False):
    folder = root / scene['id']
    complete = folder / 'full3d_encoding.json'
    if complete.exists():
        return read_json(complete)
    started = clock()
    tick = clock()
    raw = read_pcd_xyz(scene['source_pcd'])
    reading_s = clock()-tick
    raw_count = len(raw)
    tick = clock()
    points, original, keys, low, shape = representatives(
        torch.from_numpy(raw).to(config['device']), config['geometry_voxel_size_m'])
    voxel_s = clock()-tick
    del raw
    query = torch.from_numpy(np.array(np.load(folder/'coord.npy', mmap_mode='r'))).to(config['device'])
    tick = clock()
    table = VoxelHash(points, original, keys, low, shape, config['geometry_voxel_size_m'])
    source, distance, radius = table.query(query)
    matching_s = clock()-tick
    verification = verify_nearest(points, original, query, source) if verify else None
    matched_coord = points[source]
    distance_summary = dict(maximum_m=float(distance.max()), mean_m=float(distance.mean()),
        p95_m=float(torch.quantile(distance, .95)), maximum_cell_radius=int(radius.max()),
        mean_absolute_z_difference_m=float(abs(matched_coord[:, 2]-query[:, 2]).mean()),
        maximum_absolute_z_difference_m=float(abs(matched_coord[:, 2]-query[:, 2]).max()),
        z_difference_over_half_teacher_slice=int((abs(matched_coord[:,2]-query[:,2])>.25).sum()))
    del table, keys, radius, matched_coord
    tick = clock()
    np.save(folder/'geometry_coord.npy', points.cpu().numpy())
    np.save(folder/'geometry_source_index.npy', original.cpu().numpy())
    np.save(folder/'query_representative_index.npy', source.cpu().numpy())
    np.save(folder/'query_original_index.npy', original[source].cpu().numpy())
    np.save(folder/'query_match_distance.npy', distance.cpu().numpy())
    write_s = clock()-tick
    size = torch.tensor(config['window']['core_size_m'], device=config['device'])
    context = config['window']['context_m']
    grouping_s, encoding_s, encoder_calls = 0., 0., 0
    core_queries = 0
    max_context_points, peak_gib = 0, 0.
    for number, row in enumerate(rows):
        dest = folder / 'features' / row['id']
        if (dest/'ready.json').exists():
            ready = read_json(dest/'ready.json')
            core_queries += ready['points']
            max_context_points = max(max_context_points, ready['context_points'])
            continue
        tick = clock()
        lower = torch.tensor(row['lower'], device=config['device'])
        index = torch.nonzero(((points >= lower-context) & (points < lower+size+context)).all(1)).flatten()
        with np.load(folder/(row['id']+'.npz')) as block:
            point_index = block['point_index'].copy()
        query_index = torch.from_numpy(point_index).to(config['device'])
        representative_index = source[query_index]
        local = torch.searchsorted(index, representative_index)
        assert bool((local < len(index)).all()) and torch.equal(index[local], representative_index), \
            'A query source lies outside the fixed 1 m context.'
        grouping_s += clock()-tick
        torch.cuda.reset_peak_memory_stats()
        tick = clock()
        feature, back, tiny_index = encode_context(encoder, points[index], local, config)
        encoding_s += clock()-tick
        encoder_calls += 1
        peak_gib = max(peak_gib, torch.cuda.max_memory_allocated()/2**30)
        tick = clock()
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest/'features.npy', feature.cpu().numpy())
        np.save(dest/'inverse.npy', back.cpu().numpy())
        np.savez(dest/'correspondence.npz', point_index=point_index,
            representative_index=representative_index.cpu().numpy(),
            original_point_index=original[representative_index].cpu().numpy(),
            tiny_voxel_index=tiny_index.cpu().numpy(), feature_row=back.cpu().numpy())
        np.save(dest/'context_source_index.npy', index.int().cpu().numpy())
        write_json(dest/'ready.json', dict(points=len(point_index), voxels=len(feature),
            feature_channels=feature.shape[1], context_points=len(index), full3d_input=True,
            query_points_inserted_into_encoder=False))
        write_s += clock()-tick
        core_queries += len(point_index)
        max_context_points = max(max_context_points, len(index))
        if (number+1)%10 == 0 or number+1 == len(rows):
            progress(config, 'feature_cache', scene=scene['id'], split=scene['split'],
                blocks_completed=number+1, blocks_total=len(rows), geometry_points=len(points),
                raw_points=raw_count, matching=distance_summary, tiny_encoding_s=encoding_s)
            print(f'FULL3D ENCODE {scene["id"]} {number+1}/{len(rows)} | {clock()-started:.1f}s', flush=True)
    assert core_queries == len(query), 'Every Tomogram query must contribute once.'
    report = dict(complete=True, scene=scene['id'], input_pcd=scene['source_pcd'],
        raw_points=raw_count, geometry_points=len(points), supervised_queries=len(query),
        blocks=len(rows), encoder_calls=encoder_calls, maximum_context_points=max_context_points,
        query_xyz_sha256=hashlib.sha256(query.cpu().numpy().tobytes()).hexdigest(),
        matching=distance_summary, nearest_audit=verification, peak_cuda_gib=peak_gib,
        timing=dict(pcd_read_s=reading_s, gpu_voxel_s=voxel_s, gpu_correspondence_s=matching_s,
                    grouping_s=grouping_s, tiny_encoding_s=encoding_s, cache_write_s=write_s,
                    total_s=clock()-started),
        cpu_kdtree_used=False, full_geometry_input=True, labels_or_queries_in_encoder=False)
    write_json(complete, report)
    return report


def cache_features(config):
    root = workspace_path(config['data'])
    meta = read_json(root/'manifest.json')
    encoder = load_encoder(config)
    assert encoder.feature_channels == config['encoder']['feature_channels'], 'Encoder feature width mismatch'
    reference = {k:v.cpu().clone() for k,v in encoder.state_dict().items()}
    if config.get('source_run'):
        source = torch.load(workspace_path(config['source_run'])/'best.pt', map_location='cpu', weights_only=False)
        assert all(torch.equal(v, source['encoder'][k]) for k,v in reference.items())
        del source
    reports = []
    for scene in meta['scenes']:
        rows = [r for r in meta['blocks'] if r['scene']==scene['id']]
        reports.append(cache_scene(encoder, config, root, scene, rows, verify=scene['split'] in ['val','test']))
    equal = all(torch.equal(v.cpu(), reference[k]) for k,v in encoder.state_dict().items())
    assert equal, 'Frozen Tiny parameters or buffers changed during caching.'
    report = dict(complete=True, scenes=reports, frozen_weights_and_buffers_equal=equal,
        encoder=config['encoder'], encoder_passes_per_new_context=1)
    write_json(root/'encoding.json', report)
    del encoder
    torch.cuda.empty_cache()
    return report
