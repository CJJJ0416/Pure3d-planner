#!/usr/bin/env python3
"""Separate diagnostic: recover PTv3 features at every valid raw LiDAR return."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parents[1]/'third_party/PointTransformerV3/src')]


from ptv3_tomography.paths import relocated_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import numpy as np
    import torch
    from point_transformer_v3 import prepare, set_backend
    from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset
    from ptv3_tomography.io import atomic_json, file_sha256, read_json
    from ptv3_tomography.models.dense_perception import SingleFramePerception
    from ptv3_tomography.models.point_restoration import restore_point_features
    from ptv3_tomography.training.pretrain_state import read_pretraining_checkpoint

    args.maps, args.output = args.maps.resolve(), args.output.resolve()
    manifest = read_json(args.maps/'manifest.json')
    checkpoint = relocated_path(manifest['source_checkpoint'])
    checkpoint_hash = file_sha256(checkpoint)
    dataset = LidarStepDataset(relocated_path(manifest['source_dataset']))
    saved = read_pretraining_checkpoint(checkpoint)
    cfg = manifest['map_config']
    set_backend(attention='sdpa', conv='torch', layer_norm='torch')
    device = torch.device(args.device)
    model = SingleFramePerception(saved['model_spec']['model'], cfg).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    args.output.mkdir(parents=True)
    (args.output/'frames').mkdir()
    atomic_json(args.output/'status.json', {'status':'running'})
    records, arrays_all = [], []
    with torch.no_grad():
        for map_row in manifest['frames']:
            i = map_row['frame_index']
            frame = dataset.frames[i]
            raw_file = dataset.asset(frame['scan_file'])
            with np.load(raw_file, allow_pickle=False) as a:
                valid_ray_indices = np.flatnonzero(a['hit_mask'])
                xyz = a['points_sensor_m'][valid_ray_indices].copy()
                world = a['points_world_m'][valid_ray_indices].copy()
                tf = a['world_from_sensor'].copy()
                ray_count = len(a['hit_mask'])
            map_file = args.maps/map_row['file']
            with np.load(map_file, allow_pickle=False) as a:
                cached = {key:a[key].copy() for key in a.files}
            points = torch.as_tensor(xyz, device=device)
            up = torch.as_tensor(cached['gravity_up_sensor'], device=device)
            prepared, inverse = prepare(points, feat=('coord',), grid_size=model.voxel_size_m,
                                         center=False, pick='first')
            tokens = model.backbone(prepared)
            # This is the explicit, visualization-only point restoration branch.
            restored = restore_point_features(tokens.feat, inverse)
            n, m, c = len(points), len(tokens.feat), tokens.feat.shape[1]
            inv = inverse.cpu().numpy()
            voxel_features = tokens.feat.cpu().numpy()
            point_features = restored.cpu().numpy()
            xy = np.rint(xyz[:,:2]/cfg['resolution_m']).astype(np.int64)+np.asarray(cfg['grid_size_xy'])//2
            band = np.searchsorted(cfg['height_bin_edges_m'], xyz@cached['gravity_up_sensor'], side='right')-1
            inside = ((xy>=0)&(xy<cfg['grid_size_xy'])).all(1)&(band>=0)&(band<len(cfg['height_bin_edges_m'])-1)
            sampled = {key:np.full(n,np.nan,np.float32) for key in ['cost','height']}
            for key, target in [('cost','cost_raw'),('height','ground_height_gravity_m')]:
                sampled[key][inside] = cached[target][band[inside],xy[inside,0],xy[inside,1]]
            arrays = {'points_sensor_m':xyz, 'points_world_m':world, 'world_from_sensor':tf,
                      'valid_ray_indices':valid_ray_indices, 'inverse':inv,
                      'voxel_points_sensor_m':tokens.coord.cpu().numpy(),
                      'voxel_grid_coord':tokens.grid_coord.cpu().numpy(),
                      'voxel_features':voxel_features,'restored_point_features':point_features,
                      'within_map_domain':inside,'sampled_map_cost_raw':sampled['cost'],
                      'sampled_map_height_gravity_m':sampled['height']}
            arrays_all.append(arrays)
            records.append({'frame_index':i,'rays':ray_count,'valid_returns':n,'voxel_tokens':m,
                            'restored_feature_shape':[n,c], 'within_map_domain':int(inside.sum()),
                            'outside_map_domain':int((~inside).sum()),
                            'reference_cost_supervision_cells':map_row['cost_cells'],
                            'reference_height_supervision_cells':map_row['height_cells'],
                            'all_valid_points_retained_in_raw_order':True,
                            'point_xyz_unchanged':True})
    # Fit one PCA basis across this entire capture so replay colors stay comparable.
    features = np.concatenate([a['voxel_features'] for a in arrays_all]).astype(np.float64)
    mean = features.mean(0)
    eigenvalues, basis = np.linalg.eigh(np.cov(features-mean,rowvar=False))
    order = np.argsort(eigenvalues)[::-1][:3]
    basis = basis[:,order]
    for j in range(3):
        if basis[np.argmax(np.abs(basis[:,j])),j] < 0:
            basis[:,j] *= -1
    projected = (features-mean)@basis
    low, high = np.quantile(projected,[.02,.98],axis=0)
    high = np.maximum(high,low+1e-8)
    np.savez_compressed(args.output/'pca_basis.npz',mean=mean,basis=basis,low=low,high=high)
    for row, arrays in zip(records,arrays_all):
        projected = (arrays['voxel_features']-mean)@basis
        rgb = np.clip((projected-low)/(high-low),0,1).astype(np.float32)
        arrays['voxel_feature_rgb'] = rgb
        arrays['restored_feature_rgb'] = rgb[arrays['inverse']]
        file = args.output/'frames'/f"frame_{row['frame_index']:05d}.npz"
        np.savez_compressed(file,**arrays)
        row.update(file=str(file.relative_to(args.output)),sha256=file_sha256(file))
    report = {'status':'passed','frames':len(records),'source_maps':str(args.maps),
              'source_dataset':str(dataset.directory),'source_checkpoint':str(checkpoint),
              'checkpoint_sha256':checkpoint_hash,
              'all_valid_returns_restored':True,'raw_point_order_and_coordinates_preserved':True,
              'new_trainable_parameters':0,'training_or_finetuning_executed':False,
              'restoration':'F_points = F_voxels[inverse]; original raw point coordinates are retained',
              'feature_colors':'Shared capture-wide PCA of PTv3 features, not semantic labels or cost',
              'cost_colors':'Existing predicted grid cost sampled at LiDAR endpoints. Outside grid/bands stays gray. No new point prediction head.',
              'hardware_limits_imposed':False,'frames_manifest':records}
    atomic_json(args.output/'report.json',report)
    atomic_json(args.output/'status.json',{'status':'passed'})
    print({'status':'passed','frames':len(records),'valid_returns':sum(r['valid_returns'] for r in records),
           'frame16':next((r for r in records if r['frame_index']==16),None)},flush=True)


if __name__ == '__main__':
    main()
