"""Building PCD -> native geometric slices -> frozen Tiny -> conditional cost."""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np
import point_transformer_v3 as ptv3
import torch

from common.pointcloud_io import write_pcd
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.sliced import encode_window
from utils.pointcloud import ROOT, read_json, write_json, workspace_path


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pcd', type=Path, required=True)
    parser.add_argument('--condition', type=float, nargs=4, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['architecture'] == 'full3d_tiny_cost_v1':
        from Traversability.inference.full3d import predict
        predict(args.checkpoint, args.pcd, args.condition, args.output, state)
        return
    config = state['config']
    torch.set_num_threads(config['cpu_threads'])
    output = workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    tick = time.perf_counter()
    subprocess.run([config['geometry_python'], '-m', 'Traversability.data.sliced',
                    '--config', str(output / 'config.json'), '--pcd', str(workspace_path(args.pcd)),
                    '--output', str(output / 'geometry')], cwd=ROOT, check=True)
    geometry_s = time.perf_counter() - tick
    metadata = read_json(output / 'geometry/geometry.json')
    coord = np.load(output / 'geometry/coord.npy', mmap_mode='r')
    encoder = ptv3.PointTransformerV3.from_preset(config['encoder']['name']).to(config['device']).eval()
    encoder.precision = 'fp32'
    encoder.load_state_dict(state['encoder'])
    encoder.requires_grad_(False)
    head = SlicedCostHead(config).to(config['device']).eval()
    head.load_state_dict(state['model'])
    condition = torch.tensor(args.condition, device=config['device'])[None, None]
    prediction = np.empty(len(coord), np.float32)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for row in metadata['blocks']:
        with np.load(output / 'geometry' / (row['id'] + '.npz')) as archive:
            context, core, point_index = archive['context_index'], archive['core_local'], archive['point_index']
        feature, inverse = encode_window(encoder, np.array(coord[context]), core, config)
        features = torch.from_numpy(feature).to(config['device'])
        batch = torch.zeros(len(feature), dtype=torch.long, device=config['device'])
        values = head(features, condition, batch)[:, 0].cpu().numpy()
        prediction[point_index] = values[inverse]
    prediction_s = time.perf_counter() - tick
    write_pcd(output / 'prediction.pcd', coord, prediction)
    np.save(output / 'prediction.npy', prediction)
    write_json(output / 'prediction.json', dict(checkpoint=str(args.checkpoint.resolve()),
               source_pcd=str(workspace_path(args.pcd)), condition=args.condition, points=len(coord),
               architecture=state['architecture'], selected_epoch=state['epoch'],
               output_coordinates='native merged ground XYZ, map metres', encoder_calls=len(metadata['blocks']),
               teacher_cost_computed=False, external_queries_used=False, knn_used=False,
               geometry_s=geometry_s, prediction_s=prediction_s, total_s=time.perf_counter()-started))
    print(output / 'prediction.pcd', flush=True)


if __name__ == '__main__':
    main()
