"""Frozen Tiny feature caching and 50-epoch point-aligned cost training."""
import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import point_transformer_v3 as ptv3
import torch
from torch.utils.data import Dataset, DataLoader

from Traversability.data.sliced import progress
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.condition_response import response_totals
from Traversability.training.decoupling import decoupling_loss, scene_result, macro
from Traversability.training.train import save_checkpoint
from common.pointcloud_io import write_pcd
from utils.pointcloud import read_json, write_json, workspace_path


def load_encoder(config):
    spec = config['encoder']
    encoder = ptv3.load(spec['name'], weights=workspace_path(spec['weights']),
                        device=config['device'], head=False, precision='fp32', warmup=False).eval()
    encoder.requires_grad_(False)
    return encoder


@torch.inference_mode()
def encode_window(encoder, coord, core, config):
    spec = config['encoder']
    zero = np.zeros_like(coord)
    point, inverse = ptv3.prepare(coord, zero, zero, preset=spec['name'],
                                  grid_size=spec['grid_size'], scale=spec['scale'],
                                  pick='first', device=config['device'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feature = ptv3.upcast(encoder(point)).feat
    core_inverse = inverse[torch.as_tensor(core, device=config['device'])]
    used, back = torch.unique(core_inverse, sorted=True, return_inverse=True)
    return feature[used].half().cpu().numpy(), back.int().cpu().numpy()


def cache_features(config):
    root = workspace_path(config['data'])
    metadata = read_json(root / 'manifest.json')
    encoder = load_encoder(config)
    original = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    scenes = {s['id']: s for s in metadata['scenes']}
    coords = {name: np.load(root / name / 'coord.npy', mmap_mode='r') for name in scenes}
    started = time.perf_counter()
    for i, row in enumerate(metadata['blocks']):
        folder = root / row['scene'] / 'features' / row['id']
        if (folder / 'ready.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(root / row['scene'] / (row['id'] + '.npz')) as archive:
            context, core = archive['context_index'], archive['core_local']
        tick = time.perf_counter()
        feature, inverse = encode_window(encoder, np.array(coords[row['scene']][context]), core, config)
        np.save(folder / 'features.npy', feature)
        np.save(folder / 'inverse.npy', inverse)
        write_json(folder / 'ready.json', dict(points=len(inverse), voxels=len(feature),
                   feature_channels=feature.shape[1], encoding_s=time.perf_counter() - tick))
        if (i + 1) % 20 == 0:
            progress(config, 'feature_cache', blocks_completed=i+1, blocks_total=len(metadata['blocks']))
            print(f"ENCODE {i+1}/{len(metadata['blocks'])} | {time.perf_counter()-started:.1f}s", flush=True)
    unchanged = all(torch.equal(original[k], value.cpu()) for k, value in encoder.state_dict().items())
    assert unchanged, 'Frozen Tiny weights and buffers changed.'
    write_json(root / 'encoding.json', dict(complete=True, encoder=config['encoder'],
               pretrained_sha256=hashlib.sha256(workspace_path(config['encoder']['weights']).read_bytes()).hexdigest(),
               frozen_weights_and_buffers_equal=unchanged, blocks=len(metadata['blocks']),
               encoder_passes_per_new_context=1, encoding_s=time.perf_counter()-started))
    print('FEATURE CACHE COMPLETE', flush=True)


class CachedSlices(Dataset):
    def __init__(self, config, split, bank):
        self.root = workspace_path(config['data'])
        self.meta = read_json(self.root / 'manifest.json')
        self.scenes = {s['id']: s for s in self.meta['scenes']}
        self.rows = [b for b in self.meta['blocks'] if b['split'] == split]
        self.bank = bank
        self.arrays = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        name = row['scene']
        if name not in self.arrays:
            self.arrays[name] = {k: np.load(self.root / name / (k + '.npy'), mmap_mode='r')
                                 for k in ['cost', 'conditions', 'facility']}
        arrays = self.arrays[name]
        columns = self.scenes[name]['banks'][self.bank]
        with np.load(self.root / name / (row['id'] + '.npz')) as archive:
            point_index = archive['point_index']
        folder = self.root / name / 'features' / row['id']
        return dict(features=torch.from_numpy(np.load(folder / 'features.npy')),
                    inverse=torch.from_numpy(np.load(folder / 'inverse.npy').astype(np.int64)),
                    target=torch.from_numpy(np.array(arrays['cost'][point_index][:, columns], np.float32)),
                    conditions=torch.from_numpy(np.array(arrays['conditions'][columns], np.float32)),
                    facility=torch.from_numpy(np.array(arrays['facility'][point_index])),
                    point_index=point_index, scene=name)


def collate(samples):
    counts = [len(s['features']) for s in samples]
    offsets = np.cumsum([0, *counts[:-1]])
    return dict(features=torch.cat([s['features'] for s in samples]),
                inverse=torch.cat([s['inverse'] + int(o) for s, o in zip(samples, offsets)]),
                voxel_batch=torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)]),
                point_batch=torch.cat([torch.full((len(s['inverse']),), i, dtype=torch.long) for i, s in enumerate(samples)]),
                conditions=torch.stack([s['conditions'] for s in samples]),
                target=torch.cat([s['target'] for s in samples]),
                facility=torch.cat([s['facility'] for s in samples]),
                point_index=[s['point_index'] for s in samples], scene=[s['scene'] for s in samples])


def loader(dataset, config, order=None):
    return DataLoader(dataset, batch_size=config['batch_size'], sampler=order,
                      num_workers=config['num_workers'], pin_memory=True, collate_fn=collate)


def device_batch(batch, config):
    return {k: v.to(config['device'], non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@torch.inference_mode()
def evaluate(head, config, split, export=None):
    head.eval()
    records = defaultdict(lambda: dict(absolute=0., count=0, response=np.zeros((4, 10), np.float64)))
    regional = np.zeros((2, 2), np.float64)
    predictions = {}
    for bank in range(2):
        data = CachedSlices(config, split, bank)
        pairs = next(iter(data.scenes.values()))['pairs']
        for batch in loader(data, config):
            b = device_batch(batch, config)
            prediction = head(b['features'], b['conditions'], b['voxel_batch'])[b['inverse']]
            target = b['target']
            error = abs(prediction - target)
            for group, mask in enumerate([b['facility'], ~b['facility']]):
                regional[group] += [float(error[mask].sum()), int(mask.sum()) * target.shape[1]]
            for i, name in enumerate(b['scene']):
                mask = b['point_batch'] == i
                rec = records[name]
                rec['absolute'] += float(error[mask].sum())
                rec['count'] += int(mask.sum()) * target.shape[1]
                rec['response'] += response_totals(prediction[mask], target[mask], pairs,
                                                   config['change_threshold']).cpu().numpy()
                if export is not None and bank == 0:
                    if name not in predictions:
                        predictions[name] = np.empty(data.scenes[name]['points'], np.float32)
                    predictions[name][b['point_index'][i]] = prediction[mask, 3].cpu().numpy()
    per_scene = {name: scene_result(r) for name, r in records.items()}
    result = macro(list(per_scene.values()))
    result.update(per_scene=per_scene,
                  facility_mae_raw=float(50 * regional[0, 0] / regional[0, 1]) if regional[0, 1] else None,
                  background_mae_raw=float(50 * regional[1, 0] / regional[1, 1]) if regional[1, 1] else None)
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
        for name, values in predictions.items():
            coord = np.load(workspace_path(config['data']) / name / 'coord.npy')
            write_pcd(export / f'{name}_prediction.pcd', coord, values)
            np.save(export / f'{name}_prediction.npy', values)
    return result


def train(config, resume=False, stop_after_epoch=None):
    output = workspace_path(config['output'])
    torch.manual_seed(config['seed'])
    head = SlicedCostHead(config).to(config['device'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    first_data = CachedSlices(config, 'train', 0)
    updates = math.ceil(len(first_data) / config['batch_size'])
    total_updates = updates * config['epochs']
    warmup = max(1, int(total_updates * config['warmup_fraction']))
    minimum = config['minimum_learning_rate'] / config['learning_rate']

    def schedule(step):
        if step < warmup:
            return minimum + (1-minimum) * step / warmup
        portion = (step-warmup) / (total_updates-warmup)
        return minimum + (1-minimum) * .5 * (1 + math.cos(math.pi * portion))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    encoder_state = {k: v.cpu() for k, v in load_encoder(config).state_dict().items()}
    history, start_epoch, step = [], 1, 0
    initial, best_score = None, float('inf')

    def payload(epoch):
        return dict(architecture=config['architecture'], config=config, model=head.state_dict(),
                    encoder=encoder_state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    epoch=epoch, step=step, best_score=best_score,
                    geometry_contract='native geometry-only slicing, float16 ground export, height merge; XYZ map metres',
                    conditions=config['condition_names'], condition_scale=config['condition_scale'])

    if resume:
        state = torch.load(output / 'last.pt', map_location=config['device'], weights_only=False)
        head.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, step, best_score = state['epoch'] + 1, state['step'], state['best_score']
        history = read_json(output / 'history.json')
        initial = read_json(output / 'initial_val.json')
    else:
        initial = evaluate(head, config, 'val')
        best_score = initial['score']
        write_json(output / 'initial_val.json', initial)
        save_checkpoint(output / 'best.pt', payload(0))
    print('INITIAL VALIDATION', initial['cost_mae_raw'], initial['score'], flush=True)
    started = time.perf_counter()
    pairs = next(iter(first_data.scenes.values()))['pairs']
    for epoch in range(start_epoch, config['epochs'] + 1):
        data = CachedSlices(config, 'train', (epoch - 1) % 2)
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        order = torch.randperm(len(data), generator=generator).tolist()
        if len(order) % config['batch_size']:
            order += order[:config['batch_size'] - len(order) % config['batch_size']]
        head.train()
        totals = defaultdict(float)
        batches = 0
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        for batch in loader(data, config, order):
            b = device_batch(batch, config)
            prediction = head(b['features'], b['conditions'], b['voxel_batch'])[b['inverse']]
            loss, parts = decoupling_loss(prediction, b['target'], b['facility'], pairs, config)
            assert torch.isfinite(loss), 'Training loss is nonfinite.'
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                gradient = sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None) ** .5
                write_json(output / 'first_update.json', dict(loss=float(loss), gradient_norm=gradient,
                           trainable_parameters=sum(p.numel() for p in head.parameters()), tiny_in_optimizer=False,
                           true_batch=len(b['scene']), bank_conditions=b['conditions'].shape[1]))
            optimizer.step()
            scheduler.step()
            step += 1
            batches += 1
            for name, value in parts.items():
                totals[name] += float(value.detach())
            if step % config['log_every_steps'] == 0:
                progress(config, 'training', epoch=epoch, completed_epochs=epoch-1, step=step,
                         updates_per_epoch=updates, batch=batches, batches_total=updates,
                         loss={k:v/batches for k,v in totals.items()},
                         last_validation=history[-1]['validation'] if history else initial)
                print(f'EPOCH {epoch}/50 STEP {step} LOSS {float(loss):.6f}', flush=True)
        validation = evaluate(head, config, 'val')
        row = dict(epoch=epoch, step=step, bank=(epoch-1)%2,
                   loss={k:v/batches for k,v in totals.items()}, validation=validation,
                   elapsed_s=time.perf_counter()-tick, learning_rate=optimizer.param_groups[0]['lr'],
                   peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
        history.append(row)
        if validation['score'] < best_score:
            best_score = validation['score']
            save_checkpoint(output / 'best.pt', payload(epoch))
        save_checkpoint(output / 'last.pt', payload(epoch))
        write_json(output / 'history.json', history)
        progress(config, 'training', epoch=epoch, completed_epochs=epoch, step=step,
                 loss=row['loss'], last_validation=validation, best_score=best_score)
        print(f"VALIDATION epoch={epoch} raw_MAE={validation['cost_mae_raw']:.4f} score={validation['score']:.6f} epoch_s={row['elapsed_s']:.1f}", flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < config['epochs']:
            print('CONTROLLED CHECKPOINT STOP', epoch, flush=True)
            return
    selected = torch.load(output / 'best.pt', map_location=config['device'], weights_only=False)
    head.load_state_dict(selected['model'])
    test = evaluate(head, config, 'test', output / 'predictions')
    validation = evaluate(head, config, 'val', output / 'validation_predictions')
    write_json(output / 'summary.json', dict(selected_epoch=selected['epoch'], completed_epochs=config['epochs'],
               initial_validation=initial, best_validation=validation, test=test,
               best_checkpoint=str(output/'best.pt'), last_checkpoint=str(output/'last.pt'),
               training_s=time.perf_counter()-started, encoding=read_json(workspace_path(config['data'])/'encoding.json')))
    progress(config, 'complete', status_detail='50 epochs and held-out building evaluation complete',
             completed_epochs=config['epochs'], selected_epoch=selected['epoch'], last_validation=validation, test=test)
    print('TRAINING AND TEST COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['encode', 'train'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    torch.set_num_threads(config['cpu_threads'])
    if args.stage == 'encode':
        cache_features(config)
    else:
        train(config, args.resume, args.stop_after_epoch)


if __name__ == '__main__':
    main()
