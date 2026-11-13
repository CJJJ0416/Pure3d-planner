"""Resume frozen-Tiny cost heads to epoch 100 with a replaceable scene pool."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from Traversability.data.rotating import disk_bytes, fixed_manifest, generation_seeds
from Traversability.data.sliced import progress
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.decoupling import decoupling_loss
from Traversability.training.sliced import CachedSlices, cache_features, device_batch, evaluate, loader
from Traversability.training.train import save_checkpoint
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def phase_learning_rate(config, epoch_progress):
    fraction = (epoch_progress-config['start_epoch']) / (config['epochs']-config['start_epoch'])
    warmup = config['warmup_fraction']
    if fraction < warmup:
        factor = fraction / warmup
    else:
        factor = .5 * (1 + math.cos(math.pi * (fraction-warmup) / (1-warmup)))
    return config['minimum_learning_rate'] + (config['learning_rate']-config['minimum_learning_rate']) * factor


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_random(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def prepare_pool(config, generation, completed_epoch, validation):
    root, output = workspace_path(config['data']), workspace_path(config['output'])
    stage = root / 'generation.json'
    local = deepcopy(config)
    local['progress_context'] = dict(completed_epochs=completed_epoch, generation=generation,
                                     generation_seeds=generation_seeds(config, generation),
                                     generations_total=(config['epochs']-config['start_epoch'])//config['rotation']['epochs_per_generation'],
                                     last_validation=validation)
    write_json(output / 'active_config.json', local)
    current = read_json(stage) if stage.exists() else None
    if current is None or current['index'] != generation or not current['prepared']:
        torch.cuda.empty_cache()
        progress(local, 'scene_generation')
        with (output / f'generation_{generation:02d}.log').open('a') as log:
            subprocess.run([config['geometry_python'], '-u', '-m', 'Traversability.data.rotating',
                            '--config', str(output / 'active_config.json'), '--generation', str(generation)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        current = read_json(stage)
    if not current['encoded']:
        cache_features(local)
        current.update(encoded=True, temporary_bytes=disk_bytes(root / 'active'))
        encoding = read_json(root / 'encoding.json')
        metadata = read_json(root / 'manifest.json')
        current['training_blocks'] = sum(b['split'] == 'train' for b in metadata['blocks'])
        current['encoding'] = encoding
        write_json(stage, current)
        write_json(output / 'generations' / f'{generation:02d}.json', current)
    return current


def train_epoch(head, optimizer, config, epoch, step, pool, history):
    data = CachedSlices(config, 'train', (epoch-1) % 2)
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    order = torch.randperm(len(data), generator=generator).tolist()
    if len(order) % config['batch_size']:
        order += order[:config['batch_size']-len(order) % config['batch_size']]
    batches_total = len(order)//config['batch_size']
    pairs = next(iter(data.scenes.values()))['pairs']
    output = workspace_path(config['output'])
    head.train()
    totals = defaultdict(float)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch_number, batch in enumerate(loader(data, config, order), 1):
        b = device_batch(batch, config)
        before = {k: v.detach().clone() for k, v in head.state_dict().items()} if not (output/'first_update.json').exists() else None
        prediction = head(b['features'], b['conditions'], b['voxel_batch'])[b['inverse']]
        loss, parts = decoupling_loss(prediction, b['target'], b['facility'], pairs, config)
        assert torch.isfinite(loss), 'Training loss is nonfinite.'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
                     for name, module in [('adapter',head.adapter),('condition_head',head.condition_head),('cost_head',head.cost_head)]} if before is not None else None
        epoch_progress = epoch-1+batch_number/batches_total
        rate = phase_learning_rate(config, epoch_progress)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.step()
        step += 1
        if before is not None:
            changes = {name: max(float(abs(value-before[key]).max()) for key, value in head.state_dict().items()
                                if key.startswith(name+'.')) for name in gradients}
            assert all(math.isfinite(v) and v > 0 for v in gradients.values())
            assert all(v > 0 for v in changes.values())
            write_json(output/'first_update.json', dict(epoch=epoch, step=step, true_batch=len(b['scene']),
                       conditions=b['conditions'].shape[1], loss=float(loss), gradients=gradients,
                       max_parameter_change=changes, tiny_in_optimizer=False,
                       encoder_in_optimizer=False, encoder_name=config['encoder']['name'],
                       trainable_parameters=sum(p.numel() for p in head.parameters())))
        for name, value in parts.items():
            totals[name] += float(value.detach())
        if step % config['log_every_steps'] == 0:
            progress(config, 'training', completed_epochs=epoch-1, epoch=epoch, step=step,
                     generation=pool['index'], generation_seeds=pool['seeds'],
                     generations_total=math.ceil((config.get('stop_epoch',config['epochs'])-config['start_epoch'])/config['rotation']['epochs_per_generation']),
                     batch=batch_number, batches_total=batches_total, updates_per_epoch=batches_total,
                     temporary_bytes=pool['temporary_bytes'], learning_rate=rate,
                     loss={k:v/batch_number for k,v in totals.items()}, last_validation=history[-1]['validation'])
            print(f'EPOCH {epoch}/{config["epochs"]} BATCH {batch_number}/{batches_total} STEP {step} LOSS {float(loss):.6f}', flush=True)
    return dict(epoch=epoch, step=step, bank=(epoch-1)%2, generation=pool['index'], seeds=pool['seeds'],
                training_blocks=len(data), updates=batches_total, loss={k:v/batches_total for k,v in totals.items()},
                learning_rate=rate, elapsed_s=time.perf_counter()-started,
                peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30), step


@torch.inference_mode()
def response_curves(head, config, split):
    data = CachedSlices(config, split, 0)
    totals = {}
    head.eval()
    for batch in loader(data, config):
        b = device_batch(batch, config)
        predicted = head(b['features'], b['conditions'], b['voxel_batch'])[b['inverse']]
        for i, name in enumerate(b['scene']):
            take = (b['point_batch']==i) & b['facility']
            if name not in totals:
                totals[name] = dict(prediction=np.zeros(40), truth=np.zeros(40), points=0)
            totals[name]['prediction'] += predicted[take].sum(0).cpu().numpy()
            totals[name]['truth'] += b['target'][take].sum(0).cpu().numpy()
            totals[name]['points'] += int(take.sum())
    records = {}
    for name, row in totals.items():
        records[name] = dict(points=row['points'], region='facility support points',
                             conditions=np.load(data.root/name/'conditions.npy')[data.scenes[name]['banks'][0]].tolist(),
                             prediction=(row['prediction']/row['points']).tolist(),
                             truth=(row['truth']/row['points']).tolist())
    return records


def train(config, resume=False, stop_after_epoch=None):
    output, root, source = (workspace_path(config[k]) for k in ['output','data','source_run'])
    output.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    write_json(output/'config.json', config)
    if not (root/'manifest.json').exists():
        write_json(root/'manifest.json', fixed_manifest(config))
    head = SlicedCostHead(config).to(config['device'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    state = torch.load(output/'last.pt' if resume else source/'last.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    encoder_state = state['encoder']
    reference_encoder = torch.load(source/'best.pt', map_location='cpu', weights_only=False)['encoder']
    assert all(torch.equal(v, reference_encoder[k]) for k,v in encoder_state.items())
    start_epoch, step = state['epoch']+1, state['step']
    if resume:
        restore_random(state['random_state'])
        history = read_json(output/'history.json')[:state['epoch']]
        best_score = state['best_score']
        initial = read_json(output/'initial_val.json')
        previous = read_json(output/'resume_verification.json')
        previous['continuations'].append(dict(from_epoch=state['epoch'], from_step=step,
            optimizer_steps=sorted({int(v['step']) for v in optimizer.state.values()}), scheduler=state['scheduler']))
        write_json(output/'resume_verification.json', previous)
    else:
        assert state['epoch'] == config['start_epoch']
        history = read_json(source/'history.json')
        initial = read_json(source/'initial_val.json')
        write_json(output/'initial_val.json', initial)
        last_validation = evaluate(head, config, 'val')
        candidate = torch.load(source/'best.pt', map_location='cpu', weights_only=False)
        head.load_state_dict(candidate['model'])
        best_validation = evaluate(head, config, 'val')
        assert abs(best_validation['score']-read_json(source/'summary.json')['best_validation']['score']) < 1e-6
        best_score = best_validation['score']
        candidate.update(config=config, best_score=best_score, origin_checkpoint=str(source/'best.pt'))
        save_checkpoint(output/'best.pt', candidate)
        write_json(output/'selection.json', dict(epoch=candidate['epoch'], score=best_score, source='original_best',
                   baseline_validation=best_validation, validation=best_validation))
        write_json(output/'baseline.json', dict(last_epoch=state['epoch'], last_validation=last_validation,
                   best_epoch=candidate['epoch'], best_validation=best_validation, initial_validation=initial))
        write_json(output/'baseline_responses.json', response_curves(head, config, 'val'))
        head.load_state_dict(state['model'])
        write_json(output/'resume_verification.json', dict(source_epoch=state['epoch'], source_step=step,
                   source_optimizer_steps=sorted({int(v['step']) for v in optimizer.state.values()}),
                   source_scheduler=state['scheduler'], continuations=[]))
    def payload(epoch, pool):
        return dict(architecture=config['architecture'], config=config, model=head.state_dict(), encoder=encoder_state,
                    optimizer=optimizer.state_dict(), scheduler=dict(kind='epoch_fraction_warmup_cosine',
                        phase_start=config['start_epoch'], phase_end=config['epochs'], epoch_progress=epoch,
                        learning_rate=optimizer.param_groups[0]['lr']), epoch=epoch, step=step, best_score=best_score,
                    random_state=random_state(), generation=pool, conditions=config['condition_names'],
                    condition_scale=config['condition_scale'], source_checkpoint=str(source/'last.pt'),
                    geometry_contract=state['geometry_contract'])
    started = time.perf_counter()
    for epoch in range(start_epoch, config['epochs']+1):
        generation = (epoch-config['start_epoch']-1)//config['rotation']['epochs_per_generation']
        pool = prepare_pool(config, generation, epoch-1, history[-1]['validation'])
        row, step = train_epoch(head, optimizer, config, epoch, step, pool, history)
        validation = evaluate(head, config, 'val')
        row['validation'] = validation
        history.append(row)
        if validation['score'] < best_score:
            best_score = validation['score']
            save_checkpoint(output/'best.pt', payload(epoch, pool))
            write_json(output/'selection.json', dict(epoch=epoch, score=best_score, source='rotating_training',
                       baseline_validation=read_json(output/'baseline.json')['best_validation'], validation=validation))
        save_checkpoint(output/'last.pt', payload(epoch, pool))
        write_json(output/'history.json', history)
        progress(config, 'training', completed_epochs=epoch, epoch=epoch, step=step,
                 generation=generation, generation_seeds=pool['seeds'], generations_total=10,
                 loss=row['loss'], last_validation=validation, best_score=best_score,
                 temporary_bytes=pool['temporary_bytes'], learning_rate=row['learning_rate'])
        print(f'VALIDATION epoch={epoch} raw_MAE={validation["cost_mae_raw"]:.4f} score={validation["score"]:.6f}', flush=True)
        gc.collect()
        if stop_after_epoch is not None and epoch == stop_after_epoch:
            print('CONTROLLED CHECKPOINT STOP', epoch, flush=True)
            return
    selected = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(selected['model'])
    test = evaluate(head, config, 'test', output/'predictions')
    validation = evaluate(head, config, 'val', output/'validation_predictions')
    write_json(output/'final_responses.json', response_curves(head, config, 'val'))
    write_json(output/'summary.json', dict(selected_epoch=selected['epoch'], completed_epochs=config['epochs'],
        initial_validation=initial, baseline=read_json(output/'baseline.json'), best_validation=validation, test=test,
        original_test=read_json(source/'summary.json')['test'], best_checkpoint=str(output/'best.pt'),
        last_checkpoint=str(output/'last.pt'), training_s=time.perf_counter()-started,
        new_scenes=40, generations=10, initial_best_retained=selected['epoch']<=config['start_epoch']))
    last = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
    source_last = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
    verification = dict(completed_epoch=last['epoch'], last_step=last['step'], selected_epoch=selected['epoch'],
        full_100_epochs=[r['epoch'] for r in history]==list(range(1,config['epochs']+1)),
        encoder_weights_and_buffers_equal_source=all(torch.equal(v,source_last['encoder'][k]) for k,v in last['encoder'].items()),
        best_encoder_equal_source=all(torch.equal(v,source_last['encoder'][k]) for k,v in selected['encoder'].items()),
        optimizer_steps=sorted({int(v['step']) for v in last['optimizer']['state'].values()}),
        expected_step=source_last['step']+sum(r['updates'] for r in history if r['epoch']>config['start_epoch']),
        final_learning_rate=last['optimizer']['param_groups'][0]['lr'], scenes_archived=len(list((output/'scene_archive').glob('scene_*'))),
        rotations_verified=len(list((output/'rotations').glob('*.json'))), source_last_sha256=hashlib.sha256((source/'last.pt').read_bytes()).hexdigest())
    assert verification['full_100_epochs'] and verification['encoder_weights_and_buffers_equal_source']
    assert verification['optimizer_steps']==[verification['expected_step']]
    write_json(output/'training_verification.json', verification)
    progress(config, 'evaluation_complete', completed_epochs=config['epochs'], selected_epoch=selected['epoch'],
             last_validation=validation, test=test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    torch.set_num_threads(config['cpu_threads'])
    train(config, args.resume, args.stop_after_epoch)


if __name__ == '__main__':
    main()
