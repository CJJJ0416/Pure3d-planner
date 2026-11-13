"""A fresh 100-epoch head phase on complete 3-D frozen Tiny features."""
import argparse
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

from Traversability.data.full3d import prepare_fixed
from Traversability.data.rotating import disk_bytes, fixed_manifest, generation_seeds
from Traversability.data.sliced import progress
from Traversability.models.sliced import SlicedCostHead
from Traversability.training.full3d_cache import cache_features
from Traversability.training.rotating import random_state, restore_random, response_curves, train_epoch, phase_learning_rate
from Traversability.training.sliced import evaluate
from Traversability.training.train import save_checkpoint
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def prepare_pool(config, generation, completed_epoch, validation):
    root, output = workspace_path(config['data']), workspace_path(config['output'])
    path = root/'generation.json'
    local = deepcopy(config)
    local['progress_context'] = dict(completed_epochs=completed_epoch, generation=generation,
        generation_seeds=generation_seeds(config,generation), generations_total=10, last_validation=validation)
    write_json(output/'active_config.json', local)
    current = read_json(path) if path.exists() else None
    if current is None or current['index']!=generation or not current['prepared']:
        progress(local,'scene_generation')
        torch.cuda.empty_cache()
        with (output/f'generation_{generation:02d}.log').open('a') as stream:
            subprocess.run([config['geometry_python'],'-u','-m','Traversability.data.full3d',
                '--config',str(output/'active_config.json'),'--generation',str(generation)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        current = read_json(path)
    if not current['encoded']:
        encoding = cache_features(local)
        meta = read_json(root/'manifest.json')
        current.update(encoded=True, temporary_bytes=disk_bytes(root/'active'),
            training_blocks=sum(r['split']=='train' for r in meta['blocks']),encoding=encoding)
        write_json(path,current)
        write_json(output/'generations'/f'{generation:02d}.json',current)
    return current


def train(config, resume=False, stop_after_epoch=None):
    output,root,source = (workspace_path(config[k]) for k in ['output','data','source_run'])
    output.mkdir(parents=True,exist_ok=True)
    root.mkdir(parents=True,exist_ok=True)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    write_json(output/'config.json',config)
    initial_state = torch.load(source/'best.pt',map_location='cpu',weights_only=False)
    assert initial_state['epoch']==90
    encoder_state = initial_state['encoder']
    head = SlicedCostHead(config).to(config['device'])
    optimizer = torch.optim.AdamW(head.parameters(),lr=config['minimum_learning_rate'],weight_decay=config['weight_decay'])
    if resume:
        state = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        head.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_random(state['random_state'])
        start,step,best_score = state['epoch']+1,state['step'],state['best_score']
        history = read_json(output/'history.json')[:state['epoch']]
        initial = read_json(output/'initial_val.json')
        record = read_json(output/'resume_verification.json')
        record['continuations'].append(dict(epoch=state['epoch'],step=step,
            optimizer_steps=sorted({int(v['step']) for v in optimizer.state.values()}),
            scheduler=state['scheduler'],generation=state['generation']['index'],random_state_restored=True))
        write_json(output/'resume_verification.json',record)
    else:
        head.load_state_dict(initial_state['model'])
        fixed = deepcopy(config)
        fixed['fixed_data'] = str(prepare_fixed(config))
        if not (root/'manifest.json').exists():
            write_json(root/'manifest.json',fixed_manifest(fixed))
        encoding = cache_features(config)
        initial = evaluate(head,config,'val',output/'initial_validation_predictions')
        write_json(output/'initial_val.json',initial)
        write_json(output/'baseline.json',dict(source_epoch=90,best_validation=initial,
            old_sliced_validation=read_json(source/'summary.json')['best_validation'],
            note='The full3D initialization is the selection baseline. Old sliced scores are reference only.'))
        write_json(output/'baseline_responses.json',response_curves(head,config,'val'))
        history,start,step,best_score = [],1,0,initial['score']
        write_json(output/'resume_verification.json',dict(source_epoch=90,fresh_optimizer=True,
            optimizer_states_at_initialization=len(optimizer.state),initial_step=0,continuations=[]))
        print('FULL3D INITIAL VALIDATION',initial['cost_mae_raw'],initial['score'],flush=True)
    def payload(epoch,pool):
        return dict(architecture=config['architecture'],config=config,model=head.state_dict(),encoder=encoder_state,
            optimizer=optimizer.state_dict(),scheduler=dict(kind='epoch_fraction_warmup_cosine',
                phase_start=0,phase_end=config['epochs'],epoch_progress=epoch,
                learning_rate=optimizer.param_groups[0]['lr']),epoch=epoch,step=step,best_score=best_score,
            random_state=random_state(),generation=pool,conditions=config['condition_names'],
            condition_scale=config['condition_scale'],source_checkpoint=str(source/'best.pt'),source_epoch=90,
            geometry_contract='Full PCD, true representatives at 5 cm, complete 3D contexts; native support queries; map metres')
    if not resume:
        save_checkpoint(output/'best.pt',payload(0,None))
        write_json(output/'selection.json',dict(epoch=0,score=best_score,source='full3d_initialization',validation=initial))
        write_json(output/'history.json',history)
    started = time.perf_counter()
    target = config.get('stop_epoch', config['epochs'])
    generations = math.ceil(target/config['rotation']['epochs_per_generation'])
    for epoch in range(start,target+1):
        generation = (epoch-1)//config['rotation']['epochs_per_generation']
        validation = history[-1]['validation'] if history else initial
        pool = prepare_pool(config,generation,epoch-1,validation)
        # train_epoch uses the unchanged cached-query dataset, loss, batch and LR phase.
        row,step = train_epoch(head,optimizer,config,epoch,step,pool,
                             history if history else [dict(validation=initial)])
        validation = evaluate(head,config,'val')
        row['validation'] = validation
        history.append(row)
        if validation['score']<best_score:
            best_score=validation['score']
            save_checkpoint(output/'best.pt',payload(epoch,pool))
            write_json(output/'selection.json',dict(epoch=epoch,score=best_score,source='full3d_training',validation=validation))
        save_checkpoint(output/'last.pt',payload(epoch,pool))
        write_json(output/'history.json',history)
        progress(config,'training',completed_epochs=epoch,epoch=epoch,step=step,generation=generation,
            generation_seeds=pool['seeds'],generations_total=10,loss=row['loss'],last_validation=validation,
            best_score=best_score,temporary_bytes=pool['temporary_bytes'],learning_rate=row['learning_rate'])
        print(f'VALIDATION epoch={epoch} raw_MAE={validation["cost_mae_raw"]:.4f} score={validation["score"]:.6f}',flush=True)
        gc.collect()
        if stop_after_epoch is not None and epoch==stop_after_epoch:
            print('CONTROLLED CHECKPOINT STOP',epoch,flush=True)
            return
    selected = torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    head.load_state_dict(selected['model'])
    test = evaluate(head,config,'test',output/'predictions')
    validation = evaluate(head,config,'val',output/'validation_predictions')
    write_json(output/'final_responses.json',response_curves(head,config,'val'))
    head.load_state_dict(initial_state['model'])
    initial_test = evaluate(head,config,'test',output/'initial_predictions')
    write_json(output/'summary.json',dict(selected_epoch=selected['epoch'],completed_epochs=target,
        initial_validation=initial,best_validation=validation,test=test,initial_test=initial_test,
        original_test=read_json(source/'summary.json')['test'],baseline=read_json(output/'baseline.json'),
        best_checkpoint=str(output/'best.pt'),last_checkpoint=str(output/'last.pt'),
        training_s=time.perf_counter()-started,new_scenes=4*generations,generations=generations,
        scheduler_budget_epochs=config['epochs'],initial_best_retained=selected['epoch']==0))
    last = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    verification = dict(completed_epoch=last['epoch'],last_step=last['step'],selected_epoch=selected['epoch'],
        expected_epochs_complete=[r['epoch'] for r in history]==list(range(1,target+1)),
        target_epoch=target,scheduler_budget_epochs=config['epochs'],
        encoder_weights_and_buffers_equal_source=all(torch.equal(v,encoder_state[k]) for k,v in last['encoder'].items()),
        best_encoder_equal_source=all(torch.equal(v,encoder_state[k]) for k,v in selected['encoder'].items()),
        optimizer_steps=sorted({int(v['step']) for v in last['optimizer']['state'].values()}),
        expected_step=sum(r['updates'] for r in history),final_learning_rate=last['optimizer']['param_groups'][0]['lr'],
        scenes_archived=len(list((output/'scene_archive').glob('scene_*'))),
        rotations_verified=len(list((output/'rotations').glob('*.json'))),
        source_sha256=hashlib.sha256((source/'best.pt').read_bytes()).hexdigest())
    assert verification['expected_epochs_complete'] and verification['encoder_weights_and_buffers_equal_source']
    assert verification['optimizer_steps']==[verification['expected_step']]
    assert math.isclose(verification['final_learning_rate'],phase_learning_rate(config,target))
    write_json(output/'training_verification.json',verification)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    config=read_json(args.config)
    torch.set_num_threads(config['cpu_threads'])
    train(config,args.resume,args.stop_after_epoch)


if __name__=='__main__':
    main()
