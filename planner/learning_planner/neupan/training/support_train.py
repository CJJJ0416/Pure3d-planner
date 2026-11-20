"""Train and evaluate the policy on independent spatial splits."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from planner.learning_planner.neupan.data.expert import fingerprint
from planner.learning_planner.neupan.models.support_policy import SupportPolicy, batch, choose_actions, decode
from planner.learning_planner.neupan.inference.path_generator import reached
from planner.learning_planner.neupan.geometry.tomogram import STOP
from planner.learning_planner.neupan.data.supervision import Windows


class Samples:
    def __init__(self, directories, split, pairs=None):
        if pairs and len(pairs) != len(directories):
            raise ValueError('Pass one paired cloud manifest per dataset')
        sources = [Windows(path, split, pairs[i] if pairs else None)
                   for i, path in enumerate(directories)]
        self.entries = [(source, index) for source in sources for index in range(len(source))]
        if not self.entries:
            raise ValueError(f'No {split} windows available in {directories}')
        self.get = lru_cache(maxsize=96)(self._get)

    def _get(self, index):
        source, local = self.entries[index]
        return source[local]

    def __len__(self):
        return len(self.entries)


def window_batches(samples, indices, batch_size, workers):
    groups = iter(indices[first:first + batch_size] for first in range(0, len(indices), batch_size))
    if workers == 0:
        for group in groups:
            yield [samples.get(int(i)) for i in group]
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scheduled = [pool.submit(samples.get, int(i)) for i in next(groups, ())]
        for group in groups:
            ready = [future.result() for future in scheduled]
            scheduled = [pool.submit(samples.get, int(i)) for i in group]
            yield ready
        if scheduled:
            yield [future.result() for future in scheduled]


def action_epoch(model, samples, indices, batch_size, device, optimizer=None, stop_weight=1., workers=0):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = actions = stop_true = stop_pred = stop_hit = 0
    for windows in window_batches(samples, indices, batch_size, workers):
        packed = batch(windows, device, model.schema)
        if packed is None:
            continue
        x, edges, current, goal, visited, target = packed
        with torch.set_grad_enabled(training):
            scores = model.scores(model.encode(x, edges), x[:, :3], edges, current, goal, visited)
            if model.schema == 1:
                loss_each = F.cross_entropy(scores, target, reduction='none')
            else:
                # Continue/STOP is binary. Direction cross entropy only sees move
                # examples, so eight reasonable neighbors do not inflate STOP.
                moving = target != STOP
                eligible = scores[:, :STOP].isfinite().any(dim=1)
                stop_loss = F.binary_cross_entropy_with_logits(
                    scores[:, STOP], (~moving).float(), reduction='none')
                stop_cost = torch.where(eligible, stop_loss, 0.)
                edge_logits = scores[:, :STOP].masked_fill(~moving[:, None], 0.)
                edge_cost = F.cross_entropy(edge_logits, target.clamp(max=STOP - 1), reduction='none')
                loss_each = stop_cost + torch.where(moving, edge_cost, 0.)
            weight = torch.where(target == STOP, stop_weight, 1.)
            loss = (loss_each * weight).sum() / weight.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        selected = choose_actions(scores, model.schema, model.stop_logit_threshold)
        correct += int((selected == target).sum())
        stop_true += int((target == STOP).sum())
        stop_pred += int((selected == STOP).sum())
        stop_hit += int(((selected == STOP) & (target == STOP)).sum())
        actions += len(target)
        loss_sum += float(loss_each.detach().sum())
    if actions == 0:
        raise ValueError('The selected split has no supervised actions')
    return dict(loss=loss_sum / actions, accuracy=correct / actions,
                stop_precision=stop_hit / stop_pred if stop_pred else None,
                stop_recall=stop_hit / stop_true if stop_true else None,
                stop_labels=stop_true, stop_predictions=stop_pred, actions=actions)


def distribution(timings):
    return dict(p50=float(np.percentile(timings, 50)), p95=float(np.percentile(timings, 95)))


def rollouts(model, samples, max_windows=None, *, guard=True):
    model.eval()
    counters = Counter()
    by_scenario = {}
    lengths = {}
    build_times, decode_times = [], []
    index = np.arange(len(samples))
    if max_windows and len(index) > max_windows:
        index = np.linspace(0, len(index) - 1, num=max_windows, dtype=int)
    for i in index:
        source, local = samples.entries[int(i)]
        window = source[local]  # Measure a fresh build, independent of the training cache.
        build_times.append(window.build_ms)
        graph = window.graph
        start = int(window.chain[0]) if len(window.chain) else -1
        before = perf_counter()
        chain, reason = decode(model, graph, start, window.goal, guard=guard)
        decode_times.append((perf_counter() - before) * 1000.)
        tally = by_scenario.setdefault(window.scenario, Counter())
        tally['windows'] += 1
        tally[reason] += 1
        learned_xyz, expert_xyz = graph.xyz[chain], graph.xyz[window.chain]
        lengths.setdefault(window.scenario, []).append([
            float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())
            for points in (learned_xyz, expert_xyz)])
        if window.end == 'support_blocked':
            tally['blocked_prefix_halted'] += int(reason in ('support_blocked', 'stop_selected'))
        counters['windows'] += 1
        counters[f'ended_{reason}'] += 1
        if window.end == 'goal_reached':
            counters['expert_goal_windows'] += 1
            counters['premature_stop'] += int(reason in ('support_blocked', 'stop_selected'))
            if reason == 'goal_reached':
                counters['reached_goal'] += 1
        if chain:
            last = graph.xyz[chain[-1]]
            if np.linalg.norm(last[:2] - window.goal[:2]) <= .14 and not reached(last, window.goal):
                counters['wrong_goal_height'] += 1
            for a, b in zip(chain, chain[1:]):
                if b not in graph.neighbors[a]:
                    counters['illegal_edges'] += 1
                if (graph.cells[a, 0] != graph.cells[b, 0]
                        and abs(graph.xyz[a, 2] - graph.xyz[b, 2]) > .300001):
                    counters['wrong_layer_edges'] += 1
        tally[f'expert_{window.end}'] += 1
    if counters['illegal_edges']:
        raise AssertionError(f'Policy traversed {counters["illegal_edges"]} illegal edges')
    denominator = counters['expert_goal_windows']
    return dict(guard=guard, counts={**dict.fromkeys(('illegal_edges', 'wrong_layer_edges', 'wrong_goal_height',
                                       'reached_goal', 'expert_goal_windows'), 0), **counters},
                goal_rate=counters['reached_goal'] / denominator if denominator else None,
                by_scenario={key: dict(value, learned_length_p50_m=float(np.median(lengths[key], axis=0)[0]),
                                       expert_length_p50_m=float(np.median(lengths[key], axis=0)[1]))
                             for key, value in by_scenario.items()},
                graph_prepare_ms=distribution(build_times), decode_ms=distribution(decode_times))


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training = Samples(args.train, 'train', args.train_pairs)
    validation = Samples(args.validation, 'validation', args.validation_pairs)
    model = SupportPolicy().to(args.device)
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        if state['schema'] != model.schema:
            raise ValueError('Incompatible checkpoint')
        model.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=False)
    observations = []
    train_stream = rng.permutation(len(training))
    train_limit = min(args.train_windows or len(training), len(training))
    validation_indices = np.arange(len(validation))
    if args.validation_windows is not None and len(validation_indices) > args.validation_windows:
        validation_indices = np.linspace(0, len(validation) - 1, num=args.validation_windows, dtype=int)
    used = np.zeros(len(training), dtype=bool)
    cursor = 0
    best, best_epoch = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        if cursor == len(train_stream):
            train_stream, cursor = rng.permutation(len(training)), 0
        indices = train_stream[cursor:cursor + train_limit]
        cursor += len(indices)
        used[indices] = True
        metrics = dict(epoch=epoch, unique_train_windows=int(used.sum()),
                       train=action_epoch(model, training, indices, args.batch_size, args.device,
                                          optimizer, args.stop_weight, workers=args.workers),
                       validation=action_epoch(model, validation, validation_indices,
                                               args.batch_size, args.device, workers=args.workers))
        observations.append(metrics)
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
        val_loss = metrics['validation']['loss']
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            pending = args.output / 'best.pt.tmp'
            torch.save(dict(schema=model.schema, model=model.state_dict(), epoch=epoch, seed=args.seed), pending)
            pending.replace(args.output / 'best.pt')
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = args.output / 'best.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True)['model'])
    report = dict(history=observations, best_epoch=best_epoch, checkpoint_sha256=fingerprint(checkpoint),
                  seed=args.seed, stop_weight=args.stop_weight, initial_checkpoint_sha256=fingerprint(args.resume) if args.resume else None,
                  train_manifests=[fingerprint(p / 'manifest.json') for p in args.train],
                  validation_manifests=[fingerprint(p / 'manifest.json') for p in args.validation],
                  train_windows=len(training), used_train_windows=int(used.sum()),
                  validation_windows=len(validation), used_validation_windows=len(validation_indices),
                  validation_rollout=rollouts(model, validation, args.max_rollouts))
    (args.output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def action_indices(samples, limit):
    if limit is not None and len(samples) > limit:
        return np.linspace(0, len(samples) - 1, num=limit, dtype=int)
    return np.arange(len(samples))


@torch.inference_mode()
def calibrate(args):
    from planner.learning_planner.neupan.inference.path_generator import PathGenerator
    from planner.learning_planner.neupan.inference.reachability import goal_mask

    torch.set_num_threads(args.threads)
    samples = Samples(args.dataset, 'validation', args.pairs)
    planner = PathGenerator(args.checkpoint, args.device)
    model = planner.model
    if model.schema != 2:
        raise ValueError('Calibration requires a schema 2 checkpoint')
    logits, labels = [], []
    indices = action_indices(samples, args.action_windows)
    for windows in window_batches(samples, indices, args.batch_size, args.workers):
        packed = batch(windows, args.device, model.schema)
        if packed is None:
            continue
        x, edges, current, goal, visited, actions = packed
        scores = model.scores(model.encode(x, edges), x[:, :3], edges, current, goal, visited)
        at_goal = goal_mask((x[current, :3] - goal).cpu().numpy(), np.zeros(3))
        informative = scores[:, :STOP].isfinite().any(1).cpu().numpy() & ~at_goal
        logits.extend(scores[:, STOP].cpu().numpy()[informative])
        labels.extend((actions == STOP).cpu().numpy()[informative])
    logits, labels = np.asarray(logits), np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError('Validation needs both continue and optional STOP actions')
    order = np.argsort(-logits)
    ends = np.r_[np.flatnonzero(np.diff(logits[order])), len(logits) - 1]
    tp = np.cumsum(labels[order])[ends]
    f1 = 2. * tp / (ends + 1 + labels.sum())
    index = int(f1.argmax())
    threshold = float(logits[order[ends[index]]])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state['stop_logit_threshold'] = threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(state, args.output)
    report = dict(validation_windows=len(indices), actions=len(labels), positive=int(labels.sum()),
                  stop_logit_threshold=threshold, stop_f1=float(f1[index]),
                  checkpoint_sha256=fingerprint(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def evaluate(args):
    from planner.learning_planner.neupan.inference.path_generator import PathGenerator
    torch.set_num_threads(args.threads)
    samples = Samples(args.dataset, args.split, args.pairs)
    model = PathGenerator(args.checkpoint, args.device).model
    return dict(split=args.split, windows=len(samples),
                schema=model.schema,
                actions=action_epoch(model, samples, action_indices(samples, args.action_windows),
                                     args.batch_size, args.device, workers=args.workers),
                rollout=rollouts(model, samples, args.max_rollouts, guard=not args.unguarded))
