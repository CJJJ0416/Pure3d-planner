# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.training.sequence_train --help
"""Train area-supervised support sequences and export fixed validation previews."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from planner.learning_planner.neupan.data.expert import fingerprint
from planner.learning_planner.neupan.geometry.landing import read_raw_map
from planner.learning_planner.neupan.data.landing_dataset import HORIZON
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.inference.landing_route import MIN_COST_PROGRESS
from planner.learning_planner.neupan.inference.reachability import goal_mask
from planner.learning_planner.neupan.data.sequence_bank import masked
from planner.learning_planner.neupan.data.sequence_expert import DISK_AREA, SCENARIOS, Teacher, observation
from planner.learning_planner.neupan.models.support_sequence import (CHECKPOINT_TYPE, COST_UNIT, MAX_LENGTH, MAX_POINTS, SequenceGraph,
                               SequenceNetwork, SequencePlanner, tensors)
from planner.learning_planner.neupan.geometry.tomogram import Tomogram


class Corpus:
    def __init__(self, expert, raw):
        self.root = expert
        m = json.loads((expert / 'manifest.json').read_text())
        if m.get('area_gate') or m['raw_sha256'] != fingerprint(raw):
            raise ValueError('Expected area-independent expert routes and their original GT map')
        bank = json.loads((Path(m['dataset']) / 'manifest.json').read_text())
        if m['nominal_sha256'] != bank['sha256'] or fingerprint(Path(bank['tomogram'])) != bank['sha256']:
            raise ValueError('Expert, route bank, and nominal map fingerprints differ')
        nominal, gt = Tomogram.read(bank['tomogram']), read_raw_map(raw)
        with np.load(raw, allow_pickle=False) as arr:
            ceiling = arr['ceiling']
        self.nominal, self.gt = {}, {}
        with np.load(bank['ownership'], allow_pickle=False) as arr:
            folds = arr['folds']
        for index, label in enumerate(bank['splits']):
            owned = folds == index
            self.nominal[label] = masked(nominal, owned)
            clear = ~np.isfinite(ceiling) | (ceiling - gt.ground >= .5)
            self.gt[label] = SequenceGraph(masked(gt, owned & clear), np.array([1e6] * 3))
        options = defaultdict(list)
        self.rows = {}
        for row in m['routes']:
            refs = [v['support_xyz'] for v in row.get('checkpoints', []) if v.get('reason') == 'supported']
            if row['result'] != 'valid' or len(refs) < 2:
                continue
            # One query in the first half of each trajectory. Record the exact index.
            origins = np.asarray(refs[:max(1, len(refs) // 2)])
            goal = np.asarray(row['goal'], dtype=np.float32)
            candidates = np.flatnonzero(~goal_mask(origins, goal))
            if len(candidates):
                self.rows[row['id']] = (row, origins[candidates[len(candidates) // 2]], goal)
                options[row['split'], row['kind']].append(row['id'])
        self.options = options

    def select(self, split, count, seed):
        rng = np.random.default_rng(seed)
        groups = [list(rng.permutation(v)) for (s, _), v in sorted(self.options.items()) if s == split]
        routes = []
        while groups and (not count or len(routes) < count):
            for group in groups:
                if group and (not count or len(routes) < count):
                    routes.append(int(group.pop()))
            groups = [group for group in groups if group]
        return [(i, mask) for i in routes for mask in SCENARIOS]

    @lru_cache(maxsize=256)
    def episode(self, query):
        i, mask = query
        row, start, goal = self.rows[i]
        local = observation(self.nominal[row['split']], start, goal, mask)
        graph = SequenceGraph(local, goal)
        source = graph.source(start)
        teacher = Teacher(self.gt[row['split']], graph, goal) if source >= 0 else None
        return row, start, goal, local, graph, teacher, source

    @lru_cache(maxsize=256)
    def expert_states(self, query):
        _, _, _, _, graph, teacher, source = self.episode(query)
        if teacher is None:
            return None
        budget = MAX_LENGTH
        past, states = [], []
        for _ in range(MAX_POINTS):
            if graph.goals[source]:
                break
            past.append(graph.xyz[source])
            decision = teacher.label(source, budget, past)
            states.append((source, budget, decision))
            if not decision.positive.any():
                break
            i = int(np.argmax(decision.utility))
            source = int(decision.actions.ids[i])
            budget -= decision.actions.lengths[i]
        return graph.inputs(), states


def binary_threshold(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        return .5
    tests = []
    for threshold in np.linspace(.1, .9, 17):
        predicted = scores >= threshold
        quality = .5 * (predicted[labels].mean() + (~predicted[~labels]).mean())
        tests.append((quality, -abs(threshold - .5), threshold))
    return float(max(tests)[-1])


def reflected(data):
    inputs = dict(data)
    inputs['geometry'] = data['geometry'].reshape(-1, 2, 5, 5)[..., ::-1].reshape(-1, 50).copy()
    for key in ('node', 'edge'):
        inputs[key] = data[key].copy()
        inputs[key][:, 1] *= -1
    return inputs


def evaluate_steps(model, corpus, query, device, area_loss, student=False, thresholds=None, mirror=False):
    _, _, _, _, graph, teacher, source = corpus.episode(query)
    if teacher is None:
        return None
    if student:
        data, states = graph.inputs(), None
    else:
        data, states = corpus.expert_states(query)
    encoded, area, remaining, conn = model.encode(**tensors(reflected(data) if mirror else data, device))
    connect_target = np.isfinite(teacher.remaining)
    known = teacher.known
    loss = (F.binary_cross_entropy_with_logits(conn[known], torch.as_tensor(connect_target[known], device=device).float())
            if known.any() else encoded.sum() * 0)
    if connect_target.any():
        target = torch.as_tensor(teacher.remaining[connect_target] / COST_UNIT, dtype=remaining.dtype, device=device)
        loss = loss + F.smooth_l1_loss(remaining[connect_target], target)
    if area_loss:
        target_area = torch.as_tensor(teacher.coverage / DISK_AREA, device=device)
        loss = loss + .25 * F.smooth_l1_loss(area.sigmoid(), target_area)
    if thresholds is not None:
        thresholds['connections'].extend(zip(conn[known].sigmoid().cpu().tolist(), connect_target[known].tolist()))
    history = encoded.new_zeros(64)
    state_losses, past = [], []
    budget = MAX_LENGTH
    for step in range(MAX_POINTS):
        if student:
            if graph.goals[source]:
                break
            past.append(graph.xyz[source])
            label = teacher.label(source, budget, past)
        else:
            if step >= len(states):
                break
            source, budget, label = states[step]
        a = label.actions
        feature = np.column_stack((a.lengths / HORIZON, a.costs / COST_UNIT)).astype(np.float32)
        history, scores, stop = model.step(encoded, source, history, a.ids, torch.as_tensor(feature, device=device), budget)
        wants_stop = not label.positive.any()
        step_loss = F.binary_cross_entropy_with_logits(stop, stop.new_tensor(float(wants_stop)))
        if len(a.ids):
            logits = torch.cat((scores, stop[None]))
            distribution = np.zeros(len(a.ids) + 1, dtype=np.float32)
            if wants_stop:
                distribution[-1] = 1.
            else:
                values = label.utility[label.positive]
                probability = np.exp((values - values.max()) / .35)
                distribution[np.flatnonzero(label.positive)] = probability / probability.sum()
            step_loss = step_loss - (torch.as_tensor(distribution, device=device) * logits.log_softmax(0)).sum()
        if area_loss and len(label.pair_high):
            w = torch.as_tensor(label.pair_weight, device=device)
            pair_loss = F.softplus(scores[label.pair_low] - scores[label.pair_high])
            step_loss = step_loss + .15 * (w * pair_loss).sum() / w.sum().clamp_min(1e-7)
        state_losses.append(step_loss)
        if thresholds is not None:
            thresholds['stops'].append((stop.sigmoid().item(), wants_stop))
        if student:
            if not len(a.ids):
                break
            mask = np.ones(len(a.ids), dtype=bool)
            if not np.isfinite(graph.remaining[source]):
                confidence = conn[a.ids].detach().sigmoid().cpu().numpy()
                progress = (remaining[source] - remaining[a.ids]).detach().cpu().numpy() * COST_UNIT
                mask &= (confidence >= .5) & (progress >= MIN_COST_PROGRESS)
                if stop.detach().sigmoid().item() >= .5:
                    break
            if not mask.any():
                break
            idx = int(np.argmax(np.where(mask, scores.detach().cpu().numpy(), -np.inf)))
            source = int(a.ids[idx])
            budget -= a.lengths[idx]
    return loss + torch.stack(state_losses).mean() if state_losses else loss


def step_series(result):
    return [dict(xyz=p.xyz.tolist(), approach=p.approach_xyz.tolist(), length_m=p.approach_m, cost=p.trav_cost,
                 approach_cost=p.approach_cost,
                 area_m2=p.coverage_m2, inferred_area_m2=p.estimated_coverage_m2) for p in result.points]


def expert_series(graph, teacher, source, goal):
    if teacher is None:
        return [], 'no_start_support'
    decisions, reason = teacher.prefix(source, goal)
    return [dict(xyz=graph.xyz[d.actions.ids[i]].tolist(), approach=graph.xyz[d.actions.chain(i)].tolist(),
                 length_m=float(d.actions.lengths[i]), area_m2=float(d.area[i]),
                 cost=float(graph.cost[d.actions.ids[i]]), gt_approach_cost=float(d.costs[i]),
                 gt_remaining_cost=float(teacher.remaining[d.actions.ids[i]]),
                 gt_progress=float(d.progress[i])) for d, i in decisions], reason


def preview(corpus, queries, checkpoint, out, device, title='Fixed validation windows'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    planner = SequencePlanner(checkpoint, device)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    records = []
    for q, ax in zip(queries, axs.flat):
        r, start, goal, tomo, graph, teacher, source = corpus.episode(q)
        result = planner.plan(tomo, start, goal)
        expert, end = expert_series(graph, teacher, source, goal)
        student = step_series(result)
        records.append(dict(route=q[0], kind=r['kind'], variant=q[1], expert=expert, expert_stop=end,
                            student=student, student_stop=result.reason))
        if len(graph.xyz):
            level = np.abs(graph.xyz[:, 2] - start[2]) < .8
            ax.scatter(*graph.xyz[level, :2].T, c=graph.cost[level], cmap='Greys', s=3, alpha=.40)
        for points, color in ((expert, '#b7662d'), (student, '#1183a2')):
            for stage in points:
                route = np.array(stage['approach'])
                ax.plot(*route[:, :2].T, c=color, lw=2)
                x, y, _ = stage['xyz']
                ax.scatter(x, y, c=color, s=16)
                ax.annotate(f"{stage['area_m2']:.3f}/{stage['cost']:.2f}", (x, y), color=color, fontsize=7)
        ax.scatter(*start[:2], marker='^', c='black', s=22)
        ax.set_title(f"{r['kind']} · {q[1]}\n{result.reason}", fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
    for ax in axs.flat[len(records):]:
        ax.set_visible(False)
    fig.suptitle(f'{title} · VST expert (orange) / sequence (blue)\nFootprint m² / point cost')
    fig.savefig(out.with_suffix('.png'), dpi=135)
    plt.close(fig)
    dump(out.with_suffix('.json'), records)


def train(expert, raw, output, epochs, train_limit, val_limit, seed, area_loss, device):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    corpus = Corpus(expert, raw)
    tr, va = (corpus.select(split, limit, seed) for split, limit in (('train', train_limit), ('validation', val_limit)))
    if not tr or not va:
        raise ValueError('Training and validation need independent valid expert routes')
    output.mkdir(parents=True, exist_ok=False)
    model = SequenceNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    fixed = va[:4] + [q for q in va[4:] if q[1] == 'full'][:2]
    history = []
    best = np.inf
    for epoch in range(1, epochs + 1):
        start_time = perf_counter()
        model.train()
        train_losses = []
        student_count = 0
        for index, i in enumerate(rng.permutation(len(tr))):
            # Train on model-selected states in the second half, while retaining expert states.
            student = epoch > epochs // 2 and index % 3 == 0
            loss = evaluate_steps(model, corpus, tr[i], device, area_loss, student=student,
                                  mirror=bool(rng.integers(2)))
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            student_count += int(student)
        model.eval()
        with torch.no_grad():
            calibration = defaultdict(list)
            vals = [evaluate_steps(model, corpus, q, device, area_loss, thresholds=calibration) for q in va]
            score = float(np.mean([float(x) for x in vals if x is not None]))
        def calibrated(key):
            values = calibration[key]
            return binary_threshold(*zip(*values)) if values else .5

        state = dict(type=CHECKPOINT_TYPE, model=model.state_dict(), epoch=epoch, area_loss=area_loss,
                     stop_threshold=calibrated('stops'), connect_threshold=calibrated('connections'))
        if epoch > epochs // 2 and score < best:
            torch.save(state, output / 'best.pt')
            best = score
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            file = output / f'epoch_{epoch:03d}.pt'
            torch.save(state, file)
            preview(corpus, fixed, file, output / f'epoch_{epoch:03d}', device)
        row = dict(epoch=epoch, train=float(np.mean(train_losses)), validation=score,
                   student_windows=student_count, seconds=perf_counter() - start_time)
        history.append(row)
        print(json.dumps(row), flush=True)
        dump(output / 'training.json', dict(history=history, seed=seed, areas=area_loss,
                                            train_queries=tr, validation_queries=va))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('expert', 'raw-map', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--train-limit', type=int, default=0, help='Route cap, zero uses all')
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=821111)
    parser.add_argument('--no-area-loss', action='store_true')
    parser.add_argument('--device', default='cpu')
    a = parser.parse_args()
    train(a.expert, a.raw_map, a.output, a.epochs, a.train_limit, a.val_limit, a.seed, not a.no_area_loss, a.device)


if __name__ == '__main__':
    main()
