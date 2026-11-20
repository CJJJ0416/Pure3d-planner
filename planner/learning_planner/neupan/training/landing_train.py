# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.training.landing_train --help
"""Train and evaluate separate landing-area and goal-utility heads."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from planner.learning_planner.neupan.data.expert import fingerprint
from planner.learning_planner.neupan.geometry.landing import MIN_AREA
from planner.learning_planner.neupan.data.landing_dataset import DISK_AREA, FEATURE_DIM, GEOMETRIC_AREA_METRIC, HORIZON
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.models.landing_model import LandingModel, LandingScorer, candidate_mask, select_points

MIN_GAIN = .08


class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.meta = json.loads((directory / 'manifest.json').read_text())
        self.windows = self.meta['windows']
        with np.load(directory / 'windows.npz', allow_pickle=False) as npz:
            self.data = {key: npz[key] for key in npz.files}

    def subset(self, split):
        return np.array([i for i, entry in enumerate(self.windows) if entry['split'] == split])

    def window(self, i):
        bounds = self.data['offsets'][i:i + 2]
        return {key: value[slice(*bounds)] for key, value in self.data.items() if key != 'offsets'}


def packed(dataset, indices):
    items = [dataset.window(i) for i in indices]
    size = max(len(v['xyz']) for v in items)
    x = np.zeros((len(items), size, FEATURE_DIM), np.float32)
    coverage = 'approach_m' in items[0]
    targets = np.zeros((len(items), size, 3 if coverage else 2), np.float32)
    visible, known = np.zeros((len(items), size), bool), np.zeros((len(items), size), bool)
    can_land = np.zeros((len(items), size), bool)
    for index, item in enumerate(items):
        n = len(item['xyz'])
        x[index, :n] = item['features']
        visible[index, :n] = x[index, :n, 11] > .5
        can_land[index, :n] = candidate_mask(item['features'], support_area=item.get('observed_support_area'))
        targets[index, :n, 0] = item['area_gt'] / DISK_AREA
        progress = item['progress']
        targets[index, :n, 1] = np.where(progress >= MIN_GAIN, .5 + .5 * np.clip(progress / HORIZON, 0., 1.), 0.)
        gt_known = item['known']
        if coverage:
            approach = np.isfinite(item['approach_m'])
            targets[index, :n, 1] *= approach
            targets[index, :n, 2] = approach
            gt_known = gt_known | ~approach
        known[index, :n] = gt_known & visible[index, :n]
    return tuple(torch.from_numpy(v) for v in (x, targets, visible, known, can_land))


def epoch(model, dataset, indices, optimizer=None):
    losses = []
    model.train(optimizer is not None)
    for start in range(0, len(indices), 12):
        x, y, visible, known, can_land = packed(dataset, indices[start:start + 12])
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x, visible)
            area_loss = F.smooth_l1_loss(logits[..., 0].sigmoid()[visible], y[..., 0][visible],
                                         reduction='sum') / visible.sum().clamp_min(1)
            rank = F.binary_cross_entropy_with_logits(logits[..., 1], y[..., 1], reduction='none')
            # Most candidates fail the short approach or have no useful gain.
            rank_weight = known * torch.where(y[..., 1] > 0, 3., 1.)
            available = known & can_land
            good = available & (y[..., 1] > 0)
            bad = available & ~good
            rankable = good.any(dim=1) & bad.any(dim=1)
            utility = logits[..., 1][rankable]
            best_positive = utility.masked_fill(~good[rankable], -torch.inf).amax(dim=1)
            best_negative = utility.masked_fill(~bad[rankable], -torch.inf).amax(dim=1)
            # Optimize the actual window-level decision in addition to per-point utility.
            list_loss = F.softplus(.25 + best_negative - best_positive).sum() / rankable.sum().clamp_min(1)
            loss = area_loss + (rank * rank_weight).sum() / rank_weight.sum().clamp_min(1.) + .45 * list_loss
            if model.coverage_head:
                coverage_loss = F.binary_cross_entropy_with_logits(logits[..., 2][visible],
                                                                   y[..., 2][visible], reduction='sum')
                loss = loss + .4 * coverage_loss / visible.sum().clamp_min(1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def predictions(model, dataset, indices):
    model.eval()
    for i in indices:
        item = dataset.window(i)
        x = item['features']
        area = item.get('observed_support_area' if model.geometric_area else 'observed_low_cost_area')
        if area is not None:
            x = x.copy()
            x[:, 9] = area / DISK_AREA
        x = torch.from_numpy(x)
        score = model(x[None], x[None, :, 11] > .5)[0].sigmoid().numpy()
        yield int(i), item, score


def suitable(item):
    approach = item.get('approach_m')
    good = (item['area_gt'] >= MIN_AREA) & item['known'] & (item['progress'] >= MIN_GAIN)
    if approach is not None:
        good &= np.isfinite(approach)
    return good, candidate_mask(item['features'], approach, item.get('observed_support_area'))


def threshold(scored):
    tests = []
    for limit in np.linspace(.05, .8, 31):
        tp = fp = fn = 0
        for _, item, prediction in scored:
            good, allowed = suitable(item)
            selected = select_points(item['xyz'], item['features'], prediction[:, 1], limit, count=1,
                                     approach_m=item.get('approach_m'), support_area=item.get('observed_support_area'))
            tp += int(bool(selected) and good[selected[0]])
            fp += int(bool(selected) and not good[selected[0]])
            fn += int((good & allowed).any() and not (selected and good[selected[0]]))
        tests.append((2. * tp / max(1, 2 * tp + fp + fn), float(limit)))
    return max(tests)[1]


def record(out, idx, item):
    out['windows'] += 1
    good, allowed = suitable(item)
    feasible = (good & allowed).any()
    out['feasible_windows'] += int(feasible)
    if not idx:
        out['empty'] += 1
        out['stop_with_feasible' if feasible else 'stop_without_feasible'] += 1
        return
    i = idx[0]
    out['selected'] += 1
    out['any_good_top3'] += int(good[idx].any())
    out['supported'] += int(item['area_gt'][i] >= MIN_AREA)
    out['good'] += int(good[i])
    out['unsupported'] += int(item['area_gt'][i] < MIN_AREA)
    out['unresolved'] += int(not item['known'][i])
    out['approach_unconfirmed'] += int(item['known'][i] and not item['incoming'][i])
    out['goal_unconnected'] += int(item['known'][i] and not item['to_goal'][i])
    if 'approach_m' in item:
        blocked = ~np.isfinite(item['approach_m'][idx])
        out['observed_blocked_top1'] += int(blocked[0])
        out['observed_blocked_any'] += int(blocked.any())
    # A suitable different height at the same XY identifies a relevant layer confusion.
    points = item['xyz']
    colocated = np.linalg.norm(points[:, :2] - points[i, :2], axis=1) < .051
    other_height = np.abs(points[:, 2] - points[i, 2]) > .3
    out['wrong_layer'] += int(not good[i] and np.any(colocated & other_height & good & allowed))
    out['gain_sum_m'] += float(item['progress'][i])


def metrics(dataset, scored, limit, use_approach_gate=True):
    overall, groups = Counter(), {}
    coverage = Counter()
    area_error_list = []
    for i, item, prediction in scored:
        approach = item.get('approach_m') if use_approach_gate else None
        idx = select_points(item['xyz'], item['features'], prediction[:, 1], limit, approach_m=approach,
                            support_area=item.get('observed_support_area'))
        record(overall, idx, item)
        window = dataset.windows[i]
        for key in (window['scenario'], window['kind']):
            record(groups.setdefault(key, Counter()), idx, item)
        observed = item['features'][:, 11] > .5
        area_error_list.extend(abs(prediction[observed, 0] * DISK_AREA - item['area_gt'][observed]))
        if prediction.shape[1] > 2:
            supported = np.isfinite(item['approach_m'][observed])
            predicted = prediction[observed, 2] >= .5
            coverage['tp'] += int((supported & predicted).sum())
            coverage['fp'] += int((~supported & predicted).sum())
            coverage['fn'] += int((supported & ~predicted).sum())
    return dict(overall=dict(overall), by_group={key: dict(value) for key, value in groups.items()},
                landing_area_mae_m2=float(np.mean(area_error_list)), coverage_head=dict(coverage))


def train(directory, output, epochs, seed, preview_every=3, preview_threshold=.475, baseline=None):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = Dataset(directory)
    tr, va, te = (data.subset(split) for split in ('train', 'validation', 'test'))
    if not len(tr) or not len(va) or not len(te):
        raise ValueError('Each geographic fold must have samples')
    area_metric = data.meta.get('area_metric', 'connected low-cost footprint')
    model = LandingModel(coverage_head='approach_m' in data.data, geometric_area=area_metric == GEOMETRIC_AREA_METRIC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    output.mkdir(parents=True, exist_ok=False)
    preview_dir = output / 'preview'
    if preview_every:
        from planner.learning_planner.neupan.visualization.landing_progress import fixed_windows, render
        plot_indices = fixed_windows(data.windows, va)
    history = []
    best = float('inf')
    for i in range(epochs):
        state = dict(epoch=i + 1, train=epoch(model, data, rng.permutation(tr), optimizer),
                     validation=epoch(model, data, va))
        history.append(state)
        if state['validation'] < best:
            best = state['validation']
            kind = 'landing_goal_v3' if model.coverage_head else 'landing_goal_v2'
            torch.save(dict(type=kind, model=model.state_dict(), epoch=i + 1, area_metric=area_metric),
                       output / 'best.pt')
        if preview_every and (i == 0 or (i + 1) % preview_every == 0 or i + 1 == epochs):
            summary = metrics(data, predictions(model, data, va), preview_threshold)['overall']
            preview = render(preview_dir, f'epoch_{i + 1:03d}', list(predictions(model, data, plot_indices)),
                             data.windows, preview_threshold, MIN_GAIN, summary)
            latest = preview_dir / '.latest.png'
            shutil.copyfile(preview, latest)
            latest.replace(preview_dir / 'latest.png')
            state.update(preview=str(preview), preview_validation=summary)
        print(json.dumps(state), flush=True)
    best_state = torch.load(output / 'best.pt', weights_only=True)
    model.load_state_dict(best_state['model'])
    validation = list(predictions(model, data, va))
    limit = threshold(validation)
    torch.save(dict(best_state, threshold=limit), output / 'best.pt')
    val_metrics = metrics(data, validation, limit)
    if preview_every:
        render(preview_dir, f"best_{best_state['epoch']:03d}", list(predictions(model, data, plot_indices)),
               data.windows, limit, MIN_GAIN, val_metrics['overall'])
    report = dict(history=history, best_epoch=best_state['epoch'], threshold=limit, seed=seed,
                  checkpoint_sha256=fingerprint(output / 'best.pt'), validation=val_metrics,
                  test=metrics(data, predictions(model, data, te), limit),
                  fold_counts=data.meta['fold_counts'],
                  area_metric=area_metric)
    if baseline:
        scorer = LandingScorer(baseline)
        report['baseline_test'] = metrics(data, predictions(scorer.model, data, te), scorer.threshold,
                                          use_approach_gate=scorer.model.coverage_head)
        report['baseline_sha256'] = fingerprint(baseline)
    if model.coverage_head:
        report['approach_supervision'] = data.meta['approach_supervision']
    dump(output / 'report.json', report)
    return dict(best_epoch=report['best_epoch'], test=report['test']['overall'], threshold=limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=821111)
    parser.add_argument('--preview-every', type=int, default=3, help='Render after this many epochs. Zero disables plots.')
    parser.add_argument('--preview-threshold', type=float, default=.475, help='Fixed score cutoff in progress plots.')
    parser.add_argument('--baseline', type=Path, help='Evaluate a prior landing checkpoint on the same test windows.')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.preview_every < 0 or not 0 <= args.preview_threshold <= 1:
        parser.error('Preview interval must be nonnegative and the score cutoff in [0, 1]')
    print(json.dumps(train(args.dataset, args.output, args.epochs, args.seed,
                           args.preview_every, args.preview_threshold, args.baseline)), flush=True)


if __name__ == '__main__':
    main()
