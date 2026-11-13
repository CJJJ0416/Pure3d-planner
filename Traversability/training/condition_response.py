"""同一 XYZ 的单参数对照：变化区域均衡损失与按参数响应指标。"""
import numpy as np
import torch
from torch.nn import functional as F


def single_parameter_pairs(conditions):
    """只纳入恰好一个学生参数不同的条件对，联合变化不混入单参数监督。"""
    values = np.asarray(conditions)
    pairs = [[] for _ in range(values.shape[1])]
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            changed = np.flatnonzero(values[left] != values[right])
            if len(changed) == 1:
                pairs[changed[0]].append((left, right))
    return pairs


def pair_deltas(prediction, target, pairs):
    left, right = zip(*pairs)
    truth = target[:, right] - target[:, left]
    valid = torch.isfinite(truth)
    truth = torch.where(valid, truth, 0.)
    predicted = prediction[:, right] - prediction[:, left]
    return predicted, truth, valid


def balanced_difference_loss(prediction, target, pairs, beta, threshold, weights):
    """各参数内变化/基本不变区域各占一半，再按参数权重平均。绝对监督仍保留全部点。"""
    terms = []
    for group in pairs:
        predicted, truth, valid = pair_deltas(prediction, target, group)
        errors = F.smooth_l1_loss(predicted, truth, beta=beta, reduction="none")
        active = valid & (truth.abs() > threshold)
        stable = valid & ~active
        counts = torch.stack((active.sum(), stable.sum()))
        means = torch.stack(((errors * active).sum(), (errors * stable).sum())) / counts.clamp_min(1)
        # 一个批次可能没有该参数的变化点，只对实际存在的组求平均。
        terms.append(means.sum() / (counts > 0).sum().clamp_min(1))
    weight = prediction.new_tensor(weights)
    return (torch.stack(terms) * weight).sum() / weight.sum()


def response_totals(prediction, target, pairs, threshold=.02):
    """累加变化点误差/幅度/方向与稳定区域误差，最终在完整划分上汇总。"""
    rows = []
    for group in pairs:
        predicted, truth, valid = pair_deltas(prediction, target, group)
        active = valid & (truth.abs() > threshold)
        stable = valid & ~active
        error = (predicted - truth).abs()
        rows.append(torch.stack((
            (error * active).sum(), (truth.abs() * active).sum(),
            (predicted.abs() * active).sum(), ((predicted * truth > 0) & active).sum(),
            active.sum(), (error * stable).sum(), (predicted.abs() * stable).sum(),
            stable.sum(), (predicted * truth * active).sum(), (truth.square() * active).sum(),
        )).to(torch.float64))
    return torch.stack(rows)


def response_report(totals, names):
    result = {}
    for name, row in zip(names, totals.cpu().tolist()):
        error, teacher, student, correct, count, stable_error, stable_delta, stable_count, dot, square = row
        result[name] = dict(
            changed_delta_mae=error / count, teacher_abs_delta=teacher / count,
            prediction_abs_delta=student / count, direction_correct_pct=100 * correct / count,
            changed_pairs=int(count), stable_pairs=int(stable_count),
            stable_delta_mae=stable_error / stable_count, stable_prediction_abs_delta=stable_delta / stable_count,
            response_gain=dot / square,
        )
    return result


def selection_score(cost_mae, responses):
    """四版使用同一验证分数：绝对误差 + 参数响应误差，坡度/站立权重为2。"""
    weights = dict(clearance_min=1., slope_max=2., step_max=1., stand_ratio=2.)
    response = sum(weights[name] * (row["changed_delta_mae"] + .25 * row["stable_delta_mae"])
                   for name, row in responses.items()) / sum(weights[name] for name in responses)
    return cost_mae + response
