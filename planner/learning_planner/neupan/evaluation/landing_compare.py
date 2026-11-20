# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.evaluation.landing_compare --help
"""Score the original next-edge policy and the landing model on identical windows."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from planner.learning_planner.neupan.data.landing_dataset import hide, region_crop
from planner.learning_planner.neupan.data.landing_gpmp import dump
from planner.learning_planner.neupan.models.landing_model import LandingScorer, select_points
from planner.learning_planner.neupan.training.landing_train import Dataset, predictions, record
from planner.learning_planner.neupan.inference.path_generator import PathGenerator
from planner.learning_planner.neupan.geometry.tomogram import SupportGraph, Tomogram


def compare(dataset, nominal, student, baseline, output):
    torch.set_num_threads(2)
    data = Dataset(dataset)
    raw_inputs = Tomogram.read(nominal)
    new_model = LandingScorer(student)
    old_model = PathGenerator(baseline)
    totals = {'astar_step_policy': Counter(), 'landing_goal_policy': Counter()}
    groups = {key: {} for key in totals}
    for i, item, predicted in predictions(new_model.model, data, data.subset('test')):
        info = data.windows[i]
        start, goal = (np.asarray(info[key], dtype=np.float32) for key in ('start', 'goal'))
        tomo = region_crop(raw_inputs, start)
        if info['scenario'] != 'full':
            direction = goal[:2] - start[:2]
            tomo = hide(tomo, start, direction / np.linalg.norm(direction), info['scenario'])
        old = old_model.local_goals(SupportGraph(tomo), start, goal)
        old_ids = []
        for candidate in old:
            distance = np.linalg.norm(item['xyz'] - candidate.support_xyz, axis=1)
            matched = int(np.argmin(distance))
            if distance[matched] > .01:
                totals['astar_step_policy']['unmapped_output'] += 1
                break
            old_ids.append(matched)
        new_ids = select_points(item['xyz'], item['features'], predicted[:, 1], new_model.threshold,
                                support_area=item.get('observed_support_area'))
        for name, indices in (('astar_step_policy', old_ids), ('landing_goal_policy', new_ids)):
            record(totals[name], indices, item)
            for value in (info['kind'], info['scenario']):
                record(groups[name].setdefault(value, Counter()), indices, item)
    report = dict(overall={name: dict(g) for name, g in totals.items()},
                  groups={name: {k: dict(v) for k, v in split.items()} for name, split in groups.items()})
    dump(output, report)
    print(json.dumps(report['overall']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'nominal', 'student', 'baseline', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    compare(args.dataset, args.nominal, args.student, args.baseline, args.output)
