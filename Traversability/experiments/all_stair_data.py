"""由全8处楼梯的几何模板生成独立seed，标签仍使用原生VST、p=2与0..1尺度。"""
import numpy as np

from Traversability.experiments.wide_stair_data import main as generate
from utils.pointcloud import read_json, workspace_path


def specifications(config):
    scene = read_json(workspace_path(config['template_scene']))
    stairs = [r for r in scene['stairs'] if r['kind'] != 'ramp']
    result = []
    seed = config['seed']
    names = ['riser_m', 'tread_depth_m', 'width_m', 'inner_radius_m', 'turns', 'rise_m']
    for split, count in config['per_template'].items():
        for stair in stairs:
            anchor = dict(riser_m=stair['riser_m'],
                          tread_depth_m=stair.get('tread_depth_m', stair.get('tread_depth_at_mid_radius_m')),
                          width_m=stair['width_m'], inner_radius_m=stair.get('inner_radius_m', .45),
                          turns=stair.get('total_turn_degrees', 360) / 360, rise_m=stair['rise_m'])
            rng = np.random.default_rng(seed)
            focused = int(count * config['focused_fraction'])
            for distribution, number in [('focused', focused), ('broad', count - focused)]:
                unit = np.column_stack([(rng.permutation(number) + rng.random(number)) / number for _ in names])
                for j in range(number):
                    row = dict(seed=seed, split=split, kind=stair['kind'], template_id=stair['id'],
                               distribution=distribution, railings=bool(rng.integers(0, 2)),
                               mirror=bool(rng.integers(0, 2)), three_floors=bool(rng.integers(0, 2)),
                               wall=bool(rng.integers(0, 2)), side_gap_m=float(rng.uniform(*config['side_gap_m'])))
                    for k, name in enumerate(names):
                        bounds = np.asarray(config['broad_geometry'][name])
                        if distribution == 'focused':
                            bounds = np.clip(anchor[name] * np.asarray(config['focused_multipliers'][name]), *bounds)
                        row[name] = float(bounds[0] + unit[j, k] * (bounds[1] - bounds[0]))
                    # 每个模板的一个尺寸锚点，姿态、镜像、平台与楼层背景仍由独立seed生成。
                    if split == 'train' and distribution == 'focused' and j == 0:
                        row.update(anchor, distribution='dimension_anchor', railings=stair['railings'])
                    result.append(row)
                    seed += 1
    return result


if __name__ == '__main__':
    generate(specifications)
