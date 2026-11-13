"""逐楼梯审查已有 GT/预测缓存，不修改教师、权重或训练数据。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

from utils.pointcloud import ROOT, read_json, write_json


COST_MAP = LinearSegmentedColormap.from_list('vst_cost', ['#14c83c', '#ffe633', '#f02d23'])
PROFILE = 'clearance05_slope06_step04_stand005'
TYPE_NAMES = {'straight': 'Straight', 'switchback': 'Switchback', 'spiral': 'Spiral'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def local_coordinates(xyz, stair):
    angle = np.deg2rad(stair['rotation_world_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = xyz.astype(np.float64).copy()
    result[:, :2] = (xyz[:, :2] - stair['center_world_xyz'][:2]) @ rotation
    result[:, 2] -= stair['center_world_xyz'][2]
    return result


def region_mask(local, stair):
    """实际梯段 ROI。仅用于审查，不筛选或改写训练监督。"""
    x, y, z = local.T
    height = (z > .03) & (z < stair['rise_m'] - .1)
    if stair['kind'] == 'spiral':
        radius = np.hypot(x, y)
        return height & (radius > stair['inner_radius_m'] - .05) & (radius < stair['outer_radius_m'] + .1)
    half_width = (stair['width_m'] / 2 if stair['kind'] == 'straight' else stair['width_m'] + .125) + .1
    return height & (abs(x) < half_width) & (abs(y) < stair['unrotated_footprint_m'][1] / 2 - .45)


def percentage(values):
    return float(100 * np.mean(values)) if len(values) else None


def errors(gt, prediction):
    low, high = gt <= .3, gt > .3
    delta = prediction - gt
    return dict(cost_mae_raw=float(abs(delta).mean() * 50),
                cost_p90_abs_raw=float(np.quantile(abs(delta), .9) * 50),
                cost_bias_raw=float(delta.mean() * 50),
                false_block_pct=percentage(prediction[low] > .3),
                false_pass_pct=percentage(prediction[high] <= .3),
                gt_low_points=int(low.sum()), gt_high_points=int(high.sum()))


def stand_response(gt, prediction, conditions):
    """固定其余条件，统计教师代价确有变化的 stand 条件点对。"""
    delta_errors, gt_deltas, pred_deltas, correct = [], [], [], []
    pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            changed = np.flatnonzero(conditions[i] != conditions[j])
            if len(changed) == 1 and changed[0] == 4:
                g, p = gt[:, j] - gt[:, i], prediction[:, j] - prediction[:, i]
                selected = abs(g) > .001
                pairs.append([i, j])
                delta_errors.extend(abs(p[selected] - g[selected]).tolist())
                gt_deltas.extend(abs(g[selected]).tolist())
                pred_deltas.extend(abs(p[selected]).tolist())
                correct.extend((p[selected] * g[selected] > 0).tolist())
    return dict(pairs=pairs, changed_pair_points=len(delta_errors),
                delta_mae=float(np.mean(delta_errors)) if delta_errors else None,
                teacher_abs_delta=float(np.mean(gt_deltas)) if gt_deltas else None,
                prediction_abs_delta=float(np.mean(pred_deltas)) if pred_deltas else None,
                direction_correct_pct=percentage(correct))


def export(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=160, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def short_title(name, result):
    return f'{name}\nMAE {result["cost_mae_raw"]:.2f}/50 | false block {result["false_block_pct"]:.1f}%'


def panels(stair, xyz, values, metrics, stem, three_dimensional=False):
    fig = plt.figure(figsize=(13.2, 7.4))
    axes = []
    for index, (title, cost) in enumerate(zip(['GT', 'Before', 'After'], values)):
        ax = fig.add_subplot(1, 3, index + 1, projection='3d' if three_dimensional else None)
        axes.append(ax)
        if three_dimensional:
            image = ax.scatter(*xyz.T, c=cost, s=4, cmap=COST_MAP, vmin=0, vmax=1, linewidths=0,
                               rasterized=True, depthshade=False)
            ax.set_box_aspect(np.ptp(xyz, axis=0))
            ax.view_init(24, -60)
            ax.set(xlabel='Local X (m)', ylabel='Local Y (m)', zlabel='Height above lower floor (m)')
        else:
            image = ax.scatter(xyz[:, 0], xyz[:, 1], c=cost, s=8, cmap=COST_MAP, vmin=0, vmax=1,
                               linewidths=0, rasterized=True)
            ax.set(aspect='equal', xlabel='Across stair / local X (m)')
            ax.set_xlim(xyz[:, 0].min() - .1, xyz[:, 0].max() + .1)
            ax.set_ylim(xyz[:, 1].min() - .15, xyz[:, 1].max() + .15)
            if stair['kind'] == 'straight':
                ax.set_xticks([-.5, 0, .5])
        ax.set_title(title if index == 0 else short_title(title, metrics[index - 1]), fontsize=12)
    if not three_dimensional:
        axes[0].set_ylabel('Along stair / local Y (m)')
    depth = stair.get('tread_depth_m', stair.get('tread_depth_at_mid_radius_m'))
    fig.suptitle(f'{stair["audit_id"]} | {stair["id"]} | {TYPE_NAMES[stair["kind"]]}\n'
                 f'riser {stair["riser_m"]:.3f} m | tread {depth:.3f} m | width {stair["width_m"]:.3f} m | '
                 f'{stair["steps"]} steps | railings {stair["railings"]}\n'
                 f'p=2 | condition [0.5, 0.6, 0.4, 0.05] | identical XYZ | {len(xyz):,} native query points',
                 fontsize=12, y=.98)
    fig.subplots_adjust(left=.07, right=.88, top=.80, bottom=.12, wspace=.3)
    fig.colorbar(image, cax=fig.add_axes([.92, .25, .018, .43]), label='Cost (0-1)')
    export(fig, stem)


def overview(rows, floor, output):
    selected = [r for r in rows if r['geometry']['from_floor'] == floor]
    fig, axes = plt.subplots(len(selected), 3, figsize=(12.5, 3.9 * len(selected)), squeeze=False)
    for i, row in enumerate(selected):
        stair, xyz = row['geometry'], row['local_xyz']
        depth = stair.get('tread_depth_m', stair.get('tread_depth_at_mid_radius_m'))
        for j, (title, values) in enumerate(zip(['GT', 'Before', 'After'], row['values'])):
            ax = axes[i, j]
            image = ax.scatter(xyz[:, 0], xyz[:, 1], c=values, s=2.7, cmap=COST_MAP,
                               vmin=0, vmax=1, linewidths=0, rasterized=True)
            ax.set_aspect('equal')
            label = (f'{stair["audit_id"]} | {TYPE_NAMES[stair["kind"]]}\n'
                     f'h/d/w={stair["riser_m"]:.3f}/{depth:.3f}/{stair["width_m"]:.3f} m')
            if j:
                label += f'\nMAE {row["metrics"][j-1]["cost_mae_raw"]:.2f}/50'
            ax.set_title(label, fontsize=10)
            ax.set(xlabel='Local X (m)', ylabel='Local Y (m)')
            if stair['kind'] == 'straight':
                ax.set_xticks([-.5, .5])
                ax.tick_params(axis='x', labelsize=9)
    fig.suptitle(f'Floor {floor} to {floor+1} | all four stairs | same XYZ / color scale\n'
                 'p=2 | condition [0.5, 0.6, 0.4, 0.05] | projected XY, no smoothing', fontsize=13, y=.997)
    fig.subplots_adjust(top=.86, bottom=.05, left=.09, right=.87, hspace=.75, wspace=.35)
    for j, title in enumerate(['GT', 'Before', 'After']):
        box = axes[0, j].get_position()
        fig.text((box.x0 + box.x1) / 2, .93, title, ha='center', fontsize=15, weight='bold')
    fig.colorbar(image, cax=fig.add_axes([.92, .24, .018, .48]), label='Cost (0-1)')
    export(fig, output / f'floor_{floor}_all_stairs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/cost_wide_stair_v2')
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / 'all_stairs_audit'
    for folder in [output / 'plots', output / 'points']:
        folder.mkdir(parents=True, exist_ok=True)
    before, after = [np.load(run / (name + '_regression.npz')) for name in ['initial', 'best']]
    scene = read_json(ROOT / 'datasets/wfc_unseen_20261127/scene_20261127/scene.json')
    metadata = read_json(ROOT / 'datasets/wfc_regression_20261127/blocks.json')
    profile_names = after['profiles'].tolist()
    by_name = {p['id']: p['condition'] for p in metadata['scenes'][0]['profiles']}
    conditions = np.asarray([by_name[name] for name in profile_names])
    column = profile_names.index(PROFILE)
    xyz, gt = after['coord'], after['target']
    comparison = dict(scene='scene_20261127', p=2, profile=PROFILE, condition=[.5, .6, .4, .05],
                      cost_scale=50, threshold=.3, before=read_json(run / 'protocol.json')['initial_checkpoint'],
                      after=str(run / 'best.pt'), identical_xyz=bool(np.array_equal(before['coord'], xyz)),
                      identical_targets=bool(np.array_equal(before['target'], gt)), stairs=[])
    geometry_rows, condition_rows, plotting = [], [], []
    for stair in [s for s in scene['stairs'] if s['kind'] != 'ramp']:
        stair = dict(stair, audit_id=f'S{len(geometry_rows)+1}')
        local = local_coordinates(xyz, stair)
        mask = region_mask(local, stair)
        values = [gt[mask, column], before['prediction'][mask, column], after['prediction'][mask, column]]
        metrics = [errors(values[0], prediction) for prediction in values[1:]]
        responses = [stand_response(gt[mask], data['prediction'][mask], conditions) for data in [before, after]]
        geometry = dict(audit_id=stair['audit_id'], id=stair['id'], kind=stair['kind'],
                        from_floor=stair['from_floor'], to_floor=stair['to_floor'],
                        riser_m=stair['riser_m'], tread_depth_m=stair.get('tread_depth_m', stair.get('tread_depth_at_mid_radius_m')),
                        width_m=stair['width_m'], rise_m=stair['rise_m'], steps=stair['steps'], railings=stair['railings'],
                        turn_degrees=stair.get('total_turn_degrees'), inner_radius_m=stair.get('inner_radius_m'),
                        query_points=int(mask.sum()))
        flat = dict(geometry)
        for name, metric in zip(['before', 'after'], metrics):
            flat.update({name + '_' + k: v for k, v in metric.items()})
        for name, response in zip(['before', 'after'], responses):
            flat[name + '_stand_delta_mae'] = response['delta_mae']
            flat[name + '_stand_prediction_abs_delta'] = response['prediction_abs_delta']
        geometry_rows.append(flat)
        comparison['stairs'].append(dict(geometry=stair, region_definition='stair run, 0.03<relative_Z<rise-0.1',
                                        metrics=dict(zip(['before', 'after'], metrics)),
                                        stand_response=dict(zip(['before', 'after'], responses))))
        for j, name in enumerate(profile_names):
            condition_rows.append(dict(stair_id=stair['id'], profile=name,
                                      condition=json.dumps(conditions[j].tolist()),
                                      **{'before_' + k: v for k, v in errors(gt[mask, j], before['prediction'][mask, j]).items()},
                                      **{'after_' + k: v for k, v in errors(gt[mask, j], after['prediction'][mask, j]).items()}))
        np.savez(output / 'points' / (stair['id'] + '.npz'), world_xyz=xyz[mask], local_xyz=local[mask],
                 target=gt[mask], before=before['prediction'][mask], after=after['prediction'][mask],
                 profiles=after['profiles'], conditions=conditions)
        panels(stair, local[mask], values, metrics, output / 'plots' / (stair['id'] + '_GT_before_after'))
        panels(stair, local[mask], values, metrics, output / 'plots' / (stair['id'] + '_3d'), True)
        plotting.append(dict(geometry=stair, local_xyz=local[mask], values=values, metrics=metrics))
        print(stair['audit_id'], stair['id'], 'points', int(mask.sum()), 'MAE',
              metrics[0]['cost_mae_raw'], metrics[1]['cost_mae_raw'], flush=True)
    for name, rows in [('geometry_and_errors.csv', geometry_rows), ('all_condition_errors.csv', condition_rows)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for floor in [0, 1]:
        overview(plotting, floor, output / 'plots')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(xyz[:, 0], xyz[:, 1], c='#d9dee5', s=.3, linewidths=0, rasterized=True)
    for row in plotting:
        stair = row['geometry']; angle = np.deg2rad(stair['rotation_world_deg'])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        w, h = np.asarray(stair['unrotated_footprint_m']) / 2
        corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ rotation.T + stair['center_world_xyz'][:2]
        color = '#1765ad' if stair['from_floor'] == 0 else '#d66a18'
        ax.add_patch(Polygon(corners, fill=False, edgecolor=color, linewidth=1.4))
        ax.text(*stair['center_world_xyz'][:2], stair['audit_id'], color=color, weight='bold', fontsize=14,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2))
    ax.set(aspect='equal', xlabel='Map X (m)', ylabel='Map Y (m)', title='All 8 stair locations | blue F0-F1 / orange F1-F2')
    export(fig, output / 'plots/stair_locations')
    lines = ['# 全楼梯几何与预测误差审查', '',
             '场景 seed20261127。4处直梯、2处折返梯、2处旋转梯。2处斜坡不计入本次楼梯统计。', '',
             'GT为p=2，统一条件为[clearance_min=0.5, slope_max=0.6, step_max=0.4, stand_ratio=0.05]。',
             f'Before来源为{comparison["before"]}，After为{comparison["after"]}。本模块只读取已有预测，不启动训练。', '',
             '图中三列使用相同原生Tomogram查询XYZ，未插值、未平滑。仅旋转显示坐标，不改变cost。', '',
             'ROI保留实际梯段，排除上下整层地板。直梯沿宽度和井口长度裁剪，折返梯覆盖两段梯面，旋转梯采用内外半径环带。',
             '旋转梯多圈在XY投影中可能相互覆盖，应同时查看对应_3d.png。', '',
             'MAE以原始0..50尺度计算。误阻断率=GT<=0.3的点中预测>0.3的比例。误放行率=GT>0.3的点中预测<=0.3的比例。',
             '这些是点级代价误差，不是路线断裂率或机器人执行成功率。', '',
             '|编号|楼梯|高度/深度/宽度(m)|踏步数|栏杆|MAE前→后 /50|误阻断前→后 %|误放行前→后 %|',
             '|---|---|---|---:|---|---:|---:|---:|']
    for r in geometry_rows:
        lines.append(f'|{r["audit_id"]}|{r["id"]}|{r["riser_m"]:.3f}/{r["tread_depth_m"]:.3f}/{r["width_m"]:.3f}|'
                     f'{r["steps"]}|{r["railings"]}|{r["before_cost_mae_raw"]:.2f}→{r["after_cost_mae_raw"]:.2f}|'
                     f'{r["before_false_block_pct"]:.1f}→{r["after_false_block_pct"]:.1f}|'
                     f'{r["before_false_pass_pct"]:.1f}→{r["after_false_pass_pct"]:.1f}|')
    lines.extend(['', '旋转梯深度为中径处的弧长，内外侧踏步深度不同。踏步高度是单阶高度，非整梯总高度。', '',
                  'stand响应固定clearance_min=0.5、slope_max=0.4、step_max=0.3、interval_free=0.5，',
                  '在stand_ratio=[0.05,0.1,0.3,0.5]间两两对照，只统计教师差值绝对值>0.001的点。此项条件与主图不同。',
                  'all_condition_errors.csv包含25组条件下每处楼梯的完整误差。points目录保留对应XYZ、GT和前后预测，可直接复画。', '',
                  '## 按类型统计', '', '|类型|楼梯数|MAE前→后 /50，逐楼梯等权|', '|---|---:|---:|'])
    comparison['by_kind'] = {}
    for kind in ['straight', 'switchback', 'spiral']:
        group = [r for r in geometry_rows if r['kind'] == kind]
        means = {name:float(np.mean([r[name + '_cost_mae_raw'] for r in group])) for name in ['before', 'after']}
        comparison['by_kind'][kind] = dict(stairs=len(group), **means)
        lines.append(f'|{TYPE_NAMES[kind]}|{len(group)}|{means["before"]:.2f}→{means["after"]:.2f}|')
    lines.extend(['', '## stand_ratio 条件响应', '',
                  '下列数值采用0..1代价尺度。幅度比=预测绝对差值均值/教师绝对差值均值，不代表方向正确率。', '',
                  '|编号|有效变化点对数|教师变化幅度|当前预测变化幅度|幅度比|条件差值MAE前→后|当前方向正确率 %|',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for row in comparison['stairs']:
        before_response, after_response = [row['stand_response'][name] for name in ['before', 'after']]
        ratio = after_response['prediction_abs_delta'] / after_response['teacher_abs_delta']
        after_response['amplitude_ratio'] = ratio
        lines.append(f'|{row["geometry"]["audit_id"]}|{after_response["changed_pair_points"]}|'
                     f'{after_response["teacher_abs_delta"]:.3f}|{after_response["prediction_abs_delta"]:.3f}|'
                     f'{ratio:.1%}|{before_response["delta_mae"]:.3f}→{after_response["delta_mae"]:.3f}|'
                     f'{after_response["direction_correct_pct"]:.1f}|')
    lines.extend(['', '## 重点楼梯', ''])
    for r in sorted(geometry_rows, key=lambda x: x['after_cost_mae_raw'], reverse=True):
        lines.append(f'- {r["audit_id"]} {r["id"]}：当前MAE {r["after_cost_mae_raw"]:.2f}/50，'
                     f'误阻断{r["after_false_block_pct"]:.1f}%，误放行{r["after_false_pass_pct"]:.1f}%。')
    improved = [r['audit_id'] for r in geometry_rows if r['after_cost_mae_raw'] < r['before_cost_mae_raw']]
    blocked = [r['audit_id'] for r in geometry_rows if r['after_false_block_pct'] > r['before_false_block_pct']]
    response_worse = [r['geometry']['audit_id'] for r in comparison['stairs']
                      if r['stand_response']['after']['delta_mae'] > r['stand_response']['before']['delta_mae']]
    lines.extend(['', '## 结论与后续增强依据', '',
                  f'1. MAE降低的楼梯：{improved}。误阻断比例上升的楼梯：{blocked}。平均回归误差不能替代低代价通道保留能力。',
                  '2. 按类型误差表与逐楼梯误差排名判断重点，重点检查折返平台、双梯段、旋转半径与转角。',
                  f'3. stand条件差值MAE上升的楼梯：{response_worse}。条件响应幅度及方向应结合上表单独判断。',
                  '4. 几何回归和机器人条件响应应分别评估，不以单个楼梯或单组条件代表全部效果。',
                  '', '## 图像索引', '',
                  '[地图位置](plots/stair_locations.png) · [F0到F1总览](plots/floor_0_all_stairs.png) · [F1到F2总览](plots/floor_1_all_stairs.png)', ''])
    for row in geometry_rows:
        name = row['id']
        lines.append(f'- {row["audit_id"]} {name}：[俯视三列对比](plots/{name}_GT_before_after.png) · [三维对比](plots/{name}_3d.png)')
    write_json(output / 'summary.json', comparison)
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('AUDIT COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
