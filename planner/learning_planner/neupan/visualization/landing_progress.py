"""Render fixed, geographically held-out landing queries during training."""
import json
from pathlib import Path

import numpy as np

from planner.learning_planner.neupan.geometry.landing import MIN_AREA, RADIUS
from planner.learning_planner.neupan.data.landing_dataset import DISK_AREA, HORIZON
from planner.learning_planner.neupan.models.landing_model import candidate_mask, select_points


def fixed_windows(windows, validation):
    """Use paired observations of a descent, plus ascent, detour and flat ground."""
    groups = {}
    for i in map(int, validation):
        w = windows[i]
        groups.setdefault((w['kind'], w['route'], tuple(w['start'])), {})[w['scenario']] = i
    scenarios = ('full', 'approach_gap', 'target_occluded')
    paired = [v for k, v in groups.items() if k[0] == 'down' and all(s in v for s in scenarios)]
    descent = next((v for v in paired if windows[v['full']]['goal'][2] -
                    windows[v['full']]['start'][2] <= -.15), paired[0] if paired else {})
    chosen = [descent[s] for s in scenarios if s in descent]
    for kind in ('down', 'up', 'detour', 'flat'):
        if kind == 'down' and descent:
            continue
        full = [v['full'] for k, v in groups.items() if k[0] == kind and 'full' in v]
        candidate = next(iter(full), None)
        if kind == 'up':
            candidate = next((i for i in full if windows[i]['goal'][2] -
                              windows[i]['start'][2] >= .15), candidate)
        if candidate is not None:
            chosen.append(candidate)
    return chosen or list(map(int, validation[:6]))


def outcome(item, i, min_gain):
    if item['area_gt'][i] < MIN_AREA:
        return 'Insufficient GT area'
    if not item['known'][i]:
        return 'GT unresolved'
    if not item['to_goal'][i]:
        return 'Goal disconnected'
    if not item['incoming'][i]:
        return 'Approach unverified'
    if item['progress'][i] < min_gain:
        return 'No goal progress'
    return 'Verified'


COLORS = {'Verified': '#178459', 'GT unresolved': '#8064a2', 'Approach unverified': '#cb563a',
          'Goal disconnected': '#cb563a', 'No goal progress': '#b0761f', 'Insufficient GT area': '#cb563a'}


def render(directory, name, scored, windows, limit, min_gain, summary):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    fig = Figure(figsize=(14.3, 7.9), layout='constrained', facecolor='white')
    FigureCanvasAgg(fig)
    grid = fig.subplots(2, len(scored) // 2 + len(scored) % 2, squeeze=False)
    records = []
    for ax, (i, item, pred) in zip(grid.flat, scored):
        w = windows[i]
        start, goal = np.array(w['start']), np.array(w['goal'])
        direction = goal[:2] - start[:2]
        distance = np.linalg.norm(direction)
        heading = direction / distance if distance > 1e-6 else np.array([1., 0.])
        lateral = np.array([-heading[1], heading[0]])
        relative = item['xyz'] - start
        forward, across = relative[:, :2] @ heading, relative[:, :2] @ lateral
        observed = item['features'][:, 11] > .5
        area = item.get('observed_support_area')
        measured = item['features'][:, 9] * DISK_AREA if area is None else area
        enough = measured >= MIN_AREA - 1e-7
        approach = item.get('approach_m')
        relative_z = relative[:, 2]
        ax.set_title(f"#{i}  {w['kind']}  /  {w['scenario'].replace('_', ' ')}"
                     f"  ·  goal ΔZ {goal[2] - start[2]:+.2f} m", loc='left', fontsize=10)
        ax.set(xlim=(-HORIZON - .22, HORIZON + .22), ylim=(-HORIZON - .18, HORIZON + .18),
               aspect='equal', xlabel='Toward goal (m)', ylabel='Lateral (m)')
        ax.set_facecolor('#f6f7f9')
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.24, zorder=0)
        same_height = np.abs(relative_z) <= .30
        ax.scatter(forward[observed & ~same_height], across[observed & ~same_height],
                   c='#afa0c0', marker='s', s=9, linewidths=0, alpha=.48, zorder=1)
        on_level = observed & same_height
        background = ax.scatter(forward[on_level], across[on_level], c=item['features'][on_level, 10],
                                cmap='viridis_r', vmin=0, vmax=.8, marker='s', s=11, linewidths=0, zorder=2)
        ax.scatter(forward[observed & ~enough], across[observed & ~enough],
                   c='#838b97', s=7, marker='x', linewidths=.6, zorder=3)
        if approach is not None:
            blocked = observed & enough & ~np.isfinite(approach)
            ax.scatter(forward[blocked], across[blocked], c='#de665b', s=16, marker='x', linewidths=.85, zorder=4)
        ax.scatter(0., 0., c='#272b32', marker='^', s=57, edgecolors='white', linewidths=.8, zorder=6)
        if distance > 1e-6:
            target_x = min(float(distance), HORIZON + .12)
            ax.scatter(target_x, 0., c='#2b507c', marker='*' if distance <= HORIZON else '>',
                       s=88, edgecolors='white', linewidths=.7, zorder=6)
        ids = select_points(item['xyz'], item['features'], pred[:, 1], limit, approach_m=approach, support_area=area)
        rows = []
        for rank, j in enumerate(ids, 1):
            status = outcome(item, j, min_gain)
            color = COLORS[status]
            ax.add_patch(Circle((forward[j], across[j]), RADIUS,
                                fill=False, edgecolor=color, linewidth=1.9, zorder=5))
            ax.text(forward[j], across[j], str(rank), color='white', ha='center', va='center',
                    fontsize=8, weight='bold', zorder=7,
                    bbox=dict(boxstyle='circle,pad=.18', facecolor=color, edgecolor='white', linewidth=.6))
            rows.append(dict(rank=rank, xyz=item['xyz'][j].tolist(), status=status,
                             observed_area_m2=float(measured[j]),
                             gt_area_m2=float(item['area_gt'][j]), delta_z_m=float(relative_z[j]),
                             score=float(pred[j, 1]), gt_progress_m=float(item['progress'][j])))
            if approach is not None:
                rows[-1]['observed_approach_m'] = float(approach[j])
                rows[-1]['predicted_approach_coverage'] = float(pred[j, 2])
        for offset, row in enumerate(rows):
            ax.text(.02, .97 - .07 * offset,
                    f"{row['rank']}  {row['status']}  |  GT {row['gt_area_m2']:.3f} m²"
                    f"  |  ΔZ {row['delta_z_m']:+.2f} m",
                    transform=ax.transAxes, fontsize=7.5, va='top', color=COLORS[row['status']], zorder=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.86, pad=1.2))
        stop = None
        if not ids:
            stop = ('No supported approach' if approach is not None and not candidate_mask(item['features'], approach, area).any()
                    else 'Low goal score')
            ax.text(.02, .97, f'STOP  ·  {stop}', transform=ax.transAxes, va='top', fontsize=9)
        records.append(dict(window=i, **w, candidates=rows, stop_reason=stop))
    for ax in grid.flat[len(scored):]:
        ax.set_visible(False)
    fig.colorbar(background, ax=list(grid.flat), label='Observed cost, same height band', fraction=.021)
    legend = [
        Line2D([], [], marker='^', linestyle='', color='#272b32', label='Start'),
        Line2D([], [], marker='>', linestyle='', color='#2b507c', label='Goal or bearing'),
        Line2D([], [], marker='s', linestyle='', color='#afa0c0', label='Other Z band'),
        Line2D([], [], marker='x', linestyle='', color='#838b97', label='Observed area < 0.04 m²'),
    ]
    if any('approach_m' in item for _, item, _ in scored):
        legend.append(Line2D([], [], marker='x', linestyle='', color='#de665b', label='No supported approach'))
    top = summary.get('good', 0)
    size = summary.get('windows', 0)
    fig.legend(handles=legend, loc='outside lower center', ncol=len(legend), frameon=False, fontsize=9)
    fig.suptitle(f"Landing-goal validation  ·  {name}  ·  top 1 verified {top}/{size}"
                 f"  ·  score cutoff {limit:.3f}", weight='bold', fontsize=15)
    target = directory / f'{name}.png'
    fig.savefig(target, dpi=140)
    (directory / f'{name}.json').write_text(json.dumps(
        dict(score_threshold=limit, validation=summary, windows=records), indent=2, allow_nan=False) + '\n')
    return target
