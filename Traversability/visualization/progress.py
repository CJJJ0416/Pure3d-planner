"""把运行日志转换为自动刷新的文字进度JSON，不读取权重或训练点云。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
STAGE_NAMES = {
    'data_preparation': '场景与教师监督生成',
    'prepare_connectivity': '验证集连通图准备',
    'prepare_retention_connectivity': '旧验证集连通图准备',
    'initial_val': '微调前固定验证',
    'initial_retention_val': '微调前旧建筑验证',
    'train': '增强训练',
    'final_val': '最佳权重固定验证',
    'initial_test': '微调前独立测试',
    'initial_regression': '微调前大地图回归',
    'best_test': '微调后独立测试',
    'best_regression': '微调后大地图回归',
    'retention_val': '微调后旧建筑验证',
    'predict_new_test': '直梯完整PCD推理',
    'predict_new_test_switchback': '折返梯完整PCD推理',
    'predict_new_test_spiral': '旋转梯完整PCD推理',
    'predict_large_terrain': '大地图完整PCD推理',
    'report': '全部楼梯对比图与报告',
}


def lines(path):
    """只解析已写完换行的日志记录。"""
    return path.read_text().split('\n')[:-1] if path.exists() else []


def bar(done, total, unit):
    filled = int(20 * done / total)
    return f'[{"█" * filled}{"░" * (20-filled)}] {100*done/total:.1f}% ({done}/{total} {unit})'


def validation_values(metrics):
    stairs = metrics['stair_metrics']
    graph = stairs['radii']['0.45']
    return {
        'Cost MAE /50': metrics['cost_mae_raw'],
        '楼梯断裂率 %': graph['break_pct'],
        '错误放行率 %': graph['false_pass_pct'],
        'stand条件差值MAE': metrics['parameter_response']['stand_ratio']['changed_delta_mae'],
        '非楼梯Cost MAE /50': stairs['non_stair_mae'] * 50,
        '危险代价低估率 %': metrics['dangerous_underprediction_pct'],
    }


def difference_text(baseline, value, percent=False):
    if baseline is None or value is None:
        return '无有效案例'
    change = value - baseline
    unit = '个百分点' if percent else ''
    direction = '下降' if change < 0 else '上升' if change > 0 else '持平'
    return f'{baseline:.4f} → {value:.4f}（{direction}{abs(change):.4f}{unit}）'


def validation_progress(run, done_stages, history):
    if 'initial_val' not in done_stages:
        return {'状态': '等待微调前基线评估'}
    baseline_metrics = json.loads((run / 'initial_val.json').read_text())
    baseline = validation_values(baseline_metrics)
    evaluated = [row for row in history if row['validation'] is not None]
    latest_epoch, latest_metrics = (evaluated[-1]['epoch'], evaluated[-1]['validation']) if evaluated else (0, baseline_metrics)
    latest = validation_values(latest_metrics)
    trend = [dict(epoch=0, **baseline)] + [
        dict(epoch=row['epoch'], **validation_values(row['validation'])) for row in evaluated]
    best = min(evaluated, key=lambda row: tuple(row['validation']['selection_key'])) if evaluated else None
    graph = latest_metrics['stair_metrics']['radii']['0.45']
    return {
        '最新结果epoch': latest_epoch,
        '当前最佳epoch': best['epoch'] if best is not None else None,
        '指标方向': '以下误差与比例均越低越好',
        '基线→最新': {name: difference_text(baseline[name], latest[name], name.endswith('%')) for name in baseline},
        '最新数值': latest,
        '最佳数值': validation_values(best['validation']) if best is not None else None,
        'Cost MAE趋势': ' → '.join(f'{row["Cost MAE /50"]:.4f}' for row in trend),
        '趋势对应epoch': [row['epoch'] for row in trend],
        '完整验证历史': trend,
        '连通诊断样本': {key: graph[key] for key in ['gt_connected', 'gt_blocked', 'geometry_disconnected']},
        '说明': '验证每5轮更新。断裂/放行为0.45m邻接图、cost阈值0.3的诊断。stand差值采用归一化cost尺度。',
    }


def post_training_results(run, done_stages):
    result = {}
    for stage, title in [('initial_test', '独立测试_增强前'), ('best_test', '独立测试_增强后'),
                         ('initial_retention_val', '旧建筑验证_增强前'), ('retention_val', '旧建筑验证_增强后')]:
        if stage in done_stages:
            result[title] = validation_values(json.loads((run / (stage + '.json')).read_text()))
    if 'report' in done_stages:
        result['已知目标楼梯连通性'] = json.loads((run / 'stair_regression.json').read_text())
        audit = run / 'all_stairs_audit/summary.json'
        if audit.exists():
            rows = json.loads(audit.read_text())['stairs']
            result['全部8处楼梯回归'] = {
                row['geometry']['id']: {
                    'Cost MAE /50': difference_text(row['metrics']['before']['cost_mae_raw'], row['metrics']['after']['cost_mae_raw']),
                    '误阻断率 %': difference_text(row['metrics']['before']['false_block_pct'], row['metrics']['after']['false_block_pct'], True),
                    '误放行率 %': difference_text(row['metrics']['before']['false_pass_pct'], row['metrics']['after']['false_pass_pct'], True),
                } for row in rows}
        result['完整报告'] = str(run / 'REPORT.md')
    return result


def snapshot(run):
    settings = json.loads((run / 'protocol.json').read_text())['settings']
    training = settings['training']
    epochs, batch = training['epochs'], training['batch_size']
    steps_per_epoch = (training['blocks_per_epoch'] + batch - 1) // batch
    total_steps = epochs * steps_per_epoch
    pipeline = lines(run / 'pipeline.log')
    done_stages = {line.split()[1] for line in pipeline if line.startswith('DONE ')}
    started = [line.split()[1] for line in pipeline if line.startswith('START ')]
    phase = started[-1] if started else 'data_preparation'
    complete = any(line.startswith('COMPLETE ') for line in pipeline)
    pipeline_pid = int((run / 'pipeline.pid').read_text())
    running = (Path('/proc') / str(pipeline_pid)).exists()

    scenes = sum(settings['splits'].values())
    teacher_per_train = 4 + 36 * settings['training_banks']
    teacher_per_eval = 47
    teacher_total = settings['splits']['train'] * teacher_per_train + (
        settings['splits']['val'] + settings['splits']['test']) * teacher_per_eval
    ready = {}
    partial = {}
    for line in lines(run / 'data_preparation.log'):
        match = re.match(r'READY (\d+)/(\d+) (scene_\d+) (train|val|test) ', line)
        if match:
            ready[match[3]] = teacher_per_train if match[4] == 'train' else teacher_per_eval
        match = re.match(r'teacher (scene_\d+)/conditioned: (\d+)/(\d+)', line)
        if match:
            partial[match[1]] = int(match[2])
    teacher_done = sum(ready.values()) + sum(count for scene, count in partial.items() if scene not in ready)

    completed_epochs, latest_step, logged_epoch = 0, 0, 0
    batch_loss, epoch_loss = None, None
    for line in lines(run / 'train.log'):
        match = re.match(r'epoch=(\d+) step=(\d+) loss=([\d.eE+-]+)', line)
        if match:
            logged_epoch, latest_step, batch_loss = int(match[1]), int(match[2]), float(match[3])
        match = re.match(r'epoch=(\d+) loss=([\d.eE+-]+)', line)
        if match:
            completed_epochs, epoch_loss = int(match[1]), float(match[2])
            latest_step = completed_epochs * steps_per_epoch
    train_done = 'train' in done_stages
    current_epoch = (min(epochs, max(logged_epoch, completed_epochs + 1))
                     if phase == 'train' or completed_epochs else 0)
    epoch_step = latest_step - (current_epoch - 1) * steps_per_epoch if current_epoch else 0
    if train_done:
        current_epoch, epoch_step = epochs, steps_per_epoch

    post_stages = ['final_val', 'initial_test', 'initial_regression', 'best_test',
                   'best_regression', 'retention_val', 'predict_new_test',
                   'predict_large_terrain', 'report']
    if settings.get('template_scene'):
        post_stages += ['predict_new_test_switchback', 'predict_new_test_spiral']
    post_done = [name for name in post_stages if name in done_stages]
    baseline_done = sum(name in done_stages for name in ['initial_val', 'initial_retention_val'])
    status = '已完成' if complete else '运行中' if running else '已停止，流程未完成'
    current_stage = STAGE_NAMES[phase]
    if phase == 'train' and epoch_step == steps_per_epoch and completed_epochs < current_epoch:
        current_stage += '，本轮收尾（验证或保存）'
    history_file = run / 'history.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    result = {
        '更新时间': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        '状态': status,
        '当前阶段': '全部完成' if complete else current_stage,
        '进度': {
            '场景与监督': bar(len(ready), scenes, '场景'),
            '教师条件标签': bar(teacher_done, teacher_total, '组'),
            '基线评估': bar(baseline_done, 2, '任务'),
            '完整训练轮次': bar(completed_epochs, epochs, '轮'),
            '参数更新': bar(latest_step, total_steps, '步'),
            '当前训练轮': bar(epoch_step, steps_per_epoch, f'步，第{current_epoch}轮') if current_epoch else '等待训练启动',
            '训练后测试与绘图': bar(len(post_done), len(post_stages), '任务'),
        },
        '训练': {
            '当前epoch': current_epoch,
            '已完成epoch': completed_epochs,
            '日志已记录更新步数': latest_step,
            '目标更新步数': total_steps,
            '真实batch': batch,
            '每轮训练块数': training['blocks_per_epoch'],
            '最新batch_loss': batch_loss,
            '最近完整轮loss': epoch_loss,
            '验证间隔epoch': training['evaluation_every_epochs'],
            '完整轮loss历史': [dict(epoch=row['epoch'], loss=row['loss']) for row in history],
        },
        '固定验证指标': validation_progress(run, done_stages, history),
        '测试与回归结果': post_training_results(run, done_stages),
        '数据划分': settings['splits'],
        'cost尺度': '0..1',
        '教师p': 2,
        '最近断点': str(run / 'last.pt') if (run / 'last.pt').exists() else '尚未保存',
        '最佳权重': str(run / 'best.pt') if (run / 'best.pt').exists() else '首次验证后生成',
        '说明': '参数更新按训练日志计数，批次日志每50步记录一次。每5秒刷新，流程结束后停止刷新。',
    }
    target = run / 'progress.json'
    temporary = run / '.progress.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        progress = snapshot(args.run.resolve())
        if not args.watch or progress['状态'] != '运行中':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
