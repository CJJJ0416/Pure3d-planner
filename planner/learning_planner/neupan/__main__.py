# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan --help
"""Generate, train, evaluate and visualize the isolated local path component."""
import argparse
import json
import math
from pathlib import Path

from planner.learning_planner.neupan.data.expert import DEFAULT_MAP, generate


def split_options(cmd, default):
    cmd.add_argument('--split', choices=('train', 'validation', 'test'), default=default)


def inference_options(cmd):
    cmd.add_argument('--checkpoint', type=Path, required=True)
    cmd.add_argument('--device', default='cpu')
    cmd.add_argument('--threads', type=int, default=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    labels = commands.add_parser('generate')
    labels.add_argument('--tomogram', type=Path, default=DEFAULT_MAP)
    labels.add_argument('--output', type=Path, required=True)
    labels.add_argument('--count', type=int, default=10_000)
    labels.add_argument('--attempts', type=int)
    labels.add_argument('--seed', type=int, default=821111)
    labels.add_argument('--tile-size', type=float, default=12.)
    group = labels.add_mutually_exclusive_group()
    group.add_argument('--scene-split', choices=('train', 'validation', 'test'))
    group.add_argument('--regions', type=Path)

    train = commands.add_parser('train')
    train.add_argument('--train', type=Path, nargs='+', required=True)
    train.add_argument('--validation', type=Path, nargs='+', required=True)
    train.add_argument('--train-pairs', type=Path, nargs='+')
    train.add_argument('--validation-pairs', type=Path, nargs='+')
    train.add_argument('--resume', type=Path)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--device', default='cpu')
    train.add_argument('--threads', type=int, default=2)
    train.add_argument('--seed', type=int, default=821111)
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--patience', type=int, default=10)
    train.add_argument('--learning-rate', type=float, default=3e-4)
    train.add_argument('--stop-weight', type=float, default=1.)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--workers', type=int, default=0)
    train.add_argument('--max-rollouts', type=int)
    train.add_argument('--train-windows', type=int)
    train.add_argument('--validation-windows', type=int)

    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--dataset', type=Path, nargs='+', required=True)
    evaluate.add_argument('--pairs', type=Path, nargs='+')
    evaluate.add_argument('--batch-size', type=int, default=8)
    evaluate.add_argument('--workers', type=int, default=0)
    evaluate.add_argument('--max-rollouts', type=int)
    evaluate.add_argument('--action-windows', type=int)
    evaluate.add_argument('--unguarded', action='store_true')
    evaluate.add_argument('--output', type=Path)
    split_options(evaluate, 'test')
    inference_options(evaluate)

    calibrate = commands.add_parser('calibrate')
    calibrate.add_argument('--dataset', type=Path, nargs='+', required=True)
    calibrate.add_argument('--pairs', type=Path, nargs='+')
    calibrate.add_argument('--output', type=Path, required=True)
    calibrate.add_argument('--action-windows', type=int)
    calibrate.add_argument('--batch-size', type=int, default=8)
    calibrate.add_argument('--workers', type=int, default=0)
    inference_options(calibrate)

    view = commands.add_parser('view')
    view.add_argument('--dataset', type=Path, required=True)
    view.add_argument('--index', type=int, default=0)
    view.add_argument('--pairs', type=Path)
    view.add_argument('--snapshot-version')
    view.add_argument('--publish-pure-path', action='store_true')
    view.add_argument('--domain', type=int, default=186)
    view.add_argument('--rviz', action='store_true')
    mode = view.add_mutually_exclusive_group()
    mode.add_argument('--beam', action='store_true', help='Visualize up to three distinct candidates')
    mode.add_argument('--local-goals', action='store_true', help='Visualize up to three short-range support goals')
    view.add_argument('--rollout', action='store_true',
                      help='Roll local goals on a fixed paired cost snapshot')
    view.add_argument('--route-id', type=int, help='Roll local goals from a held-out VST route start')
    view.add_argument('--lookahead', type=float, default=1.2, help='Local-goal horizontal search limit in metres')
    view.add_argument('--beam-width', type=int, default=64)
    view.add_argument('--report', type=Path, help='Save decoded paths and metrics as JSON')
    view.add_argument('--plan-only', action='store_true', help='Decode without starting ROS')
    split_options(view, 'validation')
    inference_options(view)

    args = parser.parse_args()
    for field in ('count', 'attempts', 'threads', 'epochs', 'patience', 'batch_size', 'max_rollouts', 'action_windows', 'train_windows', 'validation_windows'):
        value = getattr(args, field, None)
        if value is not None and value <= 0:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    if getattr(args, 'workers', 0) < 0:
        parser.error('--workers must be nonnegative')
    if args.command == 'generate':
        if not args.tomogram.is_file():
            parser.error(f'VST Tomogram not found: {args.tomogram}')
        report = generate(args.tomogram, args.output, args.count, args.seed,
                          args.attempts or args.count * 30, args.scene_split, args.tile_size, args.regions)
    elif args.command == 'train':
        from planner.learning_planner.neupan.training.support_train import train
        for field in ('learning_rate', 'stop_weight'):
            if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
                parser.error(f'--{field.replace("_", "-")} must be finite and positive')
        if bool(args.train_pairs) != bool(args.validation_pairs):
            parser.error('Paired fine-tuning requires --train-pairs and --validation-pairs')
        report = train(args)
    elif args.command == 'evaluate':
        from planner.learning_planner.neupan.training.support_train import evaluate
        report = evaluate(args)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    elif args.command == 'calibrate':
        from planner.learning_planner.neupan.training.support_train import calibrate
        report = calibrate(args)
    else:
        if (args.beam or args.local_goals) and (args.beam_width < 3 or args.publish_pure_path):
            parser.error('Candidate previews require beam width >= 3')
        if args.rollout and (not args.local_goals or not args.pairs or args.route_id is not None):
            parser.error('--rollout requires --local-goals and a paired snapshot')
        if args.route_id is not None and (not args.local_goals or args.pairs or args.split != 'test'):
            parser.error('--route-id requires --local-goals --split test and a VST dataset')
        if args.local_goals and (not math.isfinite(args.lookahead) or args.lookahead <= 0):
            parser.error('--lookahead must be finite and positive')
        if args.plan_only and (args.rviz or args.publish_pure_path):
            parser.error('--plan-only cannot publish or launch RViz')
        from planner.learning_planner.neupan.visualization.ros_view import view
        view(args)
        return
    print(json.dumps(report, ensure_ascii=False))
    if args.command == 'generate' and report['generated'] < args.count:
        raise SystemExit(2)  # Partial data and failure counts remain usable.


if __name__ == '__main__':
    main()
