#!/usr/bin/env python3
"""Train current-frame PTv3 + map decoder over complete registered captures."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "third_party/PointTransformerV3/src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/perception_pretrain_rtx4060_batch.json")
    parser.add_argument("--output", type=Path, required=True, help="Run name or runs/<name>; training products always go under repository runs/")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path, help="Restore model, optimizer, RNG and sample cursor")
    start.add_argument("--init-checkpoint", type=Path, help="Initialize model weights only, with a fresh optimizer and cursor")
    parser.add_argument("--max-updates", type=int, help="Stop at this global update boundary, preserving resume state")
    args = parser.parse_args()
    from ptv3_tomography.training.batch_pretrain import train
    train(args.registry, args.config, args.output, args.resume, args.max_updates, args.init_checkpoint)


if __name__ == "__main__":
    main()
