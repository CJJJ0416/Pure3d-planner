#!/usr/bin/env python3
"""Prepare memory-mapped point/map arrays once for a registered training dataset."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "third_party/PointTransformerV3/src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from ptv3_tomography.data.training_cache import prepare_training_cache
    from ptv3_tomography.io import read_json
    config = read_json(args.config)
    torch.set_num_threads(config.get("torch_cpu_threads", 4))
    metadata = prepare_training_cache(args.registry, config, args.output)
    print(f"Prepared training arrays in {metadata['elapsed_s']:.3f}s: {args.output}")


if __name__ == "__main__":
    main()
