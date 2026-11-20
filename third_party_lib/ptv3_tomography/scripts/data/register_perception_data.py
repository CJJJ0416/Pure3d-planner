#!/usr/bin/env python3
"""Create a relative-path registry for one portable capture. Add further captures explicitly."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset
from ptv3_tomography.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--building-id", required=True, help="Variants of the same building must share this ID")
    parser.add_argument("--capture-id", default="capture_001")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = LidarStepDataset(args.dataset)
    entry = {"id": args.capture_id, "building_id": args.building_id, "split": args.split,
             "path": os.path.relpath(dataset.directory, args.output.resolve().parent)}
    atomic_json(args.output, {"version": "single_frame_registry_v1", "datasets": [entry]})
    print(f"Registered {len(dataset)} frames as {args.split}. All frames of this building remain together.")


if __name__ == "__main__":
    main()
