#!/usr/bin/env python3
"""Reload a pretraining checkpoint and export current-frame cost, height and spatial F."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "third_party/PointTransformerV3/src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=52)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    memory_limit = None  # No application-imposed hardware quota.
    import numpy as np
    import torch
    from point_transformer_v3 import prepare, set_backend
    from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset
    from ptv3_tomography.data.perception_samples import cropped_config, load_sample
    from ptv3_tomography.io import atomic_json, file_sha256
    from ptv3_tomography.models.dense_perception import SingleFramePerception
    from ptv3_tomography.training.pretrain_state import read_pretraining_checkpoint
    set_backend(attention="sdpa", conv="torch", layer_norm="torch")
    saved = read_pretraining_checkpoint(args.checkpoint)
    config = saved["resolved_config"]
    dataset = LidarStepDataset(args.dataset)
    map_config = cropped_config(dataset, config["grid_size_xy"])
    sample = load_sample(dataset, args.frame, config["max_points"], config["grid_size_xy"])
    device = torch.device(args.device)
    model = SingleFramePerception(saved["model_spec"]["model"], map_config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        point, _ = prepare(torch.as_tensor(sample["points"], device=device), feat=("coord",),
                           grid_size=model.voxel_size_m, center=False, pick="first")
        features = model.backbone(point)
        prediction = model.decoder(features.coord, features.feat, torch.as_tensor(sample["up"], device=device))
    arrays = {"cost_normalized": prediction["cost"].cpu().numpy(),
              "cost_raw": prediction["cost"].cpu().numpy() * map_config["cost_scale"],
              "ground_height_gravity_m": prediction["height"].cpu().numpy(),
              "feature_coordinates_sensor_m": features.coord.cpu().numpy(),
              "spatial_features": features.feat.cpu().numpy(), "gravity_up_sensor": sample["up"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    atomic_json(args.output.with_suffix(".json"), {"status": "passed", "frame_index": args.frame,
                "map_shape": list(arrays["cost_raw"].shape), "feature_shape": list(arrays["spatial_features"].shape),
                "height_units": "metres relative to LiDAR along gravity; not sensor Z",
                "dense_unobserved_predictions_are_not_verified_labels": True,
                "checkpoint_sha256": file_sha256(args.checkpoint), "device": str(device),
                "enforced_host_memory_limit_bytes": memory_limit})
    print(f"Saved cost, gravity height and spatial F to {args.output}")


if __name__ == "__main__":
    main()
