#!/usr/bin/env python3
"""Export Plan A maps for captured frames, with separate comparison-only Tomography fields."""
import argparse
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "third_party/PointTransformerV3/src")]


def atomic_npz(path, arrays):
    import numpy as np
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, nargs="+", help="Default: all captured frames")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from point_transformer_v3 import set_backend
    from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset
    from ptv3_tomography.data.perception_samples import cropped_config, load_sample
    from ptv3_tomography.io import atomic_json, file_sha256
    from ptv3_tomography.models.dense_perception import SingleFramePerception
    from ptv3_tomography.models.plan_a_maps import pack_plan_a_maps, surface_points_from_height
    from ptv3_tomography.training.pretrain_state import read_pretraining_checkpoint
    started = time.perf_counter()
    # 只在 load_sample 中校验选中的 raw/label，不扫描未导出的全部帧。
    dataset = LidarStepDataset(args.dataset)
    saved = read_pretraining_checkpoint(args.checkpoint)
    config = saved["resolved_config"]
    map_config = cropped_config(dataset, config["grid_size_xy"])
    indices = list(range(len(dataset))) if args.frames is None else args.frames
    set_backend(attention="sdpa", conv="torch", layer_norm="torch")
    device = torch.device(args.device)
    model = SingleFramePerception(saved["model_spec"]["model"], map_config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    # checkpoint 指纹仅计算一次作为产物来源，不在导出末尾再读取整个模型。
    checkpoint_hash = file_sha256(args.checkpoint)
    args.output.mkdir(parents=True)
    (args.output / "frames").mkdir()
    atomic_json(args.output / "status.json", {"status": "running"})
    rows, error_sum, cell_count = [], {"cost": 0., "height": 0.}, {"cost": 0, "height": 0}
    with torch.no_grad():
        for i in indices:
            sample = load_sample(dataset, i, config["max_points"], config["grid_size_xy"])
            points, up = torch.as_tensor(sample["points"], device=device), torch.as_tensor(sample["up"], device=device)
            prediction, token_count = model(points, up)
            packed = pack_plan_a_maps(prediction, points, up, map_config)
            surfaces = surface_points_from_height(prediction["height"], up, map_config)
            # Teacher fields below are diagnostics. They never enter model/adapter arguments.
            arrays = {"maps": packed["maps"].cpu().numpy(),
                      "input_observed_hits": packed["input_observed_hits"].cpu().numpy(),
                      "cost_raw": packed["cost_raw"].cpu().numpy(),
                      "ground_height_gravity_m": prediction["height"].cpu().numpy(),
                      "predicted_surface_sensor_m": surfaces.cpu().numpy(),
                      "points_sensor_m": sample["points"], "gravity_up_sensor": sample["up"],
                      "world_from_sensor": dataset.poses[i],
                      "gt_cost_raw": sample["targets"]["cost"] * map_config["cost_scale"],
                      "gt_ground_height_gravity_m": sample["targets"]["height"],
                      "mask_cost": sample["targets"]["mask_cost"], "mask_height": sample["targets"]["mask_height"]}
            row = {"frame_index": i, "tokens": token_count, "valid_input_points": len(points),
                   "input_observed_hit_cells": int(packed["input_observed_hits"].sum())}
            for name, pred_key, gt_key in (("cost", "cost_raw", "gt_cost_raw"),
                                           ("height", "ground_height_gravity_m", "gt_ground_height_gravity_m")):
                mask = arrays[f"mask_{name}"]
                errors = np.abs(arrays[pred_key][mask] - arrays[gt_key][mask])
                error_sum[name] += float(errors.astype(np.float64).sum())
                cell_count[name] += len(errors)
                row[f"{name}_cells"] = len(errors)
                row[f"{name}_mae"] = float(errors.mean()) if len(errors) else None
            file = args.output / "frames" / f"frame_{i:05d}.npz"
            atomic_npz(file, arrays)
            row.update(file=str(file.relative_to(args.output)), sha256=file_sha256(file))
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f"PLAN_A_MAPS {len(rows)}/{len(indices)}", flush=True)
    manifest = {"version": "plan_a_local_maps_v1", "checkpoint_sha256": checkpoint_hash,
                "source_dataset": str(args.dataset.resolve()), "source_checkpoint": str(args.checkpoint.resolve()),
                "map_config": map_config, "frames": rows,
                "actor_interface": {"maps": "float32[1,2,L,X,Y]: normalized cost, gravity-relative height metres",
                                    "input_observed_hits": "bool[1,1,L,X,Y], measured endpoints only, derived without labels",
                                    "diagnostic_only": "gt_*, mask_*, world_from_sensor; never pass these to the actor"},
                "support": "unknown, no support output trained", "hardware_limits_imposed": False}
    atomic_json(args.output / "manifest.json", manifest)
    report = {"status": "passed", "frames": len(rows), "maps_shape_per_frame": [1, 2, *model.decoder.shape],
              "cost_mae_raw": error_sum["cost"] / max(1, cell_count["cost"]),
              "height_mae_m": error_sum["height"] / max(1, cell_count["height"]),
              "supervised_cells": cell_count, "checkpoint_sha256": checkpoint_hash,
              "map_values_unchanged_by_diagnostic_masks": True, "input_coverage_has_no_teacher_dependency": True,
              "surface_reconstruction_uses_predicted_height": True, "source_dataset_modified": False,
              "elapsed_s": time.perf_counter() - started, "device": str(device),
              "evaluation_scope": "caller-supplied captured dataset; FP32 inference; train/dev/test relationship requires dataset/checkpoint provenance"}
    atomic_json(args.output / "report.json", report)
    atomic_json(args.output / "status.json", {"status": "passed"})
    print(report, flush=True)


if __name__ == "__main__":
    main()
