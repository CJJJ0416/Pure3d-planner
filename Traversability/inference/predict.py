"""地图坐标系建筑 PCD + 机器人条件 -> 全部原始点的三维代价点云。"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from Traversability.data.dataset import collate_queries, to_device
from Traversability.inference.inference import QueryBlocks
from Traversability.models.model import ConditionedCost
from Traversability.training.train import autocast, make_loader
from common.pointcloud_io import write_pcd
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


@torch.inference_mode()
def predict(model, dataset, config, device):
    """把核心块内的逐点预测散布回原始建筑 PCD 行序。

    每个查询点只属于一个核心块，因此可以直接按原始 query_index 写回，
    得到与建筑 PCD 全部原始点一一对应的代价数组。
    """
    result = np.empty(dataset.scene["query_points"], dtype=np.float32)
    model.eval()
    loader = make_loader(dataset, config, device, collate_fn=collate_queries)
    completed = 0
    for number, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        with autocast(device, config["amp"]):
            cost = model(batch)
        result[batch["query_index"].cpu().numpy()] = cost[:, 0].cpu().numpy()
        completed += len(cost)
        if number % 10 == 0 or number == len(loader):
            print(f"Prediction: {number}/{len(loader)} batches, {completed:,}/{len(result):,} points", flush=True)
    return result


def main():
    """只从建筑 PCD 自动准备三维分块，保持地图原点坐标系。"""
    entry_started = time.perf_counter()
    timings = {}
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--pretrained", action="store_true", help="仅验证接口：预训练编码器加未训练任务头")
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/train_local.json")
    parser.add_argument("--pcd", type=Path, required=True, help="地图原点坐标系下的建筑 PCD")
    parser.add_argument("--geometry-python", default=sys.executable,
                        help="运行三维分块的 Python，需有 NumPy/SciPy")
    parser.add_argument("--condition", type=float, nargs="+", required=True, metavar="VALUE",
                        help="按模型 condition_names 输入；当前为 clearance_min slope_max step_max stand_ratio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, help="推理块批量，默认沿用checkpoint配置")
    parser.add_argument("--num-workers", type=int, help="数据加载进程数，默认沿用checkpoint配置")
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    config = checkpoint["config"] if checkpoint else read_json(args.config)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if checkpoint:
        condition_scale = checkpoint["condition_scale"]
        blocks_config = checkpoint["blocks_config"]
    else:
        robots = read_json(ROOT / "Traversability/configs/robots.json")
        condition_scale = [robots["condition_scale"][robots["condition_names"].index(name)]
                           for name in config["model"]["condition_names"]]
        blocks_config = read_json(ROOT / "Traversability/configs/dataset.json")["blocks"]
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    output = workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root, scene = output / "geometry", "inference"
    settings_path = output / "geometry_config.json"
    write_json(settings_path, blocks_config)
    timings["checkpoint_config_and_setup_s"] = time.perf_counter() - started
    started = time.perf_counter()
    subprocess.run([
        args.geometry_python, "-m", "Traversability.inference.inference",
        "--pcd", str(workspace_path(args.pcd)),
        "--output", str(root), "--config", str(settings_path),
    ], cwd=ROOT, check=True)
    timings["geometry_preparation_process_s"] = time.perf_counter() - started
    started = time.perf_counter()
    dataset = QueryBlocks(root, scene, config["model"], config["seed"], args.condition)
    model = ConditionedCost(config["model"], condition_scale, pretrained=checkpoint is None).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings["dataset_model_and_cuda_setup_s"] = time.perf_counter() - started
    started = time.perf_counter()
    cost = predict(model, dataset, config, device)
    prediction_s = time.perf_counter() - started
    timings["prediction_s"] = prediction_s
    started = time.perf_counter()
    coord = np.load(root / dataset.scene["arrays"]["query_coord"], mmap_mode="r")
    write_pcd(output / "prediction.pcd", coord, cost)
    np.savez(output / "prediction.npz", coord=coord, cost=cost, condition=args.condition)
    timings["prediction_output_s"] = time.perf_counter() - started
    report = dict(
        checkpoint=str(args.checkpoint.resolve()) if checkpoint else None,
        task_heads="trained checkpoint" if checkpoint else "untrained, interface validation only",
        data=str(root), scene=scene, blocks=len(dataset),
        source_pcd=str(workspace_path(args.pcd)), query_source="input_pcd_xyz",
        condition_names=config["model"]["condition_names"], condition=args.condition, points=len(coord),
        fixed_teacher_conditions=config["model"]["fixed_teacher_conditions"],
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        cost_range=[float(cost.min()), float(cost.max())],
        coordinate_frame="map", map_origin_preserved=True,
        coordinates="original input PCD XYZ in metres, original point order", cost_scale="0..1",
        teacher_xyz_used=False, teacher_cost_used=False, slices_computed=False, prediction_s=prediction_s,
        cuda_peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                                 if device.type == "cuda" else None),
        timings=timings, entry_to_prediction_files_s=time.perf_counter() - entry_started,
    )
    write_json(output / "prediction.json", report)
    print(report, flush=True)
    print(output / "prediction.pcd", flush=True)


if __name__ == "__main__":
    main()
