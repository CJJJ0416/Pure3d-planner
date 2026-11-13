"""在真实数据上运行一次代价模型前向，不执行训练、不保存 checkpoint。

该脚本用于验证数据管线、编码器加载、FiLM/cost 头前向是否可用，并报告
显存占用与首次前向耗时；其中的网络头尚未训练，输出不代表有效预测。
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from Traversability.data.dataset import PointBlocks, collate_blocks, to_device
from Traversability.models.model import ConditionedCost
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def main():
    """组装指定块的真实批次，执行编码与解码，并保存前向结果与报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/train_local.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--indices", nargs="+", type=int, default=[0], help="要前向的块索引")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cost_model_forward")
    args = parser.parse_args()
    config = read_json(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    device = torch.device(args.device or config["device"])
    dataset = PointBlocks(workspace_path(config["data"]), args.split, config["model"], config["seed"])
    # 记录数据准备（mmap 读取、体素化、近邻映射）的耗时
    stamp = time.perf_counter()
    batch = collate_blocks([dataset[index] for index in args.indices])
    preparation_s = time.perf_counter() - stamp
    # 真正调用网络前删除 GT，确保前向过程完全不接触标签
    batch.pop("target")
    model = ConditionedCost(config["model"], dataset.condition_scale).to(device).eval()
    batch = to_device(batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    stamp = time.perf_counter()
    # inference_mode 关闭梯度；autocast 根据配置使用 bf16/fp16
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16 if config["amp"] == "bf16" else torch.float16,
        enabled=device.type == "cuda" and config["amp"] != "fp32",
    ):
        geometry = model.encode_geometry(batch)
        prediction = model.decode_cost(geometry, batch["conditions"], batch["query_batch"])
        # 只保留第一组条件再解码一次，验证几何复用路径与输出形状
        one_condition = model.decode_cost(geometry, batch["conditions"][:, :1], batch["query_batch"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_s = time.perf_counter() - stamp
    query_batch = batch["query_batch"]
    # 将块局部坐标经旋转与平移还原到世界坐标
    world = torch.bmm(batch["query_coord"][:, None], batch["rotation"][query_batch]).squeeze(1) + batch["origin"][query_batch]
    output = workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "forward.npz", query_world=world.cpu().numpy(),
             prediction=prediction.cpu().numpy(), query_index=batch["query_index"].cpu().numpy(),
             query_batch=query_batch.cpu().numpy(), conditions=batch["conditions"].cpu().numpy())
    # 报告显式记录“未训练、未优化、未保存 checkpoint”，避免误读结果
    report = dict(
        stage="pretrained encoder + untrained query/FiLM/cost heads, forward only",
        config=config, split=args.split, block_indices=args.indices, scenes=batch["scene"], blocks=batch["block"],
        condition_names=dataset.condition_names, condition_scale=dataset.condition_scale,
        profile_ids=dataset.profile_ids, condition_shape=list(batch["conditions"].shape),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        encoder_weights=model.encoder.weights_info, encoder_channels=model.encoder.feature_channels,
        parameters=sum(p.numel() for p in model.parameters()),
        raw_input_points=int(batch["input_counts"].sum()),
        local_voxels=len(batch["point"]["coord"]), global_voxels=len(batch["global_point"]["coord"]),
        query_points=len(prediction), query_feature_shape=list(geometry.shape),
        local_neighbors=list(batch["local_index"].shape), global_neighbors=list(batch["global_index"].shape),
        output_shape=list(prediction.shape), reused_geometry_single_condition_shape=list(one_condition.shape),
        prediction_min=float(prediction.min()), prediction_max=float(prediction.max()),
        finite=bool(torch.isfinite(prediction).all()), gt_passed_to_model=False,
        preparation_s=preparation_s, first_forward_s=forward_s,
        forward_timing="Includes first-use CUDA/Triton compilation. Not a training throughput measurement.",
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        optimizer_steps=0, checkpoint_saved=False,
    )
    write_json(output / "forward.json", report)
    print({k:v for k,v in report.items() if k not in ("config", "encoder_weights")}, flush=True)
    print(output / "forward.json", flush=True)


if __name__ == "__main__":
    main()
