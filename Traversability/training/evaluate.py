"""在独立留出的建筑划分上评估已训练 checkpoint。"""
import argparse
from pathlib import Path

import torch

from Traversability.data.dataset import PointBlocks
from Traversability.models.model import ConditionedCost
from Traversability.training.train import evaluate, make_loader
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def main():
    """读取 checkpoint，重建模型与数据集，在指定 split 上计算指标并写入 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--pretrained", action="store_true", help="评估预训练编码器与新任务头的初始基线")
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/train_local.json")
    parser.add_argument("--data", type=Path, help="独立评估数据集，含 blocks.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="保存同一监督 XYZ 上的 GT 与预测，用于绘图")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    # checkpoint 中除模型权重外还保存了配置、条件尺度等信息，故 weights_only=False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    config = checkpoint["config"] if checkpoint else read_json(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    # 验证/测试不启用随机旋转增强
    dataset = PointBlocks(
        workspace_path(args.data or config["data"]), args.split, config["model"],
        config["seed"],
    )
    # pretrained=False：网络结构由 checkpoint 完整恢复，无需再加载预训练权重
    model = ConditionedCost(config["model"], dataset.condition_scale, pretrained=checkpoint is None).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    metrics = evaluate(
        model, make_loader(dataset, config, device), device, config["amp"], dataset.metadata["cost_scale"],
        prediction_output=args.predictions,
    )
    # 记录评估划分、checkpoint 路径以及全部误差指标
    write_json(args.output, dict(split=args.split, data=str(dataset.root),
                                checkpoint=str(args.checkpoint) if args.checkpoint else None, **metrics))
    print(metrics)


if __name__ == "__main__":
    main()
