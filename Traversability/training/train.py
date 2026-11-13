"""训练代价模型；checkpoint、指标与日志统一保存在一个 runs 目录。"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Traversability.data.dataset import EpochSampler, PointBlocks, collate_blocks, to_device
from Traversability.models.model import ConditionedCost
from Traversability.training.condition_response import balanced_difference_loss, response_totals, response_report, selection_score
from Traversability.experiments.stair_metrics import StairMetrics, checkpoint_key
from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def autocast(device, precision):
    """按配置返回混合精度上下文；CPU 或 fp32 时自动关闭。"""
    return torch.autocast(
        device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
        enabled=device.type == "cuda" and precision != "fp32",
    )


def regression_loss(prediction, target, beta, difference_weight, response_config=None, pairs=None):
    """计算只在有效标签上生效的 SmoothL1 损失。

    参数:
        prediction: [Q, K] 的归一化代价预测。
        target: [Q, K] 的教师代价，NaN 表示该位置无有效监督。
        beta: SmoothL1 的分段阈值。
        difference_weight: 条件差异损失的权重；0 表示不使用。
        response_config: 变化点阈值及四参数权重，提供时使用区域均衡差异项。
        pairs: 各参数的单变量条件对，索引指向 prediction/target 的条件列。
    返回:
        (loss, 有效标签数)。条件差异项鼓励预测在不同机器人条件间
        保持与教师一致的相对变化，而不是只拟合绝对代价。
    """
    valid = torch.isfinite(target)
    # 无效位置先填 0，再通过掩码从损失中排除
    values = torch.where(valid, target, 0.0)
    loss = (F.smooth_l1_loss(prediction, values, beta=beta, reduction="none") * valid).sum() / valid.sum()
    if response_config is not None:
        loss = loss + difference_weight * balanced_difference_loss(
            prediction, target, pairs, beta, response_config["change_threshold"],
            response_config["parameter_weights"])
    elif difference_weight:
        # 以第一组条件为基准，约束其余条件相对基准的代价差
        common = valid[:, 1:] & valid[:, :1]
        delta_prediction = prediction[:, 1:] - prediction[:, :1]
        delta_target = values[:, 1:] - values[:, :1]
        delta_loss = F.smooth_l1_loss(delta_prediction, delta_target, beta=beta, reduction="none")
        loss = loss + difference_weight * (delta_loss * common).sum() / common.sum().clamp_min(1)
    return loss, valid.sum()


def make_loader(dataset, config, device, sampler=None, collate_fn=collate_blocks):
    """按配置构建 DataLoader；CUDA 时启用 pinned memory 与预取。"""
    options = dict(
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        pin_memory=device.type == "cuda", collate_fn=collate_fn, sampler=sampler,
        persistent_workers=config["num_workers"] > 0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    if config["num_workers"]:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


@torch.no_grad()
def evaluate(model, loader, device, precision, cost_scale, prediction_output=None):
    """在给定 loader 上统计 MAE、危险漏检率和条件差异误差。

    危险漏检定义为：教师代价达到 1（障碍）但预测低于 0.6。
    指标按参与训练的条件分别累计，同时给出总体值。
    """
    model.eval()
    count = len(loader.dataset.profile_ids)
    # 四行分别累计：绝对误差、有效点数、危险漏检数、危险点总数
    totals = torch.zeros((4, count), device=device, dtype=torch.float64)
    delta = torch.zeros(2, device=device, dtype=torch.float64)
    response = torch.zeros((len(loader.dataset.condition_names), 10), device=device, dtype=torch.float64)
    snapshots = {name:[] for name in ("coord", "prediction", "target", "scene", "block")}
    stair_monitor = StairMetrics(loader.dataset) if loader.dataset.metadata.get("stair_evaluation") else None
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device, precision):
            prediction = model(batch)
        target = batch["target"]
        if stair_monitor is not None:
            stair_monitor.add(batch, prediction, target)
        response += response_totals(prediction, target, loader.dataset.condition_pairs)
        if prediction_output is not None:
            indices = batch["query_batch"]
            coord = torch.bmm(batch["query_coord"][:, None], batch["rotation"][indices]).squeeze(1)
            coord = coord + batch["origin"][indices]
            snapshots["coord"].append(coord.cpu().numpy())
            snapshots["prediction"].append(prediction.cpu().numpy())
            snapshots["target"].append(target.cpu().numpy())
            rows = indices.cpu().numpy()
            for name in ("scene", "block"):
                snapshots[name].append(np.asarray(batch[name])[rows])
        valid = torch.isfinite(target)
        values = torch.where(valid, target, 0.0)
        totals[0] += ((prediction - values).abs() * valid).sum(0)
        totals[1] += valid.sum(0)
        dangerous = valid & (values >= 1.0)
        totals[2] += (dangerous & (prediction < 0.6)).sum(0)
        totals[3] += dangerous.sum(0)
        # 条件差异误差：pred(k)-pred(nominal) 与 target(k)-target(nominal) 的差
        common = valid[:, 1:] & valid[:, :1]
        error = ((prediction[:, 1:] - prediction[:, :1]) - (values[:, 1:] - values[:, :1])).abs()
        delta[0] += (error * common).sum()
        delta[1] += common.sum()
    values, change = totals.cpu().numpy(), delta.cpu().tolist()
    mae = float(values[0].sum() / values[1].sum())
    ids = loader.dataset.profile_ids
    if prediction_output is not None:
        prediction_output = Path(prediction_output)
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prediction_output, **{name:np.concatenate(values) for name,values in snapshots.items()},
                 profiles=np.asarray(ids), condition_names=np.asarray(loader.dataset.condition_names))
    # cost_mae_raw 将归一化误差换算回原始 0..cost_scale 量纲
    responses = response_report(response, loader.dataset.condition_names)
    result = dict(
        cost_mae=mae, cost_mae_raw=mae * cost_scale,
        parameter_response=responses, response_selection_score=selection_score(mae, responses),
        dangerous_underprediction_pct=(
            float(100 * values[2].sum() / values[3].sum()) if values[3].sum() else None
        ),
        condition_difference_mae=change[0] / change[1] if change[1] else None,
        profiles={
            name: dict(
                cost_mae=float(values[0, i] / values[1, i]),
                labelled_points=int(values[1, i]),
                dangerous_underprediction_pct=(
                    float(100 * values[2, i] / values[3, i]) if values[3, i] else None
                ),
            )
            for i, name in enumerate(ids)
        },
    )
    if stair_monitor is not None:
        result["stair_metrics"] = stair_monitor.report()
    return result


def save_checkpoint(path, payload):
    """先写临时文件再原子替换，避免中断时留下损坏的 checkpoint。"""
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    """训练主循环：恢复或初始化模型，逐 epoch 训练、验证并保存断点。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/train_local.json")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--resume", type=Path, help="从已有 checkpoint 继续训练")
    weights.add_argument("--initialize", type=Path, help="用既有模型初始化新训练，按条件名扩展FiLM输入，不继承优化器")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, help="总训练轮数；续训延长时同步延长余弦学习率周期")
    parser.add_argument("--stop-after-epoch", type=int, help="仅限制本次执行轮次，不改变总学习率调度周期")
    args = parser.parse_args()
    # resume 时沿用 checkpoint 内的配置，避免训练参数前后不一致
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = checkpoint["config"] if checkpoint else read_json(args.config)
    previous_epochs = config["epochs"]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.output:
        config["output"] = str(args.output)
    if args.device:
        config["device"] = args.device
    if args.initialize:
        config["initialization_checkpoint"] = str(args.initialize.resolve())
    # 固定线程数与各框架随机种子，保证可复现
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    data = workspace_path(config["data"])
    output = workspace_path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # 训练集启用随机旋转增强；验证集保持确定性
    training = PointBlocks(data, "train", config["model"], config["seed"], augment=True)
    # 消融实验可复用同一数据根目录，只选择对应的训练建筑。
    if "train_scene_ids" in config:
        training.rows = [r for r in training.rows if r["scene"] in config["train_scene_ids"]]
    # 固定独立验证集，避免扩充条件或训练建筑时改变选优口径。
    validation = PointBlocks(
        workspace_path(config.get("validation_data", str(data))), config["evaluation_split"], config["model"], config["seed"]
    )
    metadata = training.metadata
    sampler = EpochSampler(training, config["seed"], config.get("blocks_per_epoch"), config.get("stair_sampling"))
    train_loader = make_loader(training, config, device, sampler)
    val_loader = make_loader(validation, config, device)
    # 从头训练时加载预训练编码器；resume 时由 checkpoint 覆盖全部权重
    model = ConditionedCost(config["model"], training.condition_scale,
                            pretrained=checkpoint is None and args.initialize is None).to(device)
    if args.initialize:
        initial = torch.load(args.initialize, map_location="cpu", weights_only=False)
        weights = initial["model"].copy()
        expanded = torch.zeros_like(model.condition_head[0].weight, device="cpu")
        for column, name in enumerate(initial["condition_names"]):
            expanded[:, training.condition_names.index(name)] = weights["condition_head.0.weight"][:, column]
        weights["condition_head.0.weight"] = expanded
        weights["condition_scale"] = model.condition_scale.detach().cpu()
        model.load_state_dict(weights)
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_parameters}
    # 编码器与新增头使用不同学习率：编码器通常更小
    optimizer = torch.optim.AdamW([
        dict(params=encoder_parameters, lr=config["encoder_learning_rate"]),
        dict(params=[p for p in model.parameters() if id(p) not in encoder_ids], lr=config["learning_rate"]),
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    # GradScaler 仅在 CUDA + fp16 时启用；bf16 不需要梯度缩放
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp"] == "fp16")
    start_epoch, next_batch, step, best = 0, 0, 0, float("inf")
    history = []
    best_key = None
    selection_baseline = (read_json(workspace_path(config["stair_selection_baseline"]))
                          if "stair_selection_baseline" in config else None)
    # epoch_totals 依次累计：加权损失、有效标签数、块数
    epoch_totals = torch.zeros(3, device=device, dtype=torch.float64)
    epoch_elapsed = 0.0
    if checkpoint:
        # 恢复模型、优化器、调度器、AMP 状态与 epoch 内进度
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, next_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
        best, history = checkpoint["best_score"], checkpoint["history"]
        best_key = checkpoint.get("best_key")
        epoch_totals = checkpoint["epoch_totals"].to(device)
        epoch_elapsed = checkpoint["epoch_elapsed_s"]
        if config["epochs"] != previous_epochs:
            # 原短周期可能已经衰减到0。按新的总周期重定位学习率，保留AdamW动量。
            factor = (1 + math.cos(math.pi * start_epoch / config["epochs"])) / 2
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * factor
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"], eta_min=scheduler.eta_min, last_epoch=start_epoch - 1)
            config["schedule_extension"] = dict(
                previous_epochs=previous_epochs, total_epochs=config["epochs"], resumed_epoch=start_epoch,
                learning_rates=scheduler.get_last_lr(), mode="cosine_rephased_to_total_epochs")
            print(f"Extended training: epoch {start_epoch}/{config['epochs']}, "
                  f"learning_rates={scheduler.get_last_lr()}", flush=True)
        # 恢复随机状态，确保中断后续训与不中断训练等价
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    write_json(output / "config.json", config)
    epoch = start_epoch
    resume_epoch = start_epoch
    started = time.perf_counter()

    def payload(epoch_to_resume, batch_to_resume, totals, elapsed):
        """组装可恢复训练的完整 checkpoint 内容（含 RNG 与 epoch 内进度）。"""
        return dict(
            config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            condition_names=training.condition_names, condition_scale=training.condition_scale,
            profile_ids=training.profile_ids,
            coordinate_frame=metadata["coordinate_frame"], inference_query_source="input_pcd_xyz",
            blocks_config=metadata["blocks_config"],
            block_size_m=metadata["blocks_config"]["core_size_m"],
            context_m=metadata["blocks_config"]["context_m"],
            cost_scale=metadata["cost_scale"], epoch=epoch_to_resume, next_batch=batch_to_resume,
            step=step, best_score=best, history=history, epoch_totals=totals.detach().cpu(),
            best_key=best_key,
            epoch_elapsed_s=elapsed, torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )

    try:
        stop_epoch = args.stop_after_epoch or config["epochs"]
        for epoch in range(start_epoch, min(config["epochs"], stop_epoch)):
            resume_epoch = epoch
            # 把当前 epoch 注入采样器，使数据增强随 epoch 变化
            sampler.epoch = epoch
            model.train()
            # 前若干个 epoch 冻结编码器，只训练查询/FiLM/代价头
            frozen = epoch < config["freeze_encoder_epochs"]
            model.encoder.requires_grad_(not frozen)
            if frozen:
                # 冻结时切到 eval，固定 BatchNorm/统计行为
                model.encoder.eval()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for batch_index, batch in enumerate(train_loader):
                # 断点续训时跳过当前 epoch 已处理过的 batch
                if batch_index < next_batch:
                    continue
                block_count = len(batch["point"]["offset"])
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, config["amp"]):
                    prediction = model(batch)
                    loss, labelled = regression_loss(
                        prediction, batch["target"], config["smooth_l1_beta"],
                        config["condition_difference_weight"],
                        config.get("balanced_response"), training.condition_pairs,
                    )
                # AMP 缩放后的反向传播与参数更新
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # 用有效标签数加权累计 epoch 平均损失
                epoch_totals[0] += loss.detach() * labelled
                epoch_totals[1] += labelled
                epoch_totals[2] += block_count
                step += 1
                next_batch = batch_index + 1
                if step % config["log_every_steps"] == 0:
                    print(f"epoch={epoch+1} step={step} loss={loss.item():.6f}", flush=True)
                if step % config["save_every_steps"] == 0:
                    # 定期保存 last.pt，包含当前 epoch 内进度
                    elapsed = epoch_elapsed + time.perf_counter() - started
                    save_checkpoint(output / "last.pt", payload(epoch, next_batch, epoch_totals, elapsed))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall = epoch_elapsed + time.perf_counter() - started
            totals = epoch_totals.cpu().tolist()
            metric, improved = None, False
            if (epoch + 1) % config.get("evaluation_every_epochs", 1) == 0 or epoch + 1 == config["epochs"]:
                metric = evaluate(model, val_loader, device, config["amp"], metadata["cost_scale"])
                score = metric[config.get("checkpoint_metric", "cost_mae")]
                if selection_baseline is not None:
                    key = checkpoint_key(metric, selection_baseline)
                    improved = best_key is None or key < tuple(best_key)
                    metric["selection_key"] = list(key)
                    if improved:
                        best_key, best = list(key), score
                else:
                    improved = score < best
                    best = min(best, score)
            row = dict(
                epoch=epoch+1, step=step, loss=totals[0] / totals[1],
                validation=metric, train_wall_s=wall, blocks_per_s=totals[2] / wall,
                encoder_frozen=frozen, true_batch_size=config["batch_size"],
                learning_rates=[group["lr"] for group in optimizer.param_groups],
                cuda_peak_allocated_gib=(
                    torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
                ),
            )
            history.append(row)
            scheduler.step()
            epoch_totals.zero_()
            next_batch, epoch_elapsed = 0, 0.0
            resume_epoch = epoch + 1
            started = time.perf_counter()
            # 新 epoch 起点保存 last.pt；验证指标提升时另存 best.pt
            state = payload(epoch+1, 0, epoch_totals, 0.0)
            save_checkpoint(output / "last.pt", state)
            if improved:
                save_checkpoint(output / "best.pt", state)
            write_json(output / "history.json", history)
            validation_text = f" val_mae={metric['cost_mae']:.5f}" if metric is not None else ""
            print(f"epoch={epoch+1} loss={row['loss']:.6f}{validation_text} train_s={wall:.2f}", flush=True)
    except KeyboardInterrupt:
        # 手动中断时保存当前进度，方便用 --resume 继续
        save_checkpoint(
            output / "last.pt",
            payload(resume_epoch, next_batch, epoch_totals, epoch_elapsed + time.perf_counter() - started),
        )
        print(f"Interrupted. Resume from {output / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
