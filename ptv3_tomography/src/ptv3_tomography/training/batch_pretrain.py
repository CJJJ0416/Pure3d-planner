"""Packed-cloud PTv3 training with a batched map head and prefetched mmap data."""
# 官方比对详见 src/ptv3_tomography/docs/OFFICIAL_TRAINING_COMPARISON.md。
# 模型来源：Pointcept/PointTransformerV3 @ 4e277e6。
# 训练参考：Pointcept/Pointcept @ 77f87ab 的 Trainer、CheckpointSaver/Loader。
# 本文件实现当前帧地图回归适配，当前使用单 GPU。
import json
import logging
import math
import os
import resource
import signal
import time
from pathlib import Path

import torch
from point_transformer_v3.modules import Block
from torch.utils.data import DataLoader, Sampler

from ptv3_tomography.data.training_cache import CachedFrames, collate_frames, prepare_training_cache
from ptv3_tomography.io import read_json
from ptv3_tomography.paths import relocated_path, training_run
from ptv3_tomography.models.dense_perception import SingleFramePerception
from ptv3_tomography.models.perception import batched_perception_loss
from ptv3_tomography.training.runtime import autocast, configure
from ptv3_tomography.training.pretrain_state import capture_rng, read_pretraining_checkpoint, restore_training_rng


def write_json(path, value):
    """Replace training metadata without a synchronous disk flush in the hot path."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def link_checkpoint(source, destination):
    """Retain an immutable checkpoint alias without copying the model bytes."""
    # last.pt 每次原子替换为新 inode，best.pt 因而保留选中时的权重。
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".link")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


# 官方单卡由 DataLoader shuffle，多卡由 DistributedSampler.set_epoch 洗牌。
# 本项目用 seed+epoch 生成顺序，再由 next_offset 接续剩余帧。
class EpochSampler(Sampler):
    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def loader(dataset, config, batch_size, sampler=None):
    # 对应 Pointcept Trainer.build_train_loader：真实 batch、多 worker、pin_memory。
    # 本项目额外指定 spawn 和预取数。worker 读取 CPU mmap，预取占用主机内存。
    workers = config["num_workers"]
    options = {"num_workers": workers, "pin_memory": True, "collate_fn": collate_frames,
               "batch_size": batch_size, "sampler": sampler}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=config.get("prefetch_factor", 2),
                       multiprocessing_context="spawn")
    return DataLoader(dataset, **options)


def on_device(batch, device):
    # 对应官方 Trainer.run_step 的 cuda(non_blocking=True)，配合锁页内存传输。
    point = {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
             for name, value in batch["point"].items()}
    return point, batch["up"].to(device, non_blocking=True), {
        name: value.to(device, non_blocking=True) for name, value in batch["targets"].items()}


def metrics(prediction, targets, loss, parts):
    # detach 只用于日志统计。训练 loss、地图头和骨干特征之间的梯度路径保留。
    values = [loss.detach() * len(targets["cost"]),
              parts["cost"].detach() * len(targets["cost"]),
              parts["height"].detach() * len(targets["cost"])]
    for name in ("cost", "height"):
        mask = targets["mask_" + name]
        safe = torch.where(mask, targets[name], prediction[name].detach())
        values.append((prediction[name].detach() - safe).abs().sum())
    values += [targets["mask_cost"].sum(), targets["mask_height"].sum(),
               loss.new_tensor(len(targets["cost"]))]
    return torch.stack(values).detach()


def summary(stats, cost_scale):
    # 回归适配：loss 汇总 batch 均值，MAE 按各目标的有效格累计。
    loss, cost, height, cost_abs, height_abs, cost_cells, height_cells, frames = stats.cpu().tolist()
    return {"frames_used": int(frames), "mean_loss": loss / max(frames, 1),
            "cost_loss": cost / max(frames, 1), "height_loss": height / max(frames, 1),
            "cost_mae_raw": cost_abs * cost_scale / cost_cells if cost_cells else None,
            "height_mae_m": height_abs / height_cells if height_cells else None,
            "cost_cells": int(cost_cells), "height_cells": int(height_cells)}


@torch.no_grad()
def evaluate(model, data_loader, config, device, cost_scale):
    model.eval()
    stats = torch.zeros(8, device=device)
    for batch in data_loader:
        point, up, targets = on_device(batch, device)
        with autocast(config, device):
            prediction = model.forward_batch(point, up, batch["max_tokens"])
        loss, parts = batched_perception_loss(prediction, targets, config["loss"])
        stats += metrics(prediction, targets, loss, parts)
    model.train()
    return summary(stats, cost_scale)


def train(registry_path, config_path, output, resume=None, max_updates=None,
          init_checkpoint=None):
    started = time.perf_counter()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    output = training_run(output)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"PTV3_RUN_DIR {output}", flush=True)
    registry_path = relocated_path(registry_path)
    config = read_json(config_path)
    torch.set_num_threads(config.get("torch_cpu_threads", 4))
    device = configure(config)
    # 项目适配：静态体素和监督掩码只准备一次，训练期直接读数组。
    # 官方 ScanNet 配方的随机旋转、缩放与 GridSample 原本在数据变换中执行。
    cache = Path(config.get("training_cache", "training_cache_teacher_v1"))
    if not cache.is_absolute():
        cache = output / "cache" / cache
    if not (cache / "metadata.json").exists():
        prepare_training_cache(registry_path, config, cache)
    metadata = read_json(cache / "metadata.json")
    map_config = metadata["map_config"]
    train_data, dev_data = CachedFrames(cache, "train"), CachedFrames(cache, "dev")
    batch_size = config["batch_size"]
    # 官方通过 DataLoader.drop_last 处理尾部。这里先截断 sampler 的帧序列。
    # 本地 drop_last=False 会保留尾 batch，H100 配方设 True。
    drop_last = config.get("drop_last", True) and len(train_data) >= batch_size
    epoch_frames = len(train_data) // batch_size * batch_size if drop_last else len(train_data)
    steps_per_epoch = math.ceil(epoch_frames / batch_size)
    total_epochs = config.get("schedule_total_epochs", config["epochs"])
    sampler = EpochSampler()
    train_loader = loader(train_data, config, batch_size, sampler)
    dev_loader = loader(dev_data, config, config.get("eval_batch_size", batch_size)) if len(dev_data) else None
    # 完整官方编码器/解码器与地图头共同训练，所有参数进入下面的优化器分组。
    # 参数、梯度和 Adam 状态由模型大小决定，batch 主要改变激活和工作区占用。
    model = SingleFramePerception(config["model"], map_config).to(device)
    # 对应官方 ScanNet param_dicts(keyword="block") 的低学习率组。
    # 这里按官方 Block 类型识别参数，适配 pip 包的模块命名。
    block_ids = {id(parameter) for module in model.backbone.modules() if isinstance(module, Block)
                 for parameter in module.parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) not in block_ids]},
              {"params": [p for p in model.parameters() if id(p) in block_ids]}]
    max_lrs = [config["learning_rate"], config["block_learning_rate"]]
    # AdamW 沿用官方优化器。CUDA fused=True 是本项目的执行设置。
    optimizer = torch.optim.AdamW(groups, lr=max_lrs[0], weight_decay=config["weight_decay"],
                                  fused=device.type == "cuda")
    # 对应官方 Trainer.build_scheduler 与 ScanNet OneCycle 配方。
    # 本项目 epoch 是实际数据遍历次数，schedule_total_epochs 固定整次训练预算。
    # 官方 eval_epoch 是外层循环数，还需结合 data.train.loop 理解。
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs,
        total_steps=total_epochs * steps_per_epoch, pct_start=config.get("warmup_fraction", .05),
        anneal_strategy="cos", div_factor=10., final_div_factor=1000.)
    progress = {"epoch": 0, "next_offset": 0, "global_step": 0, "epochs": [], "best_dev_score": None}
    stats = torch.zeros(8, device=device)
    initialization = {"mode": "random"}
    # 对应官方 CheckpointLoader 的完整状态恢复，额外保留 RNG 和帧游标。
    if resume:
        saved = read_pretraining_checkpoint(resume)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        progress = saved["progress"]
        stats = saved["epoch_stats"].to(device)
        initialization = saved["initialization"]
        restore_training_rng(saved["rng"], device)
        del saved
    # 仅权重初始化用于新 batch/配方，优化器与 OneCycle 从本次运行起点建立。
    elif init_checkpoint:
        saved = read_pretraining_checkpoint(init_checkpoint)
        model.load_state_dict(saved["model"], strict=True)
        initialization = {"mode": "weights_only", "checkpoint": str(Path(init_checkpoint).resolve()),
                          "source_global_step": saved["progress"]["global_step"], "optimizer_restored": False}
        del saved
    write_json(logs / "resolved_config.json", config)
    write_json(logs / "initialization.json", initialization)
    write_json(logs / "data_summary.json", {"training_cache": str(cache), "frames": {
        name: len(value["frames"]) for name, value in metadata["splits"].items()}, "map_config": map_config})
    stopped = {"requested": False}
    handlers = {sig: signal.signal(sig, lambda *_: stopped.update(requested=True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    checkpoint_s, local_updates, trained_frames = 0., 0, 0
    checkpoint_timing = {"serialize_s": 0., "retain_links_s": 0.}
    current_batch_frames = []
    step_records, pending, first_update_time = [], [], None
    train_start = time.time()
    write_json(logs / "status.json", {"status": "running", "train_start_unix_s": train_start})
    print(f"PTV3_BATCH_START batch={batch_size} workers={config['num_workers']} frames={len(train_data)} "
          f"total_schedule_steps={total_epochs * steps_per_epoch}", flush=True)

    # 官方 InformationWriter 每步读取标量，本项目按窗口归约后写日志。
    # batch_time_s 是窗口摊销时间，gpu_batch_time_s 是 CUDA 事件区间，均不等于 SM 利用率。
    def flush_updates(window_started):
        nonlocal first_update_time
        if not pending:
            return
        if device.type == "cuda":
            pending[-1]["events"][1].synchronize()
        values = torch.stack([item.pop("loss_tensor") for item in pending]).cpu().tolist()
        elapsed = time.perf_counter() - window_started
        compute_s = max(0., elapsed - sum(item["data_time_s"] for item in pending)) / len(pending)
        now = time.time()
        if first_update_time is None:
            first_update_time = now
        for item, value in zip(pending, values):
            events = item.pop("events")
            item.update(loss=value, batch_time_s=compute_s, unix_s=now,
                        gpu_batch_time_s=events[0].elapsed_time(events[1]) / 1000 if events else None)
            item["samples_per_s"] = item["batch_size"] / (item["data_time_s"] + compute_s)
        with (logs / "updates.jsonl").open("a") as stream:
            stream.write("".join(json.dumps(item, allow_nan=False) + "\n" for item in pending))
        print("PTV3_BATCH " + json.dumps(pending[-1]), flush=True)
        step_records.extend(pending)
        pending.clear()
        write_json(logs / "status.json", {"status": "running", "epoch": progress["epoch"] + 1,
                    "global_step": progress["global_step"], "train_start_unix_s": train_start,
                    "first_update_unix_s": first_update_time})

    # 只保留 last.pt 和经过验证选择的 best.pt，不生成阶段或编号副本。
    # 同一文件系统用硬链接更新 best，避免再次序列化完整模型。
    # 增加项目进度与 RNG，BF16 路径的 scaler 为 None。
    def save(best=False):
        nonlocal checkpoint_s
        before = time.perf_counter()
        payload = {"format": "ptv3_single_frame_pretraining_v2",
                   "resolved_config": config, "model_spec": {"model": config["model"], "map": map_config},
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": None, "progress": progress,
                   "epoch_stats": stats.cpu(), "rng": capture_rng(device), "initialization": initialization}
        temporary = output / "last.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / "last.pt")
        serialized = time.perf_counter()
        checkpoint_timing["serialize_s"] += serialized - before
        if best and config.get("save_best", True):
            link_checkpoint(output / "last.pt", output / "best.pt")
        checkpoint_timing["retain_links_s"] += time.perf_counter() - serialized
        checkpoint_s += time.perf_counter() - before

    try:
        model.train()
        for epoch in range(progress["epoch"], config["epochs"]):
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            order = torch.randperm(len(train_data), generator=generator).tolist() if config["shuffle"] else list(range(len(train_data)))
            sampler.indices = order[progress["next_offset"]:epoch_frames]
            epoch_started = time.perf_counter()
            data_ready = time.perf_counter()
            window_started = data_ready
            for batch in train_loader:
                # 已有 CPU 元数据，仅在失败报告中记录，不增加逐步检查或 GPU 同步。
                current_batch_frames = batch["frames"]
                batch_started = time.perf_counter()
                data_s = batch_started - data_ready
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else None
                if events:
                    events[0].record()
                point, up, targets = on_device(batch, device)
                # 对应官方 Trainer.run_step。一次迭代处理一个真实点云 batch。
                # 当前累计次数为 1，因此每次 batch 前向对应一次 backward 和参数更新。
                optimizer.zero_grad(set_to_none=True)
                # 外层 BF16 autocast 覆盖官方骨干 precision="fp32" 的默认上下文。
                # 依据 PTv3 model._autocast：调用者已有 autocast 时保持调用者精度。
                with autocast(config, device):
                    prediction = model.forward_batch(point, up, batch["max_tokens"])
                # 项目回归目标：cost 与高度各自 masked SmoothL1，归约使用 FP32。
                # 官方通用 Trainer 的 AMP 分支使用 GradScaler。本 BF16 分支按
                # PTv3 docs/acceleration.md 的建议直接 backward，不启用 FP16 缩放。
                loss, parts = batched_perception_loss(prediction, targets, config["loss"])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                optimizer.step()
                # 沿用官方顺序：optimizer 更新之后，再推进一次 scheduler。
                scheduler.step()
                stats += metrics(prediction, targets, loss, parts)
                size = len(batch["frames"])
                progress["global_step"] += 1
                progress["next_offset"] += size
                local_updates += 1
                trained_frames += size
                if events:
                    events[1].record()
                # points 是原始有效回波数，tokens 是体素代表点数。骨干计算规模取决于 tokens。
                row = {"epoch": epoch + 1, "global_step": progress["global_step"], "batch_size": size,
                       "points": sum(f["points"] for f in batch["frames"]),
                       "tokens": sum(f["tokens"] for f in batch["frames"]),
                       "loss_tensor": loss.detach(), "events": events, "data_time_s": data_s,
                       "host_enqueue_time_s": time.perf_counter() - batch_started,
                       "learning_rates": scheduler.get_last_lr()}
                pending.append(row)
                if local_updates <= 2 or progress["global_step"] % config.get("log_every_updates", 20) == 0:
                    flush_updates(window_started)
                    window_started = time.perf_counter()
                del prediction, loss, parts, targets, point, up, norm
                stopped["requested"] |= (output / "STOP").exists() or (max_updates is not None and progress["global_step"] >= max_updates)
                data_ready = time.perf_counter()
                if stopped["requested"]:
                    break
            flush_updates(window_started)
            if progress["next_offset"] >= epoch_frames:
                result = {"epoch": epoch + 1, "train": summary(stats, map_config["cost_scale"]),
                          "train_wall_s": time.perf_counter() - epoch_started}
                improved = False
                if dev_loader and (epoch + 1) % config["eval_every_epochs"] == 0:
                    result["dev"] = evaluate(model, dev_loader, config, device, map_config["cost_scale"])
                    dev = result["dev"]
                    if dev["cost_mae_raw"] is not None and dev["height_mae_m"] is not None:
                        # 项目模型选择以回归误差越小越好，官方语义分割常用 mIoU 越大越好。
                        score = dev["cost_mae_raw"] / 60. + dev["height_mae_m"]
                        improved = progress["best_dev_score"] is None or score < progress["best_dev_score"]
                        if improved:
                            progress["best_dev_score"] = score
                progress["epochs"].append(result)
                progress.update(epoch=epoch + 1, next_offset=0)
                stats.zero_()
                write_json(logs / "epochs.json", progress["epochs"])
                print("PTV3_EPOCH " + json.dumps(result), flush=True)
                if improved or epoch + 1 == config["epochs"] or stopped["requested"] or (epoch + 1) % config.get("save_last_every_epochs", 1) == 0:
                    save(best=improved)
            elif stopped["requested"]:
                save()
            if stopped["requested"]:
                break
        # 首个本地 epoch 可能触发多种形状的编译，稳定吞吐从后续 epoch 汇总。
        first_local_epoch = step_records[0]["epoch"] if step_records else None
        steady = [row for row in step_records if row["epoch"] != first_local_epoch]
        steady_s = sum(row["data_time_s"] + row["batch_time_s"] for row in steady)
        report = {"status": "completed" if progress["epoch"] >= config["epochs"] else "stopped",
                  "format": "ptv3_single_frame_pretraining_v2", "global_step": progress["global_step"],
                  "completed_epochs": progress["epoch"], "local_optimizer_updates": local_updates,
                  "batch_size": batch_size, "num_workers": config["num_workers"], "trained_frames": trained_frames,
                  "elapsed_s": time.perf_counter() - started, "checkpoint_wall_s": checkpoint_s,
                  "checkpoint_timing": checkpoint_timing,
                  "train_start_unix_s": train_start, "first_update_unix_s": first_update_time,
                  "steady_samples_per_s": sum(row["batch_size"] for row in steady) / steady_s if steady_s else None,
                  "steady_data_time_s": sum(row["data_time_s"] for row in steady),
                  "steady_batch_time_s": sum(row["batch_time_s"] for row in steady),
                  "steady_timing_scope": "Subsequent local epochs, excluding the first local epoch, evaluation and checkpoint saves",
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
                  "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  # allocated 是活跃 PyTorch 分配，reserved 还含分配器缓存。
                  # 两者均不包含桌面等其他进程，不能与整卡 nvidia-smi 显存直接混用。
                  "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                  "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None,
                  "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "hardware_limits_imposed": False, "initialization": initialization,
                  "training_cache": str(cache), "independent_dev_available": dev_loader is not None,
                  "test_evaluated": False, "isaac_started": False}
        write_json(logs / "report.json", report)
        write_json(logs / "status.json", report)
        return report
    except BaseException as exc:
        write_json(logs / "status.json", {"status": "failed", "error": str(exc),
                    "global_step": progress["global_step"], "batch_frames": current_batch_frames})
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
