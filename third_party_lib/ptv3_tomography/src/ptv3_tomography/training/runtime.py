"""Random seeds, device/backend setup and mixed precision."""
from contextlib import nullcontext
import random

import numpy as np
import torch
from point_transformer_v3 import set_backend


def configure(config):
    device = torch.device(config["device"])
    set_backend(**config["backends"])
    torch.use_deterministic_algorithms(config["deterministic"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    return device


def autocast(config, device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and config["amp"] == "bf16" else nullcontext()
