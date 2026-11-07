"""Model/optimizer checkpoint loading and random-state restoration."""
import torch
from ptv3_tomography.training.checkpoint import restore_rng, rng_state


def capture_rng(device):
    state = rng_state()
    state["cuda"] = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    return state


def restore_training_rng(state, device):
    restore_rng(state)
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])


def read_pretraining_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)
