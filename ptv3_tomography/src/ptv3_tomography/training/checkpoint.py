"""CPU random-state serialization for resumable training."""
import random
import numpy as np
import torch


def rng_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": {"name": name, "keys": torch.from_numpy(keys.astype(np.int64)),
                      "position": position, "has_gauss": has_gauss, "cached": cached}}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s["name"], s["keys"].numpy().astype(np.uint32), s["position"], s["has_gauss"], s["cached"]))
