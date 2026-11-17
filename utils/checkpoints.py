"""Store model files under weights and expose links in training run folders."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_destination(path, module):
    path = Path(os.path.abspath(path))
    runs = ROOT / 'runs'
    if not path.is_relative_to(runs):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    relative = path.relative_to(runs)
    if relative.parts[0] != module:
        relative = Path(module) / relative
    destination = ROOT / 'weights' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.symlink_to(os.path.relpath(destination, path.parent))
    return destination
