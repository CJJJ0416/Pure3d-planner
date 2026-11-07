"""Workspace paths and runtime resolution of saved pre-relocation asset paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
PTV3_ROOT = WORKSPACE_ROOT / "third_party/PointTransformerV3"
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs/ptv3_tomography"
RUN_ROOT = WORKSPACE_ROOT / "runs"


def training_run(value):
    """Keep training products under runs, including commands with old output paths."""
    path = Path(value).expanduser()
    absolute = path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()
    return absolute if absolute.is_relative_to(RUN_ROOT) else RUN_ROOT / path.name


def relocated_path(value):
    """Resolve saved legacy absolute paths without rewriting datasets or checkpoints."""
    path = Path(value).expanduser()
    if not path.is_absolute() or path.exists():
        return path
    parts = path.parts
    old_task = ("src", "PointTransformerV3", "projects", "tomography")
    old_core = ("src", "PointTransformerV3")
    for index in range(len(parts)):
        if tuple(parts[index:index + len(old_task)]) == old_task:
            suffix = parts[index + len(old_task):]
            if suffix[:1] == ("outputs",):
                return OUTPUT_ROOT.joinpath(*suffix[1:])
            return PROJECT_ROOT.joinpath(*suffix)
        if tuple(parts[index:index + len(old_core)]) == old_core:
            return PTV3_ROOT.joinpath(*parts[index + len(old_core):])
    return path


def asset_path(directory, value):
    path = Path(value)
    candidate = path if path.is_absolute() else Path(directory) / path
    return relocated_path(candidate).resolve()
