"""Load prediction parameters separately from checkpoint model and training settings."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "utils/Traversability/configs/predict.json"
CONDITION_NAMES = ("clearance_min", "slope_max", "step_max", "stand_ratio")
RUNTIME_KEYS = ("device", "cpu_threads", "geometry_python", "geometry_voxel_size_m", "window")


def validate(settings):
    """Validate prediction-only fields and normalize named conditions to model order."""
    if not isinstance(settings, dict):
        raise ValueError("Prediction config must be a JSON object")
    unknown = settings.keys() - {"checkpoint", "condition", *RUNTIME_KEYS}
    if unknown:
        raise ValueError("Unsupported prediction parameters: " + ", ".join(sorted(unknown)))
    settings = deepcopy(settings)

    def number(value, name, minimum=None, strict=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if minimum is not None and (value <= minimum if strict else value < minimum):
            raise ValueError(f"Invalid {name}: {value}")

    for name in ("checkpoint", "device", "geometry_python"):
        if name in settings and (not isinstance(settings[name], str) or not settings[name].strip()):
            raise ValueError(f"{name} must be a non-empty string")
    if "condition" in settings:
        values = settings["condition"]
        if isinstance(values, dict):
            if values.keys() != set(CONDITION_NAMES):
                raise ValueError("condition requires " + ", ".join(CONDITION_NAMES))
            values = [values[name] for name in CONDITION_NAMES]
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("condition requires four values")
        for name, value in zip(CONDITION_NAMES, values):
            number(value, name, 0)
        settings["condition"] = values
    if "cpu_threads" in settings:
        value = settings["cpu_threads"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("cpu_threads must be a positive integer")
    if "geometry_voxel_size_m" in settings:
        number(settings["geometry_voxel_size_m"], "geometry_voxel_size_m", 0, strict=True)
    if "window" in settings:
        window = settings["window"]
        if not isinstance(window, dict) or window.keys() - {"core_size_m", "context_m"}:
            raise ValueError("window supports only core_size_m and context_m")
        if "core_size_m" in window:
            size = window["core_size_m"]
            if not isinstance(size, list) or len(size) != 3:
                raise ValueError("core_size_m requires three values")
            for value in size:
                number(value, "core_size_m", 0, strict=True)
        if "context_m" in window:
            number(window["context_m"], "context_m", 0)
    return settings


def load_prediction_config(path=DEFAULT_CONFIG):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_prediction_config(checkpoint_config, settings=None):
    """Override runtime fields without mutating checkpoint state or model structure."""
    config = deepcopy(checkpoint_config)
    for key, value in validate(settings or {}).items():
        if key == "window":
            config["window"].update(value)
        elif key in RUNTIME_KEYS:
            config[key] = value
    return config


def prediction_parameters(config, checkpoint, condition):
    return dict(
        checkpoint=str((ROOT / checkpoint).resolve()),
        condition=dict(zip(CONDITION_NAMES, condition)),
        **{key: deepcopy(config[key]) for key in RUNTIME_KEYS if key in config},
    )


def parse_prediction_args(description, argv=None):
    """Resolve JSON defaults, then apply explicit checkpoint/condition/device flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="预测参数 JSON")
    parser.add_argument("--checkpoint", type=Path, help="覆盖预测配置中的权重")
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--condition", type=float, nargs=4, help="覆盖四维机器人条件")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", help="覆盖预测配置中的设备")
    args = parser.parse_args(argv)
    try:
        settings = load_prediction_config(args.config)
        for name in ("checkpoint", "condition", "device"):
            value = getattr(args, name)
            if value is not None:
                settings[name] = str(value) if name == "checkpoint" else value
        settings = validate(settings)
        if "checkpoint" not in settings or "condition" not in settings:
            raise ValueError("Provide checkpoint and condition in JSON or command-line flags")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.checkpoint = (ROOT / settings["checkpoint"]).resolve()
    args.condition = settings["condition"]
    args.device = settings.get("device")
    args.inference_config = settings
    return args
