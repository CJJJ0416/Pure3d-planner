"""Seeded physical geometry parameters shared by scene generation and campaigns."""
from copy import deepcopy
import numpy as np


NOMINAL = dict(
    west_deck_size_m=[5.,7.2], east_deck_size_m=[5.5,5.5],
    ramp_landing_size_m=[3.,3.], open_stair_landing_size_m=[3.,2.5],
    open_stair_depth_m=.425, open_stair_width_m=1.8,
    switchback_depth_m=[.3,.34], switchback_width_m=1.3,
    switchback_landing_depth_m=1.2, switchback_top_size_m=[1.5,2.],
    spiral_outer_radius_m=2.65, spiral_connector_size_m=[2.85,3.],
    solid_stair_steps=12, solid_stair_depth_m=.4, solid_stair_width_m=1.2,
    west_ramp_width_m=1.8, east_ramp_width_m=1.7,
    open_stair_bend_degrees=0., open_stair_width_ratios=[1.,1.,1.],
    switchback_width_ratios=[1.,1.,1.],
    platform_opening_mode="solid", platform_hole_size_fraction=[.4,.4],
    platform_hole_offset_fraction=[0.,0.],
    obstacle_shapes=["box","cylinder"],
    obstacle_size_range_m=[[.44,1.0],[.4,.9],[.35,1.25]],
)


def sample_geometry(config):
    cfg = dict(deepcopy(NOMINAL), **deepcopy(config))
    ranges = cfg.get("geometry_randomization")
    if not ranges or not ranges["enabled"]:
        return cfg
    if cfg.get("geometry_randomization_seed") == cfg["seed"]:
        return cfg
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]),75103]))
    for key, bounds in ranges["uniform"].items():
        bounds = np.asarray(bounds,float)
        sample = np.asarray(rng.uniform(bounds[...,0],bounds[...,1]))
        cfg[key] = float(sample) if sample.ndim == 0 else sample.tolist()
    for key, (lo,hi) in ranges["integer"].items():
        cfg[key] = int(rng.integers(lo,hi+1))
    for key,values in ranges.get("choice",{}).items():
        cfg[key] = str(rng.choice(values))
    cfg["geometry_randomization_seed"] = int(cfg["seed"])
    return cfg
