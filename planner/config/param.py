"""正式DC方案及其无跨阶边消融的共同配置。"""
from dataclasses import dataclass


@dataclass
class PlannerConfig:
    voxel_resolution: float = .1
    cost_threshold: float = .5
    cost_weight: float = 5.
    step_edges: bool = True
    clearance_min: float = .5
    slope_max: float = .4
    step_max: float = .3
    stand_ratio: float = .1
    step_horizontal_radius: float = .3
    minimum_step_height: float = .05
    interval_free: float = .5  # 教师固定净空，仅约束新增跨阶边，非FiLM条件
    search_normal_offset: float = .1
    gravity_up: tuple = (0., 0., 1.)
    roi_min: tuple | None = None
    roi_max: tuple | None = None
    reference_speed: float = .5
    max_speed: float = .8
    gp_qc: float = .2
    gp_cost_weight: float = 100.
    gp_speed_weight: float = 20.
    field_neighbors: int = 32
    field_sigma: float = .1
    endpoint_offset_m: float = .5
    interpolation_points: int = 3
    max_iterations: int = 100
    horizontal_corridor_m: float = .2
    height_smoothing_m: float = .4
    height_tolerance_m: float = .1
