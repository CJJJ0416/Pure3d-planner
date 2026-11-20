"""Candidate-set model for landing area and route-independent goal utility."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from planner.learning_planner.neupan.geometry.landing import HEIGHT_TOLERANCE, MIN_AREA, observed_candidates, support_patches
from planner.learning_planner.neupan.data.landing_approach import approach_distances
from planner.learning_planner.neupan.data.landing_dataset import DISK_AREA, FEATURE_DIM, GEOMETRIC_AREA_METRIC, HORIZON, features
from planner.learning_planner.neupan.inference.landing_route import MIN_COST_PROGRESS, LandingRoutes


class LandingModel(nn.Module):
    def __init__(self, coverage_head=False, geometric_area=False):
        super().__init__()
        self.coverage_head = coverage_head
        self.geometric_area = geometric_area
        self.encode = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 3 if coverage_head else 2))

    def forward(self, values, observed):
        encoded = self.encode(values)
        masked = encoded * observed[..., None]
        context = masked.sum(1) / observed.sum(1, keepdim=True).clamp_min(1)
        return self.score(torch.cat((encoded, context[:, None].expand_as(encoded)), dim=-1))


@dataclass
class GoalCandidate:
    xyz: np.ndarray
    observed_area_m2: float
    estimated_area_m2: float
    score: float
    approach_m: float | None = None
    approach_cost: float | None = None
    remaining_cost: float | None = None
    cost_progress: float | None = None
    observed_goal: bool | None = None
    approach_xyz: np.ndarray | None = None
    trav_cost: float | None = None


def candidate_mask(features_array, approach_m=None, support_area=None):
    area = features_array[:, 9] * DISK_AREA if support_area is None else support_area
    mask = (features_array[:, 11] > .5) & (area >= MIN_AREA - 1e-7)
    if approach_m is not None:
        mask &= approach_m <= HORIZON + 1e-5
    return mask


def select_points(points, features_array, scores, score_threshold, count=3, separation=.30,
                  approach_m=None, support_area=None):
    mask = candidate_mask(features_array, approach_m, support_area)
    order = np.flatnonzero(mask & (scores >= score_threshold))
    order = order[np.argsort(-scores[order], kind='stable')]
    chosen = []
    for i in order:
        if all(np.linalg.norm(points[i] - points[j]) >= separation for j in chosen):
            chosen.append(int(i))
            if len(chosen) >= count:
                break
    return chosen


def center_on_support(tomo, points, support_area, rank, goal_cells, count, radius=.20, separation=.30):
    """Within each route proposal's connected surface, maximize footprint support."""
    remaining = np.isfinite(rank)
    ij = np.rint((points[:, :2] - tomo.origin) / tomo.resolution).astype(int)
    stride = tomo.ground.shape[2]
    keys = ij[:, 0] * stride + ij[:, 1]
    chosen = []
    while remaining.any() and len(chosen) < count:
        seed = np.argmax(np.where(remaining, rank, -np.inf))
        near = remaining & (np.linalg.norm(points[:, :2] - points[seed, :2], axis=1) <= radius + 1e-6)
        near &= np.abs(points[:, 2] - points[seed, 2]) <= HEIGHT_TOLERANCE
        if goal_cells[seed]:
            near &= goal_cells
        ids = np.flatnonzero(near)
        if len(ids) > 1:
            patch = support_patches(tomo, points[seed:seed + 1], radius=radius, include_cells=True)
            cells = patch.support_cells[0]
            ids = ids[np.isin(keys[ids], cells[:, 0] * stride + cells[:, 1])]
        i = ids[np.lexsort((rank[ids], np.round(support_area[ids], 6)))[-1]] if len(ids) else seed
        chosen.append(int(i))
        remaining &= np.linalg.norm(points - points[i], axis=1) >= separation
    return chosen


class LandingScorer:
    def __init__(self, checkpoint, *, route_cost=True, tread_center=True):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('type') not in ('landing_goal_v2', 'landing_goal_v3'):
            raise ValueError('Expected a landing-area and goal-utility checkpoint')
        self.model = LandingModel(coverage_head=state['type'] == 'landing_goal_v3',
                                  geometric_area=state.get('area_metric') == GEOMETRIC_AREA_METRIC).eval()
        self.model.load_state_dict(state['model'])
        self.threshold = float(state['threshold'])
        self.route_cost = route_cost and self.model.coverage_head
        self.tread_center = tread_center and self.route_cost

    @torch.inference_mode()
    def score(self, values):
        x = torch.from_numpy(values)
        return self.model(x[None], x[None, :, 11] > .5)[0].sigmoid().numpy()

    def candidates(self, tomo, start, goal, limit=3, *, visited=()):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
            raise ValueError('Expected finite start support and goal XYZ')
        if limit <= 0:
            raise ValueError('Expected a positive candidate limit')
        from planner.learning_planner.neupan.inference.reachability import goal_mask
        if goal_mask(start, goal):
            return [], 'goal_reached'
        points = observed_candidates(tomo, start, HORIZON)
        if len(points) == 0:
            return [], 'no_observed_support'
        x, patches = features(tomo, start, goal, points, geometric_area=self.model.geometric_area)
        area = patches.support_area if self.tread_center or self.model.geometric_area else patches.area
        if not (area >= MIN_AREA).any():
            return [], 'insufficient_support_area'
        if self.route_cost:
            return self.route_candidates(tomo, start, goal, visited, points, x, area, limit)
        approach_m = approach_distances(tomo, start, points) if self.model.coverage_head else None
        if not candidate_mask(x, approach_m).any():
            return [], 'support_blocked'
        predictions = self.score(x)
        ids = select_points(points, x, predictions[:, 1], self.threshold, count=limit,
                            approach_m=approach_m, support_area=area)
        candidates = [GoalCandidate(points[i], float(patches.area[i]),
                                    float(predictions[i, 0] * DISK_AREA), float(predictions[i, 1]),
                                    float(approach_m[i]) if approach_m is not None else None,
                                    trav_cost=float(x[i, 10]))
                      for i in ids]
        return candidates, 'candidates_ready' if candidates else 'low_goal_score'

    def route_candidates(self, tomo, start, goal, visited, points, x, area, limit):
        route = LandingRoutes(tomo).costs(start, goal, points)
        if route is None:
            return [], 'support_blocked'
        can_land = candidate_mask(x, route.approach_m, area)
        if not can_land.any():
            return [], 'support_blocked'
        progress = route.progress
        eligible = can_land & (progress >= MIN_COST_PROGRESS)
        for prior in visited:
            close_xy = np.linalg.norm(points[:, :2] - np.asarray(prior)[:2], axis=1) < .10
            close_z = np.abs(points[:, 2] - prior[2]) < .08
            eligible &= ~(close_xy & close_z)
        if not eligible.any():
            return [], 'no_goal_progress'
        predictions = self.score(x)
        if not route.observed_goal:
            eligible &= predictions[:, 1] >= self.threshold
        if not eligible.any():
            return [], 'low_goal_score'
        rank = np.full(len(points), np.nan)
        excess = np.maximum(0., route.approach_cost[eligible] - progress[eligible])
        rank[eligible] = progress[eligible] - excess + .02 * predictions[eligible, 1]
        if self.tread_center:
            goals = route.observed_goal & (route.remaining_cost == 0.)
            ids = center_on_support(tomo, points, area, rank, goals, limit)
        else:
            ids = select_points(points, x, rank, -np.inf, count=limit,
                                approach_m=route.approach_m, support_area=area)
        candidates = [GoalCandidate(points[i], float(area[i]), float(predictions[i, 0] * DISK_AREA),
                                    float(predictions[i, 1]), float(route.approach_m[i]),
                                    float(route.approach_cost[i]), float(route.remaining_cost[i]),
                                    float(progress[i]), route.observed_goal, route.approach_path(i),
                                    trav_cost=float(x[i, 10])) for i in ids]
        return candidates, 'candidates_ready'
