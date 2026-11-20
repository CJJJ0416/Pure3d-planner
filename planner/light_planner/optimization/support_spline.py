"""Planar quintic tracking spline with an independent body height fit."""
from dataclasses import dataclass
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree

_MIN_EDGE_M = 1e-5


@dataclass(frozen=True)
class TerrainLimits:
    body_height_m: float = .4
    min_body_clearance_m: float = .10
    max_body_reference_lift_m: float = .15
    reference_spacing_m: float = .035
    max_xy_mps: float = .75
    max_acc_mps2: float = .5
    max_jerk_mps3: float = 1.2
    max_z_jerk_mps3: float = .25
    max_lateral_error_m: float = .08


@dataclass(frozen=True)
class SplineState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray | None = None


@dataclass(frozen=True)
class SupportSpline:
    curve: BSpline
    profile: dict

    @property
    def duration(self):
        return float(self.curve.t[-self.curve.k-1])

    def state(self, t):
        t = np.clip(t, 0., self.duration)
        return SplineState(self.curve(t), self.curve(t, 1), self.curve(t, 2), self.curve(t, 3))


class InflatedOccupancy:
    """Query the published occupied voxel centers at the two swept body axes."""

    def __init__(self, xyz, resolution, offset):
        self.tree = cKDTree(xyz)
        self.resolution = resolution
        self.half_cell = resolution / 2 + 1e-5
        self.offset = offset

    def blocked(self, positions, tangents):
        tangents = tangents[..., :2]
        length = np.linalg.norm(tangents, axis=-1, keepdims=True)
        direction = tangents / np.maximum(length, 1e-9)
        axes = np.stack((positions, positions), axis=-2)
        axes[..., 0, :2] += self.offset * direction
        axes[..., 1, :2] -= self.offset * direction
        distances, _ = self.tree.query(axes, p=np.inf, distance_upper_bound=self.half_cell)
        return np.any(np.isfinite(distances), axis=-1)

    def blocked_edges(self, graph, xyz, body_height):
        row = np.repeat(np.arange(len(xyz)), np.diff(graph.indptr))
        start = xyz[row] + [0., 0., body_height]
        tangent = xyz[graph.indices] - xyz[row]
        fractions = np.linspace(0., 1., 5)
        samples = start[:, None, :] + tangent[:, None, :] * fractions[None, :, None]
        return self.blocked(samples, tangent[:, None, :]).any(axis=1)

    def height_bounds(self, xy, tangents, desired, lower, upper):
        resolution = self.resolution
        count = int(np.ceil(np.max(upper-lower)/resolution))+2
        indices = np.floor(lower/resolution)[:, None] + np.arange(count)
        heights = (indices+.5)*resolution
        samples = np.concatenate((np.broadcast_to(xy[:, None, :], (*heights.shape, 2)), heights[..., None]), axis=2)
        blocked = self.blocked(samples, tangents[:, None, :])
        blocked &= (heights+self.half_cell >= lower[:, None]) & (heights-self.half_cell <= upper[:, None])
        floor = np.max(np.where(blocked & (heights < desired[:, None]), heights+self.half_cell+.005, -np.inf), axis=1)
        ceiling = np.min(np.where(blocked & (heights >= desired[:, None]), heights-self.half_cell-.005, np.inf), axis=1)
        return np.maximum(lower, floor), np.minimum(upper, ceiling)


def arc_path(path):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Expected a finite support path of shape N×3")
    # A rolling arc cut may land almost on an existing vertex.
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) >= _MIN_EDGE_M]
    points = points[keep]
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not len(distances):
        raise ValueError("Zero-length support path")
    return points, np.r_[0., np.cumsum(distances)]


def sample_path(points, arc, distances):
    return np.stack([np.interp(distances, arc, points[:, axis]) for axis in range(3)], axis=-1)


def project_path(points, arc, query, *, horizontal=False):
    """Project samples onto the ordered selected support polyline."""
    dimensions = 2 if horizontal else 3
    xyz = np.asarray(query)[..., :dimensions]
    delta = np.diff(points[:, :dimensions], axis=0)
    relative = xyz[..., None, :] - points[:-1, :dimensions]
    alpha = np.clip(np.sum(relative*delta, axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-16), 0., 1.)
    index = np.argmin(np.sum((relative-alpha[..., None]*delta)**2, axis=-1), axis=-1)
    fraction = np.take_along_axis(alpha, index[..., None], axis=-1)[..., 0]
    return arc[index] + fraction*np.diff(arc)[index]


def slice_path(path, start_m, end_m):
    """Arc slice with exact cut points and all original vertices in between."""
    points, arc = arc_path(path)
    start_m = float(np.clip(start_m, 0., arc[-1]))
    end_m = float(np.clip(end_m, start_m, arc[-1]))
    if end_m - start_m < .05:
        raise ValueError("Support slice shorter than 5 cm")
    distances = np.r_[start_m, arc[(arc > start_m + 1e-8) & (arc < end_m - 1e-8)], end_m]
    return sample_path(points, arc, distances)


def _support_samples(points, arc, limits):
    horizontal = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    rise = np.abs(np.diff(points[:, 2]))
    if np.any(horizontal < _MIN_EDGE_M):
        raise ValueError("Vertical edge cannot be executed by a yaw-only tracker")
    grade = rise / horizontal
    planar_arc = np.r_[0., np.cumsum(horizontal)]
    intervals = max(6, int(np.ceil(planar_arc[-1] / limits.reference_spacing_m)))
    regular = np.interp(np.linspace(0., planar_arc[-1], intervals+1), planar_arc, arc)
    samples = np.unique(np.r_[arc, regular])
    return samples, grade


def _reference_time(distances, cap, start_speed, stop, acceleration):
    speed = cap.copy()
    speed[0] = start_speed
    if stop:
        speed[-1] = 0.
    for i, distance in enumerate(np.diff(distances), 1):
        speed[i] = min(speed[i], np.sqrt(speed[i-1]**2 + 2*acceleration*distance))
    for i in range(len(speed)-2, 0, -1):
        speed[i] = min(speed[i], np.sqrt(speed[i+1]**2 + 2*acceleration*(distances[i+1]-distances[i])))
    return np.r_[0., np.cumsum(2*np.diff(distances) / np.maximum(speed[:-1]+speed[1:], 1e-6))]


def _spline_system(times, target, start, stop, spans):
    duration = times[-1]
    degree = 5
    knots = np.r_[np.zeros(degree), np.linspace(0., 1., spans+1), np.ones(degree)]
    basis = BSpline(knots, np.eye(spans+degree), degree)
    equal = [basis(0.), basis(0., 1), basis(0., 2), basis(1.)]
    boundary = [start.position-target[0], start.velocity*duration,
                start.acceleration*duration**2, target[-1]-target[0]]
    equal.append(basis(0., 3))
    boundary.append(np.zeros(3) if start.jerk is None else start.jerk*duration**3)
    if stop:
        equal.extend([basis(1., 1), basis(1., 2), basis(1., 3)])
        boundary.extend([np.zeros(3)] * 3)
    equal = np.asarray(equal)
    particular = np.linalg.lstsq(equal, boundary, rcond=None)[0] + target[0]
    return basis, particular, null_space(equal)


def _fit_xy(basis, particular, free, times, target, weights):
    duration = times[-1]
    spans = len(basis.c)-basis.k
    samples = (np.arange(3*spans)+.5)/(3*spans)
    fit = basis(times/duration)*weights[:, None]
    jerk = .14*basis(samples, 3)/(np.sqrt(3)*spans**3)
    matrix = np.vstack((fit, jerk))
    value = np.vstack((target*weights[:, None], np.zeros((len(jerk), 2))))
    coefficients = particular + free @ np.linalg.lstsq(matrix @ free, value-matrix @ particular, rcond=None)[0]
    return BSpline(basis.t*duration, coefficients, basis.k, extrapolate=False)


def _fit_height(basis, particular, free, times, target, support, xy, limits, occupancy):
    duration = times[-1]
    spans = len(basis.c)-basis.k
    t = np.unique(np.r_[times, np.linspace(0., duration, max(21, int(np.ceil(duration/.025))+1)),
                        np.linspace(0., duration, spans+1)])
    u = t/duration
    ground = np.interp(t, times, support)
    lower = ground + limits.min_body_clearance_m
    upper = ground + limits.body_height_m + limits.max_body_reference_lift_m
    desired = np.interp(t, times, target)
    if occupancy is not None and occupancy.tree.n:
        tangents = xy(np.minimum(t+.04, duration)) - xy(np.maximum(t-.04, 0.))
        lower, upper = occupancy.height_bounds(xy(t), tangents, desired, lower, upper)
    if np.any(lower > upper):
        raise ValueError('No body height clearance along the planar trajectory')

    # Only the height profile uses these limits; XY keeps its own time schedule.
    jerk = np.full(len(t), .995*limits.max_z_jerk_mps3)
    jerk[[0,-1]] = limits.max_z_jerk_mps3
    constraints = np.vstack((basis(u), basis(u, 3)/duration**3))
    lo = np.r_[lower, -jerk] - constraints @ particular
    hi = np.r_[upper, jerk] - constraints @ particular
    constraints = constraints @ free

    height_fit = basis(times/duration)
    jerk_fit = .35*basis((np.arange(3*spans)+.5)/(3*spans), 3)/(np.sqrt(3)*spans**3)
    fit = np.vstack((height_fit, jerk_fit))
    demand = np.r_[target, np.zeros(len(jerk_fit))] - fit @ particular
    fit = fit @ free
    estimate = np.linalg.lstsq(fit, demand, rcond=None)[0]
    tolerance = 2e-6

    def feasible(coefficients):
        values = constraints @ coefficients
        return np.all(values >= lo-tolerance) and np.all(values <= hi+tolerance)

    if not feasible(estimate):
        def objective(coefficients):
            residual = fit @ coefficients - demand
            return .5*float(residual @ residual), fit.T @ residual

        variable = np.linalg.norm(constraints, axis=1) > 1e-10
        if np.any(lo[~variable] > tolerance) or np.any(hi[~variable] < -tolerance):
            raise ValueError('Fixed handoff state violates the body height corridor or bounds')
        solve = minimize(objective, estimate, jac=True, method='SLSQP',
                         constraints=[LinearConstraint(constraints[variable], lo[variable], hi[variable])],
                         options={'maxiter': 40, 'ftol': 1e-9})
        if not feasible(solve.x):
            raise ValueError('Body height infeasible on the planar time schedule')
        estimate = solve.x
    return particular + free @ estimate


def fit_support_spline(path, limits=TerrainLimits(), *, start=None, stop=True, occupancy=None):
    """Fit a quintic support spline with continuous jerk and fixed handoff derivatives."""
    began = time.perf_counter()
    points, arc = arc_path(path)
    distance, grade = _support_samples(points, arc, limits)
    support_target = sample_path(points, arc, distance)
    target = support_target + [0., 0., limits.body_height_m]
    xy_steps = np.linalg.norm(np.diff(target[:, :2], axis=0), axis=1)
    xy_distance = np.r_[0., np.cumsum(xy_steps)]
    xy_cap = np.full(len(distance), limits.max_xy_mps)
    if start is None:
        start = SplineState(target[0], np.zeros(3), np.zeros(3))
    if not all(np.shape(x) == (3,) and np.isfinite(x).all()
               for x in (start.position, start.velocity, start.acceleration)):
        raise ValueError("Invalid starting PVA")
    jerk = None if start.jerk is None else np.asarray(start.jerk, dtype=float)
    if jerk is not None and (jerk.shape != (3,) or not np.isfinite(jerk).all()):
        raise ValueError('Invalid starting jerk')
    start = SplineState(np.asarray(start.position), np.asarray(start.velocity), np.asarray(start.acceleration), jerk)
    nominal_time = _reference_time(xy_distance, .80*xy_cap, np.linalg.norm(start.velocity[:2]),
                                   stop, .65*limits.max_acc_mps2)
    spans = max(5, int(np.ceil(xy_distance[-1]/.17)))
    scale = 1.
    weights = np.ones(len(target))
    collision_refits = 0
    for _ in range(8):
        times = nominal_time*scale
        if times[-1] > 60.:
            break
        t = np.unique(np.r_[np.linspace(0., times[-1], max(151, int(np.ceil(times[-1]/.01))+1)),
                            np.linspace(0., times[-1], spans+1)])
        basis, particular, free = _spline_system(times, target, start, stop, spans)
        xy = _fit_xy(basis, particular[:, :2], free, times, target[:, :2], weights)
        reference_distance = np.interp(t, times, distance)
        reference = sample_path(target, distance, reference_distance)
        xy_speed = np.linalg.norm(xy(t, 1), axis=1)
        actual_distance = project_path(points, arc, xy(t), horizontal=True)
        allowed = np.minimum(np.interp(reference_distance, distance, xy_cap),
                             np.interp(actual_distance, distance, xy_cap))
        xy_ratios = (float(np.max(xy_speed/allowed)),
                     float(np.max(np.linalg.norm(xy(t, 2), axis=1))/limits.max_acc_mps2),
                     float(np.max(np.linalg.norm(xy(t, 3), axis=1))/limits.max_jerk_mps3))
        if max(xy_ratios) > 1.0001:
            scale *= 1.03*max(xy_ratios[0], np.sqrt(xy_ratios[1]), np.cbrt(xy_ratios[2]))
            continue
        xy_error = float(np.max(np.linalg.norm(xy(t)-reference[:, :2], axis=1)))
        if xy_error > limits.max_lateral_error_m:
            raise ValueError(f'Planar spline leaves reference band (XY {xy_error:.3f} m)')
        height = _fit_height(basis, particular[:, 2], free, times, target[:, 2], support_target[:, 2], xy,
                             limits, occupancy)
        curve = BSpline(xy.t, np.column_stack((xy.c, height)), xy.k, extrapolate=False)
        xyz = curve(t)
        if occupancy is not None:
            tangents = curve(np.minimum(t+.04, t[-1])) - curve(np.maximum(t-.04, 0.))
            blocked = occupancy.blocked(xyz, tangents)
            if blocked.any():
                if collision_refits == 2:
                    raise ValueError('Spline remains occupied after planar refit')
                affected = np.clip(np.searchsorted(times, t[blocked]), 1, len(times)-1)
                weights[np.r_[affected-1, affected]] *= 4.
                collision_refits += 1
                continue

        ground = np.interp(t, times, support_target[:, 2])
        if (np.any(xyz[:, 2] < ground+limits.min_body_clearance_m-.002) or
                np.any(xyz[:, 2] > ground+limits.body_height_m+limits.max_body_reference_lift_m+.002)):
            raise ValueError('Height spline leaves supported clearance corridor between constraints')
        jerk_values = curve(t, 3)
        peak_jerk = float(np.max(np.linalg.norm(jerk_values, axis=1)))
        peak_z_jerk = float(np.max(np.abs(jerk_values[:, 2])))
        if peak_z_jerk > 1.003*limits.max_z_jerk_mps3:
            raise ValueError('Body height exceeds derivative bounds between constraints')
        body_reference = curve(times)
        return SupportSpline(curve, dict(
            reference_body_xyz=np.column_stack((target[:, :2], body_reference[:, 2])),
            reference_support_xyz=support_target,
            profile_t_s=times, profile_arc_m=distance, profile_xy_cap_mps=xy_cap,
            command_cap_t_s=t, command_cap_xy_mps=allowed,
            max_grade=float(grade.max()), min_xy_cap_mps=float(xy_cap.min()),
            max_reference_lift_m=float(np.max(body_reference[:, 2]-support_target[:, 2])-limits.body_height_m),
            fit_wall_ms=(time.perf_counter()-began)*1000, time_scale=scale, collision_refits=collision_refits,
            max_xy_tracking_error_m=xy_error, max_z_tracking_error_m=float(np.max(np.abs(xyz[:, 2]-reference[:, 2]))),
            max_speed_ratio=xy_ratios[0], max_acc_ratio=xy_ratios[1],
            peak_jerk_mps3=peak_jerk, peak_z_jerk_mps3=peak_z_jerk))
    raise ValueError('Planar spline cannot meet speed, acceleration or jerk limits at the handoff')


def scan_bspline_message(plan, stamp, trajectory_id):
    """Serialize the spline degree and coefficients for the local trajectory server."""
    from geometry_msgs.msg import Point
    from scan_planner_msgs.msg import Bspline

    return Bspline(order=plan.curve.k, start_time=stamp, traj_id=trajectory_id,
                   pos_pts=[Point(x=float(x), y=float(y), z=float(z)) for x, y, z in plan.curve.c],
                   knots=plan.curve.t.tolist())
