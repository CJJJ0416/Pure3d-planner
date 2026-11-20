"""EGO cubic B-spline parameterization, analytic derivatives and yaw tracking."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import diags, lil_matrix, vstack
from scipy.sparse.linalg import spsolve


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class ExpertSpline:
    control: np.ndarray
    knots: np.ndarray

    @property
    def duration(self):
        return float(self.knots[-4] - self.knots[3])

    def evaluate(self, seconds, derivative=0):
        t = np.clip(seconds, 0., self.duration) + self.knots[3]
        return BSpline(self.knots, self.control, 3)(t, nu=derivative)

    def heading(self, seconds):
        """EGO uses the tangent in motion and holds the last valid yaw at rest."""
        t = np.asarray(seconds)
        sample = np.r_[np.linspace(0, self.duration, int(np.ceil(self.duration / .02)) + 1),
                       t.ravel()]
        sample.sort()
        xy = self.evaluate(sample, 1)[:, :2]
        valid = np.linalg.norm(xy, axis=1) > .1
        if not valid.any():
            raise ValueError('Reference does not have a valid horizontal tangent')
        yaw = np.unwrap(np.arctan2(xy[valid, 1], xy[valid, 0]))
        ids = np.searchsorted(sample[valid], np.clip(t, 0, self.duration), side='right') - 1
        return yaw[np.maximum(ids, 0)]

    def command(self, phase, position, yaw, rate=1.):
        if phase >= self.duration or rate == 0:
            return np.zeros(3)
        goal = self.evaluate(self.duration)
        reference_yaw = self.heading(self.duration if np.linalg.norm(goal - position) < 1. else phase)
        yaw_error = wrap(reference_yaw - yaw)
        xy = rate * self.evaluate(phase, 1)[:2] @ rotation(yaw)
        if abs(yaw_error) >= .785:
            xy[:] = 0.
        return np.r_[xy, np.clip(1.5 * yaw_error, -1., 1.)]


def parameterize(points, dt, derivatives=None):
    """EGO's position, endpoint velocity and endpoint acceleration equations."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError('At least four XYZ samples are required')
    boundary = np.zeros((4, 3)) if derivatives is None else np.asarray(derivatives, dtype=float)
    if (boundary.shape != (4, 3) or not np.isfinite(points).all()
            or not np.isfinite(boundary).all() or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid EGO spline samples, duration or endpoint derivatives')
    n = len(points)
    position = diags((1/6, 4/6, 1/6), (0, 1, 2), shape=(n, n+2))
    constraints = lil_matrix((4, n+2))
    constraints[0, :3] = np.array([-1, 0, 1]) / (2*dt)
    constraints[1, -3:] = np.array([-1, 0, 1]) / (2*dt)
    constraints[2, :3] = np.array([1, -2, 1]) / dt**2
    constraints[3, -3:] = np.array([1, -2, 1]) / dt**2
    matrix = vstack((position, constraints), format='csc')
    control = spsolve((matrix.T @ matrix).tocsc(), matrix.T @ np.vstack((points, boundary)))
    return ExpertSpline(control, np.arange(-3, n+3, dtype=float)*dt)


def fit_route(xyz, dt=.1, max_speed=.9, max_accel=.5, max_yaw_rate=1.):
    """Start from GPMP geometry and scale the time knots to meet A1 limits."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or len(xyz) < 2:
        raise ValueError('Invalid GPMP route')
    travel = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    keep = np.r_[True, travel > 1e-6]
    arc, xyz = np.r_[0., np.cumsum(travel)][keep], xyz[keep]
    if arc[-1] < .05:
        raise ValueError('Degenerate GPMP route')
    times = np.arange(max(3, int(np.ceil(arc[-1]/max_speed/dt)))+1) * dt
    u = times / times[-1]
    distance = arc[-1]*u**3*(10-15*u+6*u*u)
    points = np.column_stack([np.interp(distance, arc, xyz[:, k]) for k in range(3)])
    spline = parameterize(points, dt)
    t = np.linspace(0, spline.duration, 10*len(times))
    v, a = spline.evaluate(t, 1)[:, :2], spline.evaluate(t, 2)[:, :2]
    speed = np.linalg.norm(v, axis=1)
    turning = speed > .1
    yaw_rate = np.abs(v[turning, 0]*a[turning, 1] - v[turning, 1]*a[turning, 0])
    yaw_rate /= speed[turning]**2
    scale = max(1., float(speed.max()/max_speed),
                float(np.sqrt(np.linalg.norm(a, axis=1).max()/max_accel)),
                float(yaw_rate.max(initial=0.)/max_yaw_rate))
    return ExpertSpline(spline.control, spline.knots*scale)
