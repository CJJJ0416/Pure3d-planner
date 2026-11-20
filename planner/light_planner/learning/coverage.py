"""Evaluate the next step using only the local XYZ and traversability costs."""

import numpy as np
from scipy.spatial import cKDTree

from .expert import rotation

PERIOD = .2
SPEED_LEVELS = (1., .75, .5, .25)
COMMAND_LIMITS = np.array([1., .25, 1.])
# Two front foot neighborhoods, in the yaw frame. These approximate the
# support request of the current A1 velocity interface, which has no foot plan.
FRONT_FEET = np.array([[.18, -.10], [.18, .10]])
FOOT_SAMPLES = np.array([[0., 0.], [.035, 0.], [-.035, 0.], [0., .035], [0., -.035]])


class SupportCoverage:
    def __init__(self, xyz, cost, threshold=.5, minimum=.8, radius=.075, height=.075):
        xyz, cost = np.asarray(xyz, dtype=float), np.asarray(cost, dtype=float)
        if xyz.shape != (len(cost), 3) or cost.ndim != 1:
            raise ValueError('XYZ and cost rows must correspond')
        if not (0 < threshold <= 1 and 0 < minimum <= 1 and radius > 0 and height > 0):
            raise ValueError('Invalid low-cost coverage configuration')
        valid = np.isfinite(xyz).all(1) & np.isfinite(cost) & (cost >= 0) & (cost < threshold)
        self.xyz = xyz[valid]
        self.tree = cKDTree(self.xyz[:, :2])
        self.minimum, self.radius, self.height = minimum, radius, height

    def score(self, centers, yaw, feet=FRONT_FEET):
        """Minimum covered fraction in any individual foot/trajectory sample."""
        centers, yaw = np.asarray(centers), np.asarray(yaw)
        xy = np.stack([center[:2] + (feet[:, None]+FOOT_SAMPLES) @ rotation(angle).T
                       for center, angle in zip(centers, yaw)])
        if not len(self.xyz):
            return 0.
        neighbors = self.tree.query_ball_point(xy.reshape(-1, 2), self.radius)
        expected = np.repeat(centers[:, 2], len(feet)*len(FOOT_SAMPLES))
        hits = [any(abs(self.xyz[i, 2]-z) <= self.height for i in indices)
                for indices, z in zip(neighbors, expected)]
        return float(np.asarray(hits).reshape(-1, len(FOOT_SAMPLES)).mean(1).min())


def choose_command(coverage, command_at_rate, support_z, position, yaw, measured,
                   period=PERIOD, feet=FRONT_FEET):
    """Test the swept next step, including the measured speed at its start."""
    scores = []
    for rate in SPEED_LEVELS:
        cmd = np.clip(command_at_rate(rate), -COMMAND_LIMITS, COMMAND_LIMITS)
        if np.linalg.norm(cmd[:2]) < 1e-8 and abs(cmd[2]) < 1e-8:
            scores.append(0.)
            continue
        n = max(2, int(np.ceil(period*max(np.linalg.norm(measured[:2]),
                                         np.linalg.norm(cmd[:2]))/.05)))
        t = np.linspace(0., period, n+1)
        # Constant acceleration over the control interval, expressed in the
        # current yaw frame. Recheck the actual residual motion in simulation.
        xy = position[:2]+(t[:, None]*measured[:2]
                           +.5*t[:, None]**2/period*(cmd[:2]-measured[:2])) @ rotation(yaw).T
        angles = yaw+t*measured[2]+.5*t**2/period*(cmd[2]-measured[2])
        z = np.asarray(support_z(xy), dtype=float)
        value = coverage.score(np.c_[xy, z], angles, feet)
        scores.append(value)
    padded = np.zeros(len(SPEED_LEVELS), dtype=np.float32)
    padded[:len(scores)] = scores
    eligible = np.flatnonzero(padded >= coverage.minimum)
    if not len(eligible):
        return np.zeros(3), 0., padded
    rate = SPEED_LEVELS[eligible[0]]
    return np.clip(command_at_rate(rate), -COMMAND_LIMITS, COMMAND_LIMITS), rate, padded
