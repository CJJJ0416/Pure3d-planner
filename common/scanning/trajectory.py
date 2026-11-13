"""One C2 kinematic trajectory drives both rolling rays and IMU measurements."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import open3d as o3d


def quintic(u):
    u = np.clip(u, 0, 1)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


class Trajectory:
    def __init__(self, mesh, config):
        self.config = config
        self.mount_rotation = Rotation.from_euler('xyz', config['mount_rpy_rad']).as_matrix()
        # Start within the corridor, turn left into the stairwell and return.
        xy = np.array([[-1.2, -2], [-.6, -2], [0., -2], [.382683, -1.92388],
                       [.707107, -1.707107], [.92388, -1.382683], [1., -1.],
                       [1., -.2], [1., 1.0], [1., 2.2], [1., 3.65]])
        knots = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        planar = CubicSpline(knots, xy, bc_type='natural')
        s = np.linspace(0, knots[-1], 2500)
        p = planar(s)
        query = o3d.t.geometry.RaycastingScene(nthreads=4)
        query.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        rays = np.c_[p, np.full(len(p), 2.45), np.zeros((len(p), 2)), -np.ones(len(p))].astype(np.float32)
        ground = 2.45 - query.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        # Smooth the stepped support over 9 cm; no per-frame teleports on risers.
        ground = gaussian_filter1d(ground, .09 / (s[1] - s[0]), mode='nearest')
        xyz = np.c_[p, ground + config['sensor_height_m']]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        xyz = CubicSpline(arc / arc[-1], xyz)
        sample_t = np.linspace(0, config['duration_s'], 12001)
        duration = config['duration_s'] - config['initial_static_s'] - config['final_static_s']
        phase = (sample_t - config['initial_static_s']) / (duration / 2)
        fraction = np.where(phase <= 1, quintic(phase), 1 - quintic(phase - 1))
        position = xyz(fraction)
        tangent = xyz(fraction, 1)
        heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        middle = config['initial_static_s'] + duration / 2
        turn_half = 4 * duration / 50.
        heading += np.pi * quintic((sample_t - (middle - turn_half)) / (2*turn_half))
        self.position = CubicSpline(sample_t, position, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
        self.heading = CubicSpline(sample_t, heading, bc_type=((1, 0.), (1, 0.)))
        self.length_m = float(2 * arc[-1])

    def evaluate(self, time):
        time = np.asarray(time)
        position = self.position(time)
        yaw = self.heading(time)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.zeros(time.shape + (3, 3))
        rotation[..., 0, 0] = c
        rotation[..., 0, 1] = -s
        rotation[..., 1, 0] = s
        rotation[..., 1, 1] = c
        rotation[..., 2, 2] = 1
        return position, rotation @ self.mount_rotation

    def imu(self, time):
        _, rotation = self.evaluate(time)
        acceleration = self.position(time, 2) - np.array([0, 0, -9.81])
        specific_force = np.einsum('...ji,...j->...i', rotation, acceleration)
        angular_world = np.zeros_like(specific_force)
        angular_world[..., 2] = self.heading(time, 1)
        angular_velocity = np.einsum('...ji,...j->...i', rotation, angular_world)
        return specific_force, angular_velocity

    def save(self, path):
        t = np.linspace(0, self.config['duration_s'], int(self.config['duration_s'] * self.config['imu_hz']) + 1)
        p, r = self.evaluate(t)
        acc, gyro = self.imu(t)
        speed = np.linalg.norm(self.position(t, 1), axis=1)
        omega = np.abs(self.heading(t, 1))
        np.savez(path, time=t, position=p, quaternion=Rotation.from_matrix(r).as_quat(),
                 specific_force=acc, angular_velocity=gyro)
        return dict(length_m=self.length_m, max_speed_m_s=float(speed.max()),
                    max_angular_speed_deg_s=float(np.rad2deg(omega.max())),
                    motion_speed_scale=self.config.get('motion_speed_scale', 1.),
                    initial_static_s=self.config['initial_static_s'],
                    final_static_s=self.config['final_static_s'],
                    motion_duration_s=self.config['duration_s']-self.config['initial_static_s']-self.config['final_static_s'],
                    mount_rpy_rad=self.config['mount_rpy_rad'],
                    static_specific_force=acc[0].tolist(), finite=bool(np.isfinite(acc).all()))


def actor_position(time, actor):
    time = np.asarray(time)
    speed, y = [(.6, -2.4), (.25, -1.6)][actor]
    amplitude = 1.15
    x = -1.9 + amplitude * np.sin(speed / amplitude * time + actor)
    return np.stack((x, np.full_like(x, y), np.zeros_like(x)), axis=-1)
