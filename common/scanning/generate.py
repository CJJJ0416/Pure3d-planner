"""Generate fixed rolling scans and IMU streams without starting Isaac or LIO."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh

from utils.pointcloud import read_json, write_json
from .scene import build
from .trajectory import Trajectory
from .raycast import RollingRayCaster

DEFAULT_CONFIG = Path(__file__).with_name('config.json')


def load_config(path=DEFAULT_CONFIG):
    config = read_json(Path(path).resolve())
    if 'motion_speed_scale' in config:
        config['duration_s'] = (config['initial_static_s'] +
            config['reference_motion_duration_s']/config['motion_speed_scale'] + config['final_static_s'])
    return config


def imu_stream(trajectory, config, error, streams):
    dt = 1 / config['imu_hz']
    time = np.arange(0, config['duration_s'] + .1 + dt / 2, dt)
    # Hold the already stationary endpoint through the final LiDAR/IMU bracket.
    truth_time = np.minimum(time, config['duration_s'])
    ideal_acc, ideal_gyro = trajectory.imu(truth_time)
    acc, gyro = ideal_acc.copy(), ideal_gyro.copy()
    stamps = time.copy()
    biases = {}
    if error:
        rng = np.random.default_rng(streams['imu'])
        a, g = rng.normal(size=(2, 3))
        a *= config['acc_bias_norm'] / np.linalg.norm(a)
        g *= config['gyro_bias_norm'] / np.linalg.norm(g)
        acc += a + rng.normal(0, config['acc_noise_std'], acc.shape)
        gyro += g + rng.normal(0, config['gyro_noise_std'], gyro.shape)
        # Correlated measurement jitter, independent of the shared physical trajectory.
        # Stationary OU process with configured per-axis RMS and 40 ms correlation.
        alpha = np.exp(-dt / config['imu_jitter_correlation_s'])
        jitter_rng = np.random.default_rng(streams['imu_jitter'])
        vibration = np.empty((len(time), 6))
        vibration[0] = jitter_rng.normal(size=6)
        innovations = jitter_rng.normal(size=(len(time)-1, 6)) * np.sqrt(1-alpha**2)
        for i in range(1, len(time)):
            vibration[i] = alpha * vibration[i-1] + innovations[i-1]
        acc += vibration[:, :3] * config['imu_jitter_acc_std']
        gyro += vibration[:, 3:] * config['imu_jitter_gyro_std']
        jitter = np.random.default_rng(streams['time']).normal(0, config['imu_time_jitter_std_s'], len(time))
        stamps += config['imu_time_offset_s'] + jitter
        assert np.all(np.diff(stamps) > 0), 'Nonmonotonic IMU stamps in fixed sample'
        biases = dict(acc=a.tolist(), gyro=g.tolist(),
            jitter_acc_actual_std=(vibration[:, :3]*config['imu_jitter_acc_std']).std(0).tolist(),
            jitter_gyro_actual_std=(vibration[:, 3:]*config['imu_jitter_gyro_std']).std(0).tolist())
    return dict(time=stamps, physical_time=time, acceleration=acc, angular_velocity=gyro,
                ideal_acceleration=ideal_acc, ideal_angular_velocity=ideal_gyro), biases


def generate(config, profiles):
    root = Path(config['output']).resolve()
    config['output'] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    mesh = build(root, config)
    trajectory = Trajectory(mesh, config)
    trajectory_report = trajectory.save(root / 'truth_trajectory.npz')
    scale = config.get('motion_speed_scale', 1.)
    assert trajectory_report['max_speed_m_s'] <= .6*scale, trajectory_report
    assert trajectory_report['max_angular_speed_deg_s'] <= 45*scale, trajectory_report
    actors = [trimesh.load(root / f'dynamic_{name}.ply', process=False)
              for name in ['person_proxy', 'movable_clutter']]
    caster = RollingRayCaster(mesh, actors, config)
    faces = dict(np.load(root / 'mesh_faces.npz'))
    write_json(root / 'config.actual.json', config)
    write_json(root / 'trajectory_report.json', trajectory_report)
    write_json(root / 'pattern.json', dict(sha256_uncompressed=caster.pattern_sha256,
        source=read_json(config['pattern_metadata']), rays_per_scan=config['rays_per_scan'],
        pattern_samples_per_scan=config.get('pattern_samples_per_scan', config['rays_per_scan']),
        pattern_sampling='uniform centred subsampling within each frame, original angular phase advance retained',
        timing='synthetic uniform point spacing, NOT calibrated physical MID360 timing',
        mount_rpy_rad=config['mount_rpy_rad'], mount_reference=config['mount_reference'],
        lidar_from_imu=np.eye(4).tolist(), reflectivity='synthetic ray_id modulo 251 plus 1 for point-order audit only'))
    print('SCENE', json.dumps(trajectory_report), flush=True)
    for profile in profiles:
        profile_started = perf_counter()
        folder = root / profile
        scans = folder / 'scans'
        scans.mkdir(parents=True, exist_ok=True)
        # Shared independent streams make each corruption directly comparable across profiles.
        streams = {key: int(np.random.SeedSequence([config['seed'], code]).generate_state(1)[0])
                   for code, key in enumerate(['scene', 'trajectory', 'actors', 'measurement', 'imu', 'time', 'imu_jitter'], 101)}
        dynamic = profile in ['dynamic', 'mixed']
        edge = profile in ['edge_returns', 'mixed']
        error = profile in ['lio_error', 'mixed']
        imu, biases = imu_stream(trajectory, config, error, streams)
        np.savez(folder / 'imu.npz', **imu)
        rng = np.random.default_rng(streams['measurement'])
        records = []
        ray_seconds = save_seconds = 0.
        for frame in range(config.get('frames', round(config['duration_s'] * config['scan_hz']))):
            t0 = perf_counter()
            result = caster.scan(frame, trajectory, dynamic, edge, rng)
            ray_seconds += perf_counter() - t0
            static = result['dynamic_instance'] == 0
            semantic = np.empty(len(static), np.int32)
            instance = result['dynamic_instance'].copy()
            instance[static] = faces['instance_id'][result['face_id'][static]]
            semantic[static] = faces['semantic_id'][result['face_id'][static]]
            semantic[~static] = np.where(instance[~static] == 1001, 8, 10)
            result.update(instance_id=instance, semantic_id=semantic)
            t0 = perf_counter()
            np.savez(scans / f'{frame:05d}.npz', **result)
            save_seconds += perf_counter() - t0
            end = frame / config['scan_hz'] + float(np.float32(result['offset_ns'][-1] / 1e6)) / 1000
            records.append(dict(frame=frame, start_s=frame / config['scan_hz'], end_s=end,
                points=len(result['coord']), dynamic_points=int((~static).sum()),
                edge_mixed_points=int(result['edge_mixed'].sum()),
                blind_near_rays=int((result['return_status']==2).sum()),
                blind_fov_rays=int((result['return_status']==3).sum()), path=f'scans/{frame:05d}.npz'))
            if frame % 100 == 0:
                print(profile, frame, records[-1]['points'], flush=True)
        write_json(folder / 'scans.json', records)
        write_json(folder / 'generation.json', dict(profile=profile, streams=streams, biases=biases,
            dynamic=dynamic, edge_returns=edge, lio_error=error, frames=len(records),
            returns=sum(r['points'] for r in records), dynamic_returns=sum(r['dynamic_points'] for r in records),
            edge_mixed_returns=sum(r['edge_mixed_points'] for r in records),
            blind_near_rays=sum(r['blind_near_rays'] for r in records),
            blind_fov_rays=sum(r['blind_fov_rays'] for r in records),
            ray_query_and_sensor_s=ray_seconds, raw_save_s=save_seconds,
            imu_and_other_s=perf_counter()-profile_started-ray_seconds-save_seconds,
            wall_s=perf_counter()-profile_started))
        print('GENERATED', profile, sum(r['points'] for r in records), flush=True)
    write_json(root / 'generation.json', dict(profiles=profiles, wall_s=perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frames', type=int, help='Acquire a prefix of the configured trajectory without changing its speed')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output is not None:
        config['output'] = str(args.output)
    if args.frames is not None:
        config['frames'] = args.frames
    generate(config, args.profiles or config['profiles'])


if __name__ == '__main__':
    main()
