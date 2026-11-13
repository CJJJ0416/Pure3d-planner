"""All-point raster projections, mesh overview and incremental scan comparisons."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import read_json


def raster(xyz, axes, bounds, values=None, color='#527db9'):
    width, height = 1000, 700
    low, high = bounds[:, axes]
    uv = (xyz[:, axes] - low) / (high - low)
    pixel = np.clip((uv * [width-1, height-1]).astype(np.int64), 0, [width-1, height-1])
    key = pixel[:, 1] * width + pixel[:, 0]
    count = np.bincount(key, minlength=width*height)
    occupied = count > 0
    rgba = np.ones((width*height, 4))
    if values is None:
        rgb = np.array(matplotlib.colors.to_rgb(color))
        rgba[occupied, :3] = rgb
    else:
        total = np.bincount(key, weights=values, minlength=width*height)
        mean = total[occupied] / count[occupied]
        rgba[occupied] = plt.colormaps['turbo'](np.clip(mean / .05, 0, 1))
    return rgba.reshape(height, width, 4)


def image_panel(ax, xyz, axes, bounds, title, values=None, color='#527db9'):
    ax.imshow(raster(xyz, axes, bounds, values, color), origin='lower',
              extent=[bounds[0, axes[0]], bounds[1, axes[0]], bounds[0, axes[1]], bounds[1, axes[1]]],
              interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Map ' + 'XYZ'[axes[0]] + ' (m)')
    ax.set_ylabel('Map ' + 'XYZ'[axes[1]] + ' (m)')


def overview(root):
    mesh = trimesh.load(root / 'static_mesh.ply', process=False)
    trajectory = dict(np.load(root / 'truth_trajectory.npz'))
    figure = plt.figure(figsize=(12, 8))
    ax = figure.add_subplot(projection='3d')
    colors = mesh.visual.face_colors.astype(float) / 255
    colors[:, 3] = .3
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=colors, linewidths=0))
    p = trajectory['position']
    ax.plot(*p.T, color='#c73535', linewidth=2, label='Continuous sensor trajectory')
    ax.scatter(*p[0], color='#00a0ba', s=55, label='Start / end')
    ax.set(xlim=(-5, 5), ylim=(-4, 5), zlim=(-.2, 3), xlabel='Map X (m)', ylabel='Map Y (m)', zlabel='Map Z (m)')
    ax.set_box_aspect([10, 9, 3.2])
    ax.view_init(35, -60)
    ax.legend()
    ax.set_title('Seed 821110 | connected corridor, stairs, upper landing and static controls')
    figure.savefig(root / 'overview.png', dpi=160, bbox_inches='tight')
    plt.close(figure)


def observation_model(root, profile):
    config = read_json(root/'config.actual.json')
    folder = root/profile
    imu = dict(np.load(folder/'imu.npz'))
    generation = read_json(folder/'generation.json')
    end = generation['frames']/config['scan_hz']
    mask = imu['physical_time'] <= end
    t = imu['physical_time'][mask]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for axis, key, ideal_key, component, title, unit in [
        (axes[0], 'acceleration', 'ideal_acceleration', 2, 'IMU Z specific force', 'm/s^2'),
        (axes[1], 'angular_velocity', 'ideal_angular_velocity', 2, 'IMU Z angular velocity', 'rad/s')]:
        axis.plot(t, imu[key][mask, component], linewidth=.6, alpha=.8, label='Noisy measurement')
        axis.plot(t, imu[ideal_key][mask, component], linewidth=1., color='black', label='Trajectory truth')
        axis.set(xlabel='Physical time (s)', ylabel=unit, title=title)
        axis.legend()
    axes[2].plot(t, (imu['time'][mask]-t)*1000, linewidth=.6)
    axes[2].set(xlabel='Physical time (s)', ylabel='Timestamp offset (ms)', title='IMU clock offset + jitter')
    fig.suptitle(profile+' | white noise, fixed bias and correlated measurement jitter')
    fig.tight_layout()
    fig.savefig(folder/'plots/imu_noise.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    radius = np.linspace(0, 8, 1000)
    sensor_low, sensor_high = np.deg2rad(config['vertical_fov_deg'])
    pitch = config.get('mount_rpy_rad', [0., 0., 0.])[1]
    low, high = sensor_low-pitch, sensor_high-pitch
    ax.fill_between(radius, np.tan(low)*radius, np.tan(high)*radius,
                    color='#8bc5af', alpha=.5, label='Forward slice of sensor FOV (-7 to 52 deg), after mounting')
    ax.axhspan(-1.2, -config['sensor_height_m'], color='#ddd5c7')
    ax.axhline(-config['sensor_height_m'], color='#72604e', label='Flat floor, sensor 0.8 m above')
    ax.axhline(1.8, color='#9e9e9e', linestyle='--', label='Ceiling 1.8 m above sensor')
    blind_radius = config['sensor_height_m']/np.tan(-low)
    ax.plot([0, blind_radius], [-config['sensor_height_m']]*2, color='#d14141', linewidth=4,
            label=f'Unseen forward floor: 0 to {blind_radius:.2f} m')
    ax.add_patch(plt.Circle((0, 0), config['blind_zone_m'], color='#df4d4d', zorder=5))
    ax.scatter([0], [0], color='black', s=25, zorder=6)
    ax.annotate('0.1 m proximity blind zone', xy=(.08, 0), xytext=(1, -.35),
                arrowprops=dict(arrowstyle='->'))
    ax.set(xlim=(-.3, 8), ylim=(-1.1, 3), xlabel='Horizontal distance from sensor (m)',
           ylabel='Height relative to sensor (m)',
           title=f'MID-360 forward blind geometry | mounting pitch {np.rad2deg(pitch):.3f} deg')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(folder/'plots/mid360_blind_zone.png', dpi=160)
    plt.close(fig)


def plot(root, profile):
    folder = root / profile
    out = folder / 'plots'
    out.mkdir(exist_ok=True)
    clean = read_pcd_xyz(root / 'clean.pcd')
    attrs = dict(np.load(folder / 'attributes.npz'))
    xyz, truth = attrs['coord'], attrs['truth_coord']
    metrics = read_json(folder / 'metrics.json')
    bounds = np.array([np.minimum(clean.min(0), xyz.min(0))-.1,
                       np.maximum(clean.max(0), xyz.max(0))+.1])
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for row, projection in enumerate([(0, 1), (1, 2)]):
        image_panel(axes[row, 0], clean, projection, bounds, f'Mesh-sampled GT PCD | {len(clean):,} points')
        image_panel(axes[row, 1], truth, projection, bounds, 'Same returns fused with truth poses')
        image_panel(axes[row, 2], xyz, projection, bounds, 'Faster-LIO | distance to Mesh', attrs['surface_distance_m'])
        image_panel(axes[row, 3], xyz, projection, bounds, 'Faster-LIO | distance to sampled GT PCD', attrs['gt_pcd_distance_m'])
    cax = fig.add_axes([.935, .25, .008, .5])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, .05), cmap='turbo'), cax=cax)
    bar.set_label('Distance (m), saturated at 5 cm')
    fig.suptitle(f'{profile} | {metrics["registered_frames"]} registered frames | {len(xyz):,} original returns\n'
                 'One fixed initial-frame alignment. All points contribute. Colour is mean distance per pixel.', fontsize=15)
    fig.subplots_adjust(top=.87, bottom=.1, left=.045, right=.91, wspace=.3, hspace=.3)
    fig.savefig(out / 'gt_alignment.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for key, label in [('surface_distance_m', 'To exact Mesh'), ('gt_pcd_distance_m', 'To sampled GT PCD')]:
        axes[0].hist(np.minimum(attrs[key]*100, 10), bins=np.linspace(0, 10, 101),
                     density=True, histtype='step', linewidth=1.5, label=label)
    axes[0].set(xlabel='Distance (cm, clipped at 10)', ylabel='Density', title='Observed-point correspondence')
    axes[0].legend()
    axes[1].hist(attrs['keep_target'], bins=50, color='#288b76', log=True)
    axes[1].set(xlabel='Continuous keep_target', ylabel='Points (log)', title='Mesh-distance supervision')
    axes[2].axis('off')
    d = metrics
    text = (f"Registered: {d['registered_frames']} / {d['input_frames']} frames\n"
            f"Returns: {d['points']:,}\n\n"
            f"Scan to Mesh MAE: {d['scan_to_mesh']['mean_m']*100:.3f} cm\n"
            f"Scan to GT PCD MAE: {d['scan_to_gt_pcd']['mean_m']*100:.3f} cm\n"
            f"Mesh distance <= 3 cm: {d['scan_to_mesh']['within_3cm']*100:.2f}%\n"
            f"Truth-fused to Mesh: {d['truth_fused_to_mesh']['mean_m']*100:.4f} cm\n\n"
            f"LIO position RMSE: {d['trajectory']['position_rmse_m']*100:.3f} cm\n"
            f"Reverse GT coverage <= 3 cm: {d['gt_pcd_to_truth_fused']['within_3cm']*100:.2f}%\n"
            'Unobserved GT surfaces are missing coverage,\nnot bad observed-point labels.')
    axes[2].text(0, .98, text, va='top', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'distance_diagnostics.png', dpi=150)
    plt.close(fig)
    for second in range(10, round(metrics['input_frames']/10)+1, 10):
        mask = attrs['frame_index'] < second*10
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        image_panel(axes[0], truth[mask], (1, 2), bounds, f'{second} s | truth fusion')
        image_panel(axes[1], xyz[mask], (1, 2), bounds, 'Incremental Faster-LIO')
        image_panel(axes[2], xyz[mask], (1, 2), bounds, 'Distance to Mesh (0-5 cm)', attrs['surface_distance_m'][mask])
        fig.tight_layout()
        fig.savefig(out / f'incremental_{second:03d}s.png', dpi=140)
        plt.close(fig)
    overview(root)
    if 'ideal_acceleration' in np.load(folder/'imu.npz').files:
        observation_model(root, profile)
    print('PLOTS', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    plot(args.root.resolve(), args.profile)


if __name__ == '__main__':
    main()
