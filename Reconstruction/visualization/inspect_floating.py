"""离线 PCD 几何悬空可疑度可视化，保持全部原始 XYZ 和点序。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common.pointcloud_io import read_pcd_xyz, write_pcd


def voxel_representatives(xyz, voxel):
    """每个网格保留一个真实观测点，并返回到全部原始点的 inverse。"""
    grid = np.floor(xyz.astype(np.float64) / voxel).astype(np.int64)
    grid -= grid.min(0)
    key = np.ravel_multi_index(grid.T, tuple(grid.max(0) + 1))
    _, selected, inverse = np.unique(key, return_index=True, return_inverse=True)
    return xyz[selected], inverse.astype(np.int32)


def component_rows(xyz, labels, counts, original_counts, cost, anchor_distance):
    """按组件保存空间范围和证据，便于复核而不自动删除点。"""
    import scipy.ndimage as ndi

    ids = np.arange(len(counts))
    bounds = np.array([
        [ndi.minimum(xyz[:, axis], labels, ids), ndi.maximum(xyz[:, axis], labels, ids)]
        for axis in range(3)
    ]).transpose(2, 1, 0)
    means = np.column_stack([ndi.mean(xyz[:, axis], labels, ids) for axis in range(3)])
    max_cost = ndi.maximum(cost, labels, ids)
    mean_cost = ndi.mean(cost, labels, ids)
    mean_gap = ndi.mean(anchor_distance, labels, ids)
    rows = []
    for index in np.flatnonzero(max_cost >= 30):
        rows.append(dict(
            component_id=int(index), representative_points=int(counts[index]),
            original_points=int(original_counts[index]), center=means[index].tolist(),
            bounds=bounds[index].tolist(), mean_cost=float(mean_cost[index]),
            max_cost=float(max_cost[index]), mean_anchor_distance_m=float(mean_gap[index]),
        ))
    rows.sort(key=lambda row: row['original_points'] * row['mean_cost'], reverse=True)
    return rows


def plot_results(output, xyz, cost, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    palette = LinearSegmentedColormap.from_list(
        'floating_cost', [(0, '#8195a3'), (.25, '#b9b49c'), (.5, '#ffd34e'), (1, '#e23c32')],
    )
    rng = np.random.default_rng(0)
    context = rng.choice(len(xyz), min(230000, len(xyz)), replace=False)
    highlighted = np.flatnonzero(cost >= 30)
    selected = np.unique(np.concatenate([context, highlighted]))
    selected = selected[np.argsort(cost[selected], kind='stable')]
    points, values = xyz[selected], cost[selected]
    bounds = np.array(metadata['bounds_world'])
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3, projection='3d'), fig.add_subplot(2, 2, 4, projection='3d')]
    axes[0].scatter(points[:, 0], points[:, 1], s=.3, c='#4889bd', linewidths=0)
    artist = axes[1].scatter(points[:, 0], points[:, 1], s=.6, c=values,
                             cmap=palette, vmin=0, vmax=60, linewidths=0)
    axes[2].scatter(*points.T, s=.2, c='#4889bd', linewidths=0, depthshade=False)
    axes[3].scatter(*points.T, s=.7, c=values, cmap=palette, vmin=0, vmax=60,
                    linewidths=0, depthshade=False)
    for index, axis in enumerate(axes):
        axis.set(xlabel='Map X (m)', ylabel='Map Y (m)')
        axis.set_title(['Raw PCD | top', 'Floating cost | top',
                        'Raw PCD | oblique', 'Floating cost | oblique'][index])
        if index < 2:
            axis.set_aspect('equal')
            axis.set_xlim(bounds[:, 0])
            axis.set_ylim(bounds[:, 1])
            axis.grid(alpha=.15)
        else:
            axis.set_zlabel('Map Z (m)')
            axis.set_box_aspect(bounds[1] - bounds[0])
            axis.view_init(elev=30, azim=-65)
            axis.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2])
    fig.colorbar(artist, ax=[axes[1], axes[3]], shrink=.8,
                 label='Geometric floating suspicion cost 0-60 (not VST / not PTv3)')
    fig.suptitle(f'ROT_stair | {metadata["points"]:,} original points scored\n'
                 'All original XYZ retained. Plot uses voxel representatives. '
                 'Disconnected real structures may also score high.', fontsize=15)
    fig.savefig(output / 'overview.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), subplot_kw={'projection': '3d'},
                             constrained_layout=True)
    for axis, candidate in zip(axes.flat, metadata['closeups']):
        center = np.asarray(candidate['center'])
        context = min(5.0, max(1.5, candidate['mean_anchor_distance_m'] + .5))
        half_size = np.maximum((np.diff(candidate['bounds'], axis=0)[0]) / 2 + 1.0, context)
        mask = (np.abs(xyz - center) <= half_size).all(1)
        order = np.flatnonzero(mask)
        order = order[np.argsort(cost[order], kind='stable')]
        artist = axis.scatter(*xyz[order].T, s=2, c=cost[order], cmap=palette,
                              vmin=0, vmax=60, linewidths=0, depthshade=False)
        axis.scatter(*center, s=110, facecolors='none', edgecolors='black', linewidths=1.2)
        axis.set(xlabel='Map X (m)', ylabel='Map Y (m)', zlabel='Map Z (m)',
                 xlim=[center[0]-half_size[0], center[0]+half_size[0]],
                 ylim=[center[1]-half_size[1], center[1]+half_size[1]],
                 zlim=[center[2]-half_size[2], center[2]+half_size[2]])
        axis.set_box_aspect(half_size)
        axis.view_init(elev=25, azim=-60)
        axis.set_title(f'{candidate["view_group"]} | component {candidate["component_id"]}\n'
                       f'{candidate["original_points"]:,} original points\n'
                       f'mean cost {candidate["mean_cost"]:.1f} | '
                       f'anchor gap {candidate["mean_anchor_distance_m"]:.2f} m')
    fig.colorbar(artist, ax=list(axes.flat), shrink=.75, label='Floating suspicion cost 0-60')
    fig.suptitle('Candidate close-ups | contextual geometry retained | geometric heuristic only')
    fig.savefig(output / 'closeups.png', dpi=160)
    plt.close(fig)


def analyze(args):
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    xyz = read_pcd_xyz(args.pcd)
    coord, inverse = voxel_representatives(xyz, args.voxel)
    print(f'Original {len(xyz):,} -> geometric representatives {len(coord):,}', flush=True)
    tree = cKDTree(coord)
    pairs = tree.query_pairs(args.radius, output_type='ndarray')
    graph = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(coord), len(coord)))
    number, labels = connected_components(graph, directed=False)
    del graph, pairs
    counts = np.bincount(labels)
    original_counts = np.bincount(labels, weights=np.bincount(inverse, minlength=len(coord)))
    anchor = counts[labels] >= args.anchor_points
    print(f'Components {number:,}; anchors {anchor.sum():,}; '
          f'largest {counts.max():,} representatives', flush=True)
    distances = np.empty((len(coord), 2), dtype=np.float32)
    anchor_distance = np.empty(len(coord), dtype=np.float32)
    anchor_tree = cKDTree(coord[anchor])
    for first in range(0, len(coord), 32768):
        last = min(first + 32768, len(coord))
        distance, _ = tree.query(coord[first:last], k=[17, 65], workers=-1)
        distances[first:last] = distance
        anchor_distance[first:last] = anchor_tree.query(coord[first:last], workers=-1)[0]
    baseline = float(np.quantile(distances[anchor, 0], .95))
    smallness = np.clip(np.log(args.anchor_points / counts[labels]) /
                        np.log(args.anchor_points), 0, 1)
    gap = np.clip((anchor_distance - args.radius) / (1.0 - args.radius), 0, 1)
    sparsity = np.clip((distances[:, 0] - baseline) / (3 * baseline), 0, 1)
    cost = (60 * np.clip(smallness * (.35 + .65 * gap) + .25 * sparsity, 0, 1)).astype(np.float32)
    rows = component_rows(coord, labels, counts, original_counts, cost, anchor_distance)
    closeups = []
    for name, lower, upper, min_points in [('High-score small cluster', 45, 61, 10),
                                          ('Detached fragment: ambiguous', 20, 45, 30)]:
        chosen = 0
        for row in rows:
            if lower <= row['mean_cost'] < upper and row['original_points'] >= min_points and all(
                np.linalg.norm(np.asarray(row['center']) - other['center']) > 4 for other in closeups
            ):
                closeups.append(dict(row, view_group=name))
                chosen += 1
                if chosen == 2:
                    break
    point_cost = cost[inverse]
    write_pcd(output / 'floating_cost.pcd', xyz, point_cost, field_name='floating_cost')
    np.savez_compressed(output / 'scores.npz', floating_cost=point_cost, inverse=inverse,
                        representative_coord=coord, representative_cost=cost,
                        component=labels.astype(np.int32), component_size=counts,
                        distance_16=distances[:, 0], distance_64=distances[:, 1],
                        anchor_distance=anchor_distance)
    edges = np.arange(0, 61, 10)
    hist, _ = np.histogram(point_cost, bins=edges)
    metadata = dict(
        kind='geometric_floating_diagnostic', source_pcd=str(args.pcd.resolve()),
        coordinate_frame='map', points=len(xyz), representatives=len(coord),
        bounds_world=[xyz.min(0).tolist(), xyz.max(0).tolist()],
        focus_bounds=np.quantile(xyz, [.005, .995], axis=0).tolist(),
        voxel_m=args.voxel, connection_radius_m=args.radius,
        anchor_min_representatives=args.anchor_points, components=int(number),
        anchor_components=int(np.count_nonzero(counts >= args.anchor_points)),
        anchor_distance_16_p95_m=baseline,
        formula='60*clip(smallness*(0.35+0.65*gap)+0.25*sparsity,0,1)',
        definitions={
            'smallness':'clip(log(anchor_min/component_representatives)/log(anchor_min),0,1)',
            'gap':'clip((distance_to_large_component-radius)/(1m-radius),0,1)',
            'sparsity':'clip((distance_16-anchor_distance_16_p95)/(3*anchor_distance_16_p95),0,1)',
        },
        cost_histogram=[dict(low=int(lo), high=int(hi), points=int(value))
                        for lo, hi, value in zip(edges[:-1], edges[1:], hist)],
        points_cost_ge_30=int((point_cost >= 30).sum()),
        points_cost_ge_45=int((point_cost >= 45).sum()),
        candidate_components=rows, closeups=closeups,
        analysis_and_export_s=time.perf_counter()-start,
        interpretation='Untrained geometric suspicion, not ground truth or VST traversability. '
                       'FOV-truncated true structures can score high. No points deleted.',
    )
    (output / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ['components','anchor_components',
          'points_cost_ge_30','points_cost_ge_45','analysis_and_export_s']}), flush=True)
    plot_results(output, coord, cost, metadata)
    print(f'Plots: {output / "overview.png"} and closeups.png', flush=True)


def view(output):
    import os
    import subprocess
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import TransformStamped
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    import yaml

    from common.pointcloud_io import pcd_to_pointcloud2
    from Traversability.visualization.rviz import cloud_display, view_config

    output = output.resolve()
    metadata = json.loads((output / 'report.json').read_text())
    displays = []
    for name, flat, enabled in [('Floating cost 0-60 | ALL original points | heuristic', False, True),
                                ('Raw XYZ | same points | flat color', True, False)]:
        item = cloud_display(name, '/pure3d/floating_diagnostic', enabled=enabled, flat=flat,
                             cost_max=60)
        item.update({'Channel Name':'floating_cost', 'Style':'Points', 'Size (Pixels)':1,
                     'Selectable':False, 'Use rainbow':False,
                     'Min Color':'120; 145; 160', 'Max Color':'255; 35; 20'})
        displays.append(item)
    config = view_config(dict(bounds_world=metadata['focus_bounds'], frame_id='map'), displays)
    camera = config['Visualization Manager']['Views']['Current']
    camera.update({'Name':'Main structure', 'Pitch':.8, 'Yaw':-.7})
    saved = []
    for row in metadata['closeups']:
        closeup = dict(camera)
        closeup.update(Name=f'Candidate {row["component_id"]} | cost {row["mean_cost"]:.0f}',
                       Distance=float(max(np.ptp(row['bounds'], axis=0)) + 5),
                       **{'Focal Point':dict(zip(['X','Y','Z'], row['center']))})
        saved.append(closeup)
    config['Visualization Manager']['Views']['Saved'] = saved
    config_path = output / 'floating.rviz'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    os.environ.setdefault('ROS_LOG_DIR', str(output / 'ros_logs'))
    rclpy.init()
    node = rclpy.create_node('pure3d_floating_diagnostic')
    static_tf = StaticTransformBroadcaster(node)
    transform = TransformStamped(header=Header(frame_id='world', stamp=node.get_clock().now().to_msg()),
                                   child_frame_id='map')
    transform.transform.rotation.w = 1.0
    static_tf.sendTransform(transform)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(PointCloud2, '/pure3d/floating_diagnostic', qos)
    header = Header(frame_id='map', stamp=node.get_clock().now().to_msg())
    publisher.publish(pcd_to_pointcloud2(output / 'floating_cost.pcd', header))
    viewer = subprocess.Popen(['/opt/ros/humble/bin/rviz2', '-d', str(config_path)])
    print(f'RViz pid={viewer.pid}, published {metadata["points"]:,} original points', flush=True)
    result = viewer.wait()
    print(f'RViz exited: {result}', flush=True)
    node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analyze_parser = commands.add_parser('analyze')
    analyze_parser.add_argument('--pcd', type=Path, required=True)
    analyze_parser.add_argument('--output', type=Path, required=True)
    analyze_parser.add_argument('--voxel', type=float, default=.1)
    analyze_parser.add_argument('--radius', type=float, default=.18)
    analyze_parser.add_argument('--anchor-points', type=int, default=1000)
    view_parser = commands.add_parser('view')
    view_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'analyze':
        analyze(args)
    else:
        view(args.output)


if __name__ == '__main__':
    main()
