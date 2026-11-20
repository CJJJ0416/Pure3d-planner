# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.visualization.landing_preview --help
"""Read-only RViz preview of the offline revision_1994 landing audit."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from planner.learning_planner.neupan.visualization.ros_view import path_message, slice_message


def markers(report, header):
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    result = [Marker(header=header, action=Marker.DELETEALL)]

    def spot(xyz, label, color, ns, idx, radius=None):
        x, y, z = map(float, xyz)
        rgba = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.)
        dot = Marker(header=header, ns=ns, id=idx, color=rgba, type=Marker.SPHERE)
        dot.pose.orientation.w = 1.
        dot.pose.position = Point(x=x, y=y, z=z + .04)
        dot.scale.x = dot.scale.y = dot.scale.z = .09
        text = Marker(header=header, ns=ns + '_text', id=idx, color=rgba, type=Marker.TEXT_VIEW_FACING)
        text.pose.orientation.w = 1.
        text.pose.position = Point(x=x, y=y, z=z + .26)
        text.scale.z = .095
        text.text = label
        result.extend((dot, text))
        if radius is not None:
            ring = Marker(header=header, ns=ns + '_patch', id=idx, color=rgba, type=Marker.LINE_STRIP)
            ring.pose.orientation.w = 1.
            ring.scale.x = .022
            theta = np.linspace(0, 2 * np.pi, 33)
            ring.points = [Point(x=x + radius * np.cos(t), y=y + radius * np.sin(t), z=z + .045) for t in theta]
            result.append(ring)

    if 'rollout' in report:
        spot(report['query_start'], 'Start', (.93, .95, .99), 'query_start', 0)
        spot(report['query_goal'], 'Goal', (.39, .92, .62), 'query_goal', 0)
        for label, key, color in (('B', 'before', (1., .65, .25)), ('A', 'after', (.21, .91, 1.))):
            for i, xyz in enumerate(report['rollout'][key]['selected_xyz'][1:], 1):
                point = report.get('sequence_case') and report['rollout'][key]['points'][i - 1]
                text = f"{label}{i}|A={point['area_m2']:.3f}|c={point['cost']:.2f}" if point else f'{label}{i}'
                spot(xyz, text, color, key, i, .15 if point else None)
        return MarkerArray(markers=result)

    for i, item in enumerate(report['targets'][1:], 1):
        if i == 1 and 'baseline_goals' in report:
            continue
        color = (.35, .91, .50) if item['eligible'] else (1., .45, .48)
        label = f'{item["name"]}|A={item["predicted_area_m2"]:.3f}'
        spot(item['xyz'], label, color, 'requested_goal', i, report['patch_radius_m'])
    for i, item in enumerate(report['checkpoints']):
        if 'support_xyz' in item:
            spot(item['support_xyz'], f'P{i + 1}|{item["area_m2"]:.3f}',
                 (.74, .53, .98), 'teacher_checkpoint', i)
    # The rejected curve remains visible for diagnosis. Marks on it are gray
    # and carry an explicit question mark to distinguish them from training data.
    for i, item in enumerate(report['diagnostic']):
        if 'baseline_goals' not in report:
            spot(item['xyz'], f'P{i + 1}?', (.67, .70, .77), 'diagnostic_only', i)
    if 'baseline_goals' in report:
        spot(report['query_start'], 'Start', (.93, .95, .99), 'query_start', 0)
        for prefix, key, color in (('Before', 'baseline_goals', (1., .65, .25)),
                                   ('After', 'learned_goals', (.21, .91, 1.))):
            for i, item in enumerate(report[key], 1):
                distance = item['observed_approach_m']
                length = f'{distance:.2f}m' if distance is not None else 'blocked'
                spot(item['xyz'], f'{prefix}{i}|{length}', color, key, i, report['patch_radius_m'])
        if not report['learned_goals']:
            spot(np.asarray(report['query_start']) + [0., 0., .22],
                 f'After|STOP|{report["learned_status"]}', (1., .46, .48), 'stop', 0)
    else:
        for i, item in enumerate(report['learned_goals'], 1):
            spot(item['xyz'], f'N{i}|A={item["observed_area_m2"]:.3f}',
                 (.21, .91, 1.), 'learned_goals', i)
    return MarkerArray(markers=result)


def run_preview(snapshot, rviz):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Header, String
    from tf2_ros import StaticTransformBroadcaster
    from visualization_msgs.msg import MarkerArray
    import yaml

    with np.load(snapshot, allow_pickle=False) as arrays:
        astar, gpmp, cloud = arrays['astar'], arrays['gpmp'], arrays['cloud']
        report = json.loads(str(arrays['report']))
    rollout = report.get('rollout')
    if rollout:
        first_key = 'approach_xyz' if report.get('sequence_case') else 'selected_xyz'
        astar = np.asarray(rollout['before'][first_key])
        gpmp = np.asarray(rollout['after']['approach_xyz'])
    rclpy.init()
    node = rclpy.create_node('landing_revision_preview')
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    prefix = '/entry/landing_preview/'
    header = Header(frame_id='world', stamp=node.get_clock().now().to_msg())
    tf = StaticTransformBroadcaster(node)
    origin = TransformStamped(header=header, child_frame_id='landing_revision_origin')
    origin.transform.rotation.w = 1.
    tf.sendTransform(origin)
    pubs = [
        (node.create_publisher(PointCloud2, prefix + 'cost', qos), slice_message(cloud, header)),
        (node.create_publisher(RosPath, prefix + 'astar', qos), path_message(astar, header)),
        (node.create_publisher(RosPath, prefix + 'gpmp', qos), path_message(gpmp, header)),
        (node.create_publisher(MarkerArray, prefix + 'patches', qos), markers(report, header)),
        (node.create_publisher(String, prefix + 'status', qos), String(data=json.dumps(report))),
    ]
    for publisher, message in pubs:
        publisher.publish(message)
    print(json.dumps(dict(revision=report['revision'], learned=len(report['learned_goals']),
                          curve=report['gpmp_result'])), flush=True)
    viewer = None
    with TemporaryDirectory(prefix='landing_preview_') as tmp:
        try:
            if rviz:
                config = yaml.safe_load(Path(__file__).with_name('preview.rviz').read_text())
                manager = config['Visualization Manager']
                cloud_display, astar_display, gpmp_display = manager['Displays'][1:]
                cloud_display['Topic']['Value'] = prefix + 'cost'
                cloud_display.update({'Style': 'Points', 'Size (Pixels)': 3, 'Alpha': .47,
                                      'Min Color': '87; 166; 127', 'Max Color': '205; 97; 113'})
                astar_display.update({'Name': 'VST A*', 'Color': '255; 174; 79', 'Line Width': .033})
                astar_display['Topic']['Value'] = prefix + 'astar'
                gpmp_display.update({'Name': 'GPMP diagnostic', 'Color': '166; 144; 255', 'Line Width': .035})
                gpmp_display['Topic']['Value'] = prefix + 'gpmp'
                gpmp_display['Offset']['Z'] = 0.
                if rollout:
                    names = (('VST expert prefix', 'Student sequence') if report.get('sequence_case') else
                             ('Before: selected points', 'After: supported approach'))
                    astar_display.update({'Name': names[0], 'Color': '255; 166; 64'})
                    gpmp_display.update({'Name': names[1], 'Color': '54; 232; 255'})
                elif 'baseline_goals' in report:
                    astar_display['Enabled'] = gpmp_display['Enabled'] = False
                manager['Displays'].append(dict(Class='rviz_default_plugins/MarkerArray', Name='Landing patches',
                                                Enabled=True, Topic={'Value': prefix + 'patches',
                                                                     'Reliability Policy': 'Reliable',
                                                                     'Durability Policy': 'Transient Local'}))
                view = manager['Views']['Current']
                center = np.asarray([row['xyz'] for row in report['targets']]).mean(0)
                view.update({'Focal Point': dict(zip(('X', 'Y', 'Z'), map(float, center))),
                             'Distance': 6.3, 'Pitch': .78, 'Yaw': .8})
                name = 'landing_approach_compare.rviz' if 'baseline_goals' in report else 'landing_revision_preview.rviz'
                if rollout:
                    view.update({'Distance': 4.8})
                    name = 'support_sequence_test.rviz' if report.get('sequence_case') else 'landing_route_rollout.rviz'
                settings = Path(tmp) / name
                settings.write_text(yaml.safe_dump(config, sort_keys=False))
                viewer = subprocess.Popen(['rviz2', '-d', str(settings)])
            while rclpy.ok() and (viewer is None or viewer.poll() is None):
                rclpy.spin_once(node, timeout_sec=.1)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            if viewer is not None and viewer.poll() is None:
                viewer.terminate()
                viewer.wait(timeout=5)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--rviz', action='store_true')
    args = parser.parse_args()
    run_preview(args.snapshot, args.rviz)
