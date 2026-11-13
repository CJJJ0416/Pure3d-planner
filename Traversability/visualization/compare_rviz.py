"""在 RViz 并排查看同一测试建筑的原生 VST 与多个模型。

点坐标与代价值均直接读取 evaluate 导出的 NPZ。横向平移只用于 RViz
显示 TF，不修改保存的地图 XYZ。每组条件保留全部测试查询点。
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from common.pointcloud_io import pcd_to_pointcloud2
from Traversability.visualization.rviz import cloud_display, view_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--model', nargs=3, action='append', metavar=('KEY', 'LABEL', 'NPZ'),
                        help='增加一列模型预测，可重复传入。NPZ 由 evaluate --predictions 导出。')
    parser.add_argument('--scene', default='scene_20261127')
    parser.add_argument('--profiles', nargs='+', default=[
        'wide_clearance', 'joint_wide_slope', 'nominal', 'strict_stand'])
    parser.add_argument('--before-label', default='Original four-parameter model')
    parser.add_argument('--after-label', default='D50 best - epoch 27')
    parser.add_argument('--full-before', type=Path)
    parser.add_argument('--full-after', type=Path)
    parser.add_argument('--full-label', default='wide_clearance')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_LOG_DIR'] = str(args.output / 'ros_logs')
    # --model 支持 A/B/C 等任意数量的模型，原 before/after 调用方式仍可使用。
    sources = args.model or [('before', args.before_label, args.before),
                             ('after', args.after_label, args.after)]
    models = [(key, title, np.load(path), Path(path)) for key, title, path in sources]
    masks = {key: data['scene'] == args.scene for key, title, data, path in models}
    reference_key, _, reference, _ = models[-1]
    coord = reference['coord'][masks[reference_key]]
    lower, upper = coord.min(0), coord.max(0)
    spacing = float(upper[0] - lower[0] + 7.)
    columns = [(key, title, (index-len(models)/2)*spacing)
               for index, (key, title) in enumerate(
                   [('gt', 'VST ground truth')]+[(key, title) for key, title, data, path in models])]
    rclpy.init()
    node = Node('pure3d_model_comparison')
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    stamp = node.get_clock().now().to_msg()
    broadcaster = StaticTransformBroadcaster(node)
    transforms, markers, publishers, displays = [], [], [], []
    for index, (key, title, offset) in enumerate(columns):
        frame = 'comparison_' + key
        transform = TransformStamped(header=Header(frame_id='comparison', stamp=stamp),
                                     child_frame_id=frame)
        transform.transform.translation.x = offset
        transform.transform.rotation.w = 1.
        transforms.append(transform)
        marker = Marker(header=Header(frame_id=frame, stamp=stamp), ns='titles', id=index,
                        type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        marker.pose.position = Point(x=float((lower[0]+upper[0])/2), y=float(upper[1]+4),
                                     z=float(upper[2]+2))
        marker.pose.orientation.w = 1.
        marker.scale.z = 1.3
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.
        marker.text = title
        markers.append(marker)
    broadcaster.sendTransform(transforms)

    def publish(topic, message):
        publisher = node.create_publisher(PointCloud2, topic, qos)
        publisher.publish(message)
        publishers.append(publisher)
        print(f'{topic}: {message.width:,} points', flush=True)

    metrics = {}
    for group_index, profile in enumerate(args.profiles):
        reference_index = reference['profiles'].tolist().index(profile)
        gt = reference['target'][masks[reference_key], reference_index]
        values, coordinates = {'gt': gt}, {'gt': coord}
        metrics[profile] = {}
        for key, title, data, path in models:
            index = data['profiles'].tolist().index(profile)
            values[key] = data['prediction'][masks[key], index]
            coordinates[key] = data['coord'][masks[key]]
            metrics[profile][key+'_mae_raw'] = float(abs(values[key]-gt).mean()*50)
        children = []
        for key, title, offset in columns:
            xyz = coordinates[key]
            array = np.column_stack((xyz, values[key])).astype('<f4')
            topic = f'/pure3d/comparison/{profile}/{key}'
            message = PointCloud2(
                header=Header(frame_id='comparison_'+key, stamp=stamp), height=1, width=len(array),
                fields=[PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
                        for i,name in enumerate(['x','y','z','trav_cost'])],
                is_bigendian=False, point_step=16, row_step=16*len(array), is_dense=True,
                data=array.tobytes())
            publish(topic, message)
            display = cloud_display(title, topic, enabled=True)
            display.update({'Style': 'Points', 'Size (Pixels)': 2, 'Selectable': False})
            children.append(display)
        displays.append({'Class': 'rviz_common/Group', 'Name': profile+' | same XYZ | cost 0-1',
                         'Enabled': group_index == 0, 'Displays': children})

    full_displays = []
    for key, path in [('before', args.full_before), ('after', args.full_after)]:
        if path is not None:
            topic = '/pure3d/comparison/full_' + key
            publish(topic, pcd_to_pointcloud2(path, Header(frame_id='comparison_'+key, stamp=stamp)))
            display = cloud_display(key+' | ALL input PCD points', topic, enabled=True)
            display.update({'Style': 'Points', 'Size (Pixels)': 1, 'Selectable': False})
            full_displays.append(display)
    if full_displays:
        displays.append({'Class': 'rviz_common/Group', 'Name': 'FULL PCD | '+args.full_label+' | disable query groups first',
                         'Enabled': False, 'Displays': full_displays})
    label_topic = '/pure3d/comparison/titles'
    label_publisher = node.create_publisher(MarkerArray, label_topic, qos)
    label_publisher.publish(MarkerArray(markers=markers))
    displays.append({'Class': 'rviz_default_plugins/MarkerArray', 'Name': 'VST / MODEL TITLES',
                     'Enabled': True, 'Topic': {'Value': label_topic, 'Depth': 1,
                     'Reliability Policy': 'Reliable', 'Durability Policy': 'Transient Local'}})
    bounds = [lower + [columns[0][2], 0, 0], upper + [columns[-1][2], 5, 3]]
    config = view_config(dict(bounds_world=bounds, frame_id='comparison'), displays)
    camera = config['Visualization Manager']['Views']['Current']
    camera.update({'Distance': spacing*(len(columns)+.2), 'Pitch': .85, 'Yaw': -1.5707963})
    config['Visualization Manager']['Views']['Saved'] = []
    for key, title, offset in columns:
        close = dict(camera, Name=title, Distance=float(np.max(upper-lower)*1.6))
        close['Focal Point'] = dict(X=float(offset+(lower[0]+upper[0])/2),
                                   Y=float((lower[1]+upper[1])/2), Z=float((lower[2]+upper[2])/2))
        config['Visualization Manager']['Views']['Saved'].append(close)
    config['Window Geometry'].update(Width=1600, Height=950)
    config_path = args.output/'view.rviz'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    provenance = dict(scene=args.scene, points=len(coord),
                      models={key:dict(label=title, predictions=str(path.resolve()))
                              for key, title, data, path in models},
                      profiles=args.profiles, metrics=metrics,
                      display_offset_x_m={key:offset for key,title,offset in columns},
                      stored_xyz_unchanged=True)
    (args.output/'comparison.json').write_text(json.dumps(provenance, indent=2)+'\n')
    viewer = subprocess.Popen(['/opt/ros/humble/bin/rviz2', '-d', str(config_path),
                               '--ros-args', '-r', '__node:=pure3d_comparison_rviz'])
    (args.output/'viewer.pid').write_text(str(viewer.pid)+'\n')
    print(f'RViz pid={viewer.pid}, scene={args.scene}, all {len(coord):,} test queries', flush=True)
    while viewer.poll() is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
