"""Native RViz view of independent LIO PCDs in their original map coordinates."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy,QoSProfile,ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml

from common.pointcloud_io import pcd_to_pointcloud2,read_pcd_xyz
from planner.visualization.rviz import cloud_display


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    out=(args.output or args.report.parent/'rviz_scans').resolve()
    out.mkdir(parents=True,exist_ok=True)
    rclpy.init()
    node=Node('pure3d_lio_batch_viewer')
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    header=Header(frame_id='map',stamp=node.get_clock().now().to_msg())
    tf=StaticTransformBroadcaster(node)
    transform=TransformStamped(header=Header(frame_id='world',stamp=header.stamp),child_frame_id='map')
    transform.transform.rotation.w=1.
    tf.sendTransform(transform)
    colors=['40; 200; 255','255; 85; 85','210; 85; 255','255; 205; 55']
    publishers,payloads,displays,manifest=[],[],[],[]
    required=[]
    for index,row in enumerate(report['routes']):
        path=Path(row['root'])/'mixed/rough.pcd'
        message=pcd_to_pointcloud2(path,header)
        topic='/pure3d/lio_batch/'+row['id']
        publisher=node.create_publisher(PointCloud2,topic,qos)
        publisher.publish(message)
        publishers.append((topic,publisher))
        payloads.append(message)
        required.append(topic)
        display=cloud_display(f"{row['id']} | {message.width:,} LIO points",topic,True,True)
        display.update({'Color':colors[index%len(colors)],'Alpha':.85,'Selectable':False,'Size (Pixels)':1})
        displays.append(display)
        manifest.append(dict(route=row['id'],file=str(path),topic=topic,points=message.width,
            color=colors[index%len(colors)],coordinate_frame='map',coordinate_offset=[0,0,0]))
    reference=Path(report['routes'][0]['root'])/'clean.pcd'
    topic='/pure3d/lio_batch/gt'
    message=pcd_to_pointcloud2(reference,header)
    publisher=node.create_publisher(PointCloud2,topic,qos)
    publisher.publish(message)
    publishers.append((topic,publisher))
    payloads.append(message)
    display=cloud_display('GT static reference | optional overlay',topic,False,True)
    display.update({'Color':'170; 185; 195','Alpha':.35,'Selectable':False,'Size (Pixels)':1})
    displays.append(display)
    xyz=read_pcd_xyz(reference)
    center=(xyz.min(0)+xyz.max(0))/2
    camera={'Class':'rviz_default_plugins/Orbit','Name':'All LIO maps','Distance':43.,
        'Pitch':.7,'Yaw':.85,'Target Frame':'<Fixed Frame>','Near Clip Distance':.01,
        'Focal Point':dict(zip(['X','Y','Z'],map(float,center)))}
    saved=[camera,{**camera,'Name':'Top view','Pitch':1.55}]
    for row in report['routes']:
        xyz=read_pcd_xyz(Path(row['root'])/'mixed/rough.pcd')
        middle=(xyz.min(0)+xyz.max(0))/2
        saved.append({**camera,'Name':row['id']+' close-up','Distance':max(8.,float(np.ptp(xyz,axis=0).max())*1.15),
            'Focal Point':dict(zip(['X','Y','Z'],map(float,middle)))})
    configuration={'Panels':[{'Class':'rviz_common/Displays','Name':'Displays'},{'Class':'rviz_common/Views','Name':'Views'}],
        'Visualization Manager':{'Class':'','Enabled':True,'Displays':displays,
            'Global Options':{'Fixed Frame':'map','Background Color':'24; 29; 38','Frame Rate':20},
            'Tools':[{'Class':'rviz_default_plugins/'+s} for s in ['Interact','MoveCamera','Select','FocusCamera']],
            'Views':{'Current':camera,'Saved':saved}},
        'Window Geometry':{'Width':1700,'Height':1000},'Preferences':{'PromptSaveOnExit':False}}
    file=out/'lio_scans.rviz'
    file.write_text(yaml.safe_dump(configuration,sort_keys=False))
    (out/'manifest.json').write_text(json.dumps(dict(clouds=manifest,reference=str(reference),
        all_original_points=True,cloud_height_offset_m=0,source_report=str(args.report.resolve())),indent=2)+'\n')
    (out/'publisher.pid').write_text(str(os.getpid())+'\n')
    viewer=subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(file),'--ros-args','-r','__node:=pure3d_lio_batch_rviz'])
    (out/'viewer.pid').write_text(str(viewer.pid)+'\n')
    print('RVIZ_LIO_MAPS',viewer.pid,flush=True)
    checked=False
    while viewer.poll() is None and rclpy.ok():
        rclpy.spin_once(node,timeout_sec=.2)
        counts={topic:p.get_subscription_count() for topic,p in publishers}
        if not checked and all(counts[t]>0 for t in required):
            (out/'display_check.json').write_text(json.dumps(dict(all_enabled_layers_subscribed=True,subscriptions=counts),indent=2)+'\n')
            checked=True
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':main()
