"""Show all saved random VST queries, including failed endpoints, without planning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy,QoSProfile,ReliabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker,MarkerArray
import yaml

from common.pointcloud_io import pcd_to_pointcloud2,read_pcd_xyz
from planner.light_planner.visualization.rviz import marker,marker_display,cloud_display


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('Simulator/scanning/terrain_batch_config.json'))
    parser.add_argument('--routes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--height-offset',type=float,default=0.,help='Display-only Z offset for paths and endpoint markers')
    parser.add_argument('--viewer-pid',type=int,help='Republish into an existing RViz, retaining its current camera')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    report=json.loads(args.routes.read_text())
    terrain=Path(config['terrain']).resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    rclpy.init()
    node=Node('pure3d_vst_random_queries_viewer')
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    header=Header(frame_id='map',stamp=node.get_clock().now().to_msg())
    publishers,payloads,displays={},[],[]
    prefix='/pure3d/vst_random_queries'

    def publish(name,message,kind):
        topic=prefix+'/'+name
        publisher=node.create_publisher(kind,topic,qos)
        publisher.publish(message)
        publishers[topic]=publisher
        payloads.append(message)
        return topic

    topic=publish('terrain',pcd_to_pointcloud2(terrain/config['clean_pcd'],header),PointCloud2)
    display=cloud_display('Terrain XYZ | complete static reference',topic,True,True)
    display.update(Alpha=.16,Selectable=False)
    displays.append(display)
    # Stored Tomogram PCD uses cost/50. Restore native cost only for display.
    topic=publish('vst',pcd_to_pointcloud2(terrain/'tomogram_nominal.pcd',header,cost_multiplier=50.),PointCloud2)
    display=cloud_display('VST cost | raw 0-50 | A* threshold 15',topic,True)
    display.update({'Max Intensity':50.,'Min Color':'20; 200; 70','Max Color':'240; 45; 35',
        'Size (Pixels)':2,'Alpha':.65,'Selectable':False})
    displays.append(display)
    colors=[(.15,.75,1.),(1.,.15,.2),(.85,.3,1.),(1.,.7,.05)]
    for index,row in enumerate(report['routes']):
        color=colors[index%len(colors)]
        items=[]
        if row['success']:
            path=np.load(row['gpmp']).copy()
            path[:,2]+=args.height_offset
            line=marker(header,row['id'],Marker.LINE_STRIP,color,0)
            line.scale.x=.13
            line.points=[Point(x=float(x),y=float(y),z=float(z)) for x,y,z in path]
            items.append(line)
        for number,(name,key,kind) in enumerate([('START','start_support_xyz',Marker.SPHERE),('GOAL','goal_support_xyz',Marker.CUBE)]):
            p=np.array(row[key]).copy()
            p[2]+=args.height_offset
            m=marker(header,row['id'],kind,color,number*3+1)
            m.pose.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
            m.scale.x=m.scale.y=m.scale.z=.55
            items.append(m)
            text=marker(header,row['id'],Marker.TEXT_VIEW_FACING,color,number*3+2)
            text.pose.position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2]+1.1))
            text.scale.z=.42
            text.text=f"{row['id']} {name}"+(' | A* FAILED' if not row['success'] else '')
            items.append(text)
            stem=marker(header,row['id'],Marker.LINE_LIST,color,number*3+3)
            stem.scale.x=.035
            stem.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2])),
                Point(x=float(p[0]),y=float(p[1]),z=float(p[2]+.9))]
            items.append(stem)
        topic=publish(row['id'],MarkerArray(markers=items),MarkerArray)
        displays.append(marker_display(row['id']+(' | GPMP' if row['success'] else ' | FAILED endpoints only'),topic,True))
    xyz=read_pcd_xyz(terrain/config['clean_pcd'])
    center=(xyz.min(0)+xyz.max(0))/2
    camera={'Class':'rviz_default_plugins/Orbit','Name':'All four queries','Distance':43.,
        'Pitch':.7,'Yaw':.85,'Target Frame':'<Fixed Frame>','Near Clip Distance':.01,
        'Focal Point':dict(zip(['X','Y','Z'],map(float,center)))}
    failed=next((r for r in report['routes']+report.get('attempt_history',[]) if not r['success']),None)
    saved=[camera,{**camera,'Name':'Top view','Pitch':1.55}]
    if failed is not None:
        focal=(np.array(failed['start_support_xyz'])+np.array(failed['goal_support_xyz']))/2
        saved.insert(1,{**camera,'Name':'Failed query close-up','Distance':19.,
            'Focal Point':dict(zip(['X','Y','Z'],map(float,focal)))})
    configuration={'Panels':[{'Class':'rviz_common/Displays','Name':'Displays'},{'Class':'rviz_common/Views','Name':'Views'}],
        'Visualization Manager':{'Class':'','Enabled':True,'Displays':displays,
            'Global Options':{'Fixed Frame':'map','Background Color':'24; 29; 38','Frame Rate':20},
            'Tools':[{'Class':'rviz_default_plugins/'+s} for s in ['Interact','MoveCamera','Select','FocusCamera']],
            'Views':{'Current':camera,'Saved':saved}},
        'Window Geometry':{'Width':1700,'Height':1000},'Preferences':{'PromptSaveOnExit':False}}
    file=out/'random_queries.rviz'
    file.write_text(yaml.safe_dump(configuration,sort_keys=False))
    (out/'queries.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'display.json').write_text(json.dumps(dict(height_offset_m=args.height_offset,
        applies_to=['GPMP paths','start markers','goal markers'],source_coordinates_unchanged=True),indent=2)+'\n')
    (out/'publisher.pid').write_text(str(os.getpid())+'\n')
    viewer=None if args.viewer_pid else subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(file),'--ros-args','-r','__node:=pure3d_vst_random_queries_rviz'])
    viewer_pid=args.viewer_pid or viewer.pid
    (out/'viewer.pid').write_text(str(viewer_pid)+'\n')
    print('RVIZ_ALL_QUERIES',viewer_pid,'height offset',args.height_offset,flush=True)
    checked=False
    while (Path(f'/proc/{viewer_pid}').exists() if viewer is None else viewer.poll() is None) and rclpy.ok():
        rclpy.spin_once(node,timeout_sec=.2)
        counts={topic:p.get_subscription_count() for topic,p in publishers.items()}
        if not checked and all(counts.values()):
            (out/'display_check.json').write_text(json.dumps(dict(all_layers_subscribed=True,subscriptions=counts),indent=2)+'\n')
            checked=True
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':main()
