"""查看正式DC结果、无跨阶边消融与可选的已保存参考轨迹。"""
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
from geometry_msgs.msg import Point,TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker,MarkerArray
import yaml

from common.pointcloud_io import pcd_to_pointcloud2,read_pcd_xyz


def marker(header,name,kind,color,number=0):
    m=Marker(header=header,ns=name,id=number,type=kind,action=Marker.ADD)
    m.pose.orientation.w=1.;m.color.r,m.color.g,m.color.b=map(float,color);m.color.a=1.
    return m


def marker_display(name,topic,enabled=True):
    return {'Class':'rviz_default_plugins/MarkerArray','Name':name,'Enabled':enabled,
        'Topic':{'Value':topic,'Depth':1,'History Policy':'Keep Last','Reliability Policy':'Reliable','Durability Policy':'Transient Local'}}


def cloud_display(name,topic,enabled=True,flat=False):
    return {'Class':'rviz_default_plugins/PointCloud2','Name':name,'Enabled':enabled,'Value':enabled,
        'Position Transformer':'XYZ','Color Transformer':'FlatColor' if flat else 'Intensity',
        'Color':'150; 170; 190','Channel Name':'trav_cost','Autocompute Intensity Bounds':False,
        'Min Intensity':0.,'Max Intensity':1.,'Use rainbow':False,'Min Color':'20; 200; 60','Max Color':'240; 45; 35',
        'Style':'Points','Size (Pixels)':1 if flat else 2,'Size (m)':.03,'Alpha':.2 if flat else .75,
        'Decay Time':0,'Use Fixed Frame':True,'Selectable':True,
        'Topic':{'Value':topic,'Depth':1,'History Policy':'Keep Last','Reliability Policy':'Reliable','Durability Policy':'Transient Local'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='可选静态参考清单，只读取PCD/CSV，不执行其它规划器')
    args=parser.parse_args();run=args.run.resolve();out=run/'rviz';out.mkdir(exist_ok=True)
    os.environ['ROS_LOG_DIR']=str(out/'ros_logs')
    summary=json.loads((run/'summary.json').read_text());graph=np.load(run/'map/graph_view.npz')
    rclpy.init();node=Node('pure3d_dc_planner_viewer')
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    header=Header(frame_id='map',stamp=node.get_clock().now().to_msg());prefix='/pure3d/dc_planner'
    tf=StaticTransformBroadcaster(node);t=TransformStamped(header=Header(frame_id='world',stamp=header.stamp),child_frame_id='map');t.transform.rotation.w=1.;tf.sendTransform(t)
    publishers={};payloads={};required=[];route_manifest=[]

    def publish(name,value,kind,enabled):
        topic=prefix+'/'+name;p=node.create_publisher(kind,topic,qos);p.publish(value)
        publishers[topic]=p;payloads[topic]=value
        if enabled:required.append(topic)
        return topic

    displays=[]
    for name,label,file,enabled in [('voxel','Predicted cost | mean voxel | 0-1',run/'map/voxel_cost.pcd',True),
        ('surface','DC cost surface | optional',run/'map/dc_surface_cost.pcd',False)]:
        topic=publish(name,pcd_to_pointcloud2(file,header),PointCloud2,enabled)
        displays.append(cloud_display(label,topic,enabled))

    def add_route(name,label,files,color,enabled):
        children=[]
        for stage,file in files.items():
            a=np.genfromtxt(file,delimiter=',',names=True);p=np.column_stack([a[k] for k in ['x','y','z']])
            line=marker(header,name+'/'+stage,Marker.LINE_STRIP,color);line.scale.x=.035 if stage=='astar' else .075
            line.points=[Point(x=float(x),y=float(y),z=float(z)) for x,y,z in p]
            topic=publish(name+'/'+stage,MarkerArray(markers=[line]),MarkerArray,enabled)
            length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
            children.append(marker_display(f'{stage.upper()} | {length:.3f}m',topic))
            route_manifest.append(dict(method=name,stage=stage,file=str(file),topic=topic,enabled=enabled,length_m=length))
        displays.append({'Class':'rviz_common/Group','Name':label,'Enabled':enabled,'Displays':children})

    for name,r in summary['variants'].items():
        if not r['success']:
            displays.append({'Class':'rviz_common/Group','Name':name+' | NO PATH','Enabled':False,'Displays':[]})
            continue
        main_method=name=='with_step';enabled=main_method or 'with_step' not in summary['variants']
        add_route(name,'PURPLE | DC + step' if main_method else 'DARK PURPLE | DC no-step ablation',
            dict(astar=run/name/'astar_path.csv',gpmp=run/name/'gpmp_trajectory.csv'),
            (.8,.15,1.) if main_method else (.4,.2,.8),enabled)

    if args.reference is not None:
        source=args.reference.resolve();ref=json.loads(source.read_text());base=source.parent
        for i,c in enumerate(ref['clouds']):
            topic=publish(f'reference_cloud_{i}',pcd_to_pointcloud2(base/c['file'],header),PointCloud2,c['enabled'])
            displays.append(cloud_display(c['label'],topic,c['enabled'],c['flat']))
        for i,r in enumerate(ref['routes']):
            add_route(f'reference_{i}',r['name']+' | recorded reference',
                dict(astar=base/r['astar'],gpmp=base/r['gpmp']),r['color'],r['enabled'])

    for name,key,color in [('original_edges','base_edges',(.65,.68,.73)),('added_step_edges','added_edges',(1.,.5,.02))]:
        points=graph['xyz'][graph[key]].reshape(-1,3)
        line=marker(header,name,Marker.LINE_LIST,color);line.scale.x=.004 if name=='original_edges' else .012;line.color.a=.4
        line.points=[Point(x=float(x),y=float(y),z=float(z)) for x,y,z in points]
        topic=publish(name,MarkerArray(markers=[line]),MarkerArray,False)
        displays.append(marker_display(name+' | optional graph view',topic,False))
    chosen='with_step' if 'with_step' in summary['variants'] else 'without_step'
    if summary['variants'][chosen]['success']:
        ends=json.loads((run/chosen/'endpoints.json').read_text())['gpmp_xyz']
    else:ends=[summary['query']['start'],summary['query']['goal']]
    markers=[]
    for i,(name,point,color) in enumerate([('START',ends[0],(.9,.9,.9)),('GOAL',ends[1],(1.,.2,.2))]):
        m=marker(header,'endpoints',Marker.SPHERE,color,2*i);m.pose.position=Point(x=float(point[0]),y=float(point[1]),z=float(point[2]));m.scale.x=m.scale.y=m.scale.z=.16;markers.append(m)
        m=marker(header,'endpoints',Marker.TEXT_VIEW_FACING,color,2*i+1);m.pose.position=Point(x=float(point[0]),y=float(point[1]),z=float(point[2]+.3));m.scale.z=.2;m.text=name;markers.append(m)
    topic=publish('endpoints',MarkerArray(markers=markers),MarkerArray,True);displays.append(marker_display('START / GOAL | GPMP body targets',topic))
    points=read_pcd_xyz(run/'map/voxel_cost.pcd');center=(points.min(0)+points.max(0))/2
    camera={'Class':'rviz_default_plugins/Orbit','Name':'DC planning','Distance':max(9.,float(np.ptp(points,axis=0).max())*1.45),
        'Pitch':.65,'Yaw':.8,'Target Frame':'<Fixed Frame>','Near Clip Distance':.01,
        'Focal Point':dict(zip(['X','Y','Z'],map(float,center)))}
    config={'Panels':[{'Class':'rviz_common/Displays','Name':'Displays'},{'Class':'rviz_common/Views','Name':'Views'}],
        'Visualization Manager':{'Class':'','Enabled':True,'Displays':displays,
            'Global Options':{'Fixed Frame':'map','Background Color':'24; 29; 38','Frame Rate':20},
            'Tools':[{'Class':'rviz_default_plugins/'+s} for s in ['Interact','MoveCamera','Select','FocusCamera']],
            'Views':{'Current':camera,'Saved':[dict(camera),dict(camera,Name='Top view',Pitch=1.5,Yaw=1.57)]}},
        'Window Geometry':{'Width':1750,'Height':1000},'Preferences':{'PromptSaveOnExit':False}}
    file=out/'dc_planner.rviz';file.write_text(yaml.safe_dump(config,sort_keys=False))
    (out/'manifest.json').write_text(json.dumps(dict(routes=route_manifest,reference=str(args.reference) if args.reference else None),indent=2)+'\n')
    (out/'publisher.pid').write_text(str(os.getpid())+'\n')
    viewer=subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(file),'--ros-args','-r','__node:=pure3d_dc_planner_rviz'])
    (out/'viewer.pid').write_text(str(viewer.pid)+'\n');print('RVIZ',viewer.pid,flush=True)
    started=time.monotonic();checked=False
    while viewer.poll() is None and rclpy.ok():
        rclpy.spin_once(node,timeout_sec=.2)
        if not checked and time.monotonic()-started>5:
            counts={t:p.get_subscription_count() for t,p in publishers.items()}
            if all(counts[t]>0 for t in required):
                (out/'display_check.json').write_text(json.dumps(dict(all_enabled_layers_subscribed=True,counts=counts),indent=2)+'\n');checked=True
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
