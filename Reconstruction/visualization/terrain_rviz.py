"""Whole-map Subtraction in RViz, in the original map frame."""
import argparse
import os
from pathlib import Path
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy,QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
import yaml
from common.pointcloud_io import read_pcd_xyz,pcd_to_pointcloud2
from utils.pointcloud import read_json,write_json,workspace_path
from Traversability.visualization.rviz import cloud_display,view_config


def height_cloud(xyz,header):
    values=np.column_stack([xyz,xyz[:,2]]).astype('<f4')
    return PointCloud2(header=header,height=1,width=len(values),
        fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z','height'])],
        is_bigendian=False,point_step=16,row_step=len(values)*16,is_dense=True,data=values.tobytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='datasets/recon_terrain_blocks_preview_v1/terrain_835012/global_subtraction_251113')
    p.add_argument('--output',default='runs/recon_terrain_blocks_preview_v1/rviz_global')
    a=p.parse_args();folder=workspace_path(a.data);out=workspace_path(a.output);out.mkdir(parents=True,exist_ok=True)
    os.environ['ROS_LOG_DIR']=str(out/'ros_logs')
    report=read_json(folder/'report.json');scene=read_json(folder.parent/'scene.json');bounds=np.asarray(scene['bounds_world'])
    rclpy.init();node=Node('recon_whole_terrain_subtraction')
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    header=Header(frame_id='map',stamp=node.get_clock().now().to_msg());pubs=[];displays=[]
    for key,file,title,enabled in [
        ('input','input_display.pcd','CORRUPTED TERRAIN | height colors',True),
        ('removed','removed_display.pcd','RED | deleted surface GT | toggle to inspect holes',True),
        ('gt','gt_display.pcd','ORIGINAL GT | toggle input off before comparing',False)]:
        topic='/pure3d/recon_global/'+key
        if key=='removed':message=pcd_to_pointcloud2(folder/file,header)
        else:message=height_cloud(read_pcd_xyz(folder/file),header)
        pub=node.create_publisher(PointCloud2,topic,qos);pub.publish(message);pubs.append(pub)
        d=cloud_display(title,topic,enabled=enabled,flat=True)
        d.update({'Alpha':1.,'Size (Pixels)':2,'Selectable':False})
        if key=='removed':d['Color']='255; 35; 35'
        else:d.update({'Color Transformer':'Intensity','Channel Name':'height','Use rainbow':True,
                       'Min Intensity':float(bounds[0,2]),'Max Intensity':float(bounds[1,2])})
        displays.append(d);print(f'{topic}: {message.width:,} display points',flush=True)
    cfg=view_config(dict(bounds_world=bounds.tolist(),frame_id='map'),displays)
    camera=cfg['Visualization Manager']['Views']['Current']
    camera.update(Name='Whole terrain / global Subtraction',Distance=float(np.max(bounds[1]-bounds[0])*1.5),Pitch=.48,Yaw=.8)
    cfg['Window Geometry'].update(Width=1650,Height=950)
    path=out/'terrain_subtraction.rviz';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    viewer=subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(path),'--ros-args','-r','__node:=recon_global_terrain_rviz'])
    write_json(out/'session.json',dict(viewer_pid=viewer.pid,publisher_pid=os.getpid(),data=str(folder),
                                      coordinate_frame='map',whole_terrain=True,training=False,report=report))
    print(f'RViz opened: whole terrain, {report["removed_points"]:,} removed ({report["removed_fraction"]:.2%})',flush=True)
    while viewer.poll() is None and rclpy.ok():rclpy.spin_once(node,timeout_sec=.2)
    node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
