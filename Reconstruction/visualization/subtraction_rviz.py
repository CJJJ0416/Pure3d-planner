"""RViz comparison of complete, Subtraction and Filter PCDs, with no sampling."""
import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy,QoSProfile,ReliabilityPolicy
from geometry_msgs.msg import Point,TransformStamped
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker,MarkerArray
import yaml
from common.pointcloud_io import read_pcd_xyz,pcd_to_pointcloud2
from utils.pointcloud import read_json,write_json,workspace_path
from Traversability.visualization.rviz import cloud_display,view_config


def xyz_message(xyz,header):
    a=np.ascontiguousarray(xyz,dtype='<f4')
    return PointCloud2(header=header,height=1,width=len(a),
        fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z'])],
        is_bigendian=False,point_step=12,row_step=12*len(a),is_dense=True,data=a.tobytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='datasets/recon_terrain_blocks_preview_v1/terrain_835012/block_00145')
    p.add_argument('--variant',default='random_01')
    p.add_argument('--third',choices=['filter','removed'],default='removed')
    p.add_argument('--terrain',help='Optional large-terrain directory with terrain_display.pcd and blocks.json')
    p.add_argument('--overview',action='store_true',help='Open the entire terrain instead of block comparisons.')
    p.add_argument('--output',default='runs/recon_terrain_blocks_preview_v1/rviz')
    a=p.parse_args();folder=workspace_path(a.scene);out=workspace_path(a.output);out.mkdir(parents=True,exist_ok=True)
    os.environ['ROS_LOG_DIR']=str(out/'ros_logs')
    xyz=read_pcd_xyz(folder/'complete.pcd');lo,hi=xyz.min(0),xyz.max(0)
    spacing=float(hi[0]-lo[0]+2)
    rclpy.init();node=Node('recon_subtraction_preview')
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    stamp=node.get_clock().now().to_msg();pubs=[];displays=[];transforms=[];markers=[]
    rows=read_json(folder/'records.json')
    columns=[('gt','Complete GT / red = removed',-spacing),('input','Subtraction input',0.),('kept','Filter retained' if a.third=='filter' else 'Deleted surface targets',spacing)]
    for i,(key,title,offset) in enumerate(columns):
        tr=TransformStamped(header=Header(frame_id='recon_comparison',stamp=stamp),child_frame_id='recon_'+key)
        tr.transform.translation.x=offset;tr.transform.rotation.w=1.;transforms.append(tr)
        marker=Marker(header=Header(frame_id='recon_'+key,stamp=stamp),ns='titles',id=i,type=Marker.TEXT_VIEW_FACING,action=Marker.ADD)
        marker.pose.position=Point(x=float((lo[0]+hi[0])/2),y=float((lo[1]+hi[1])/2),z=float(hi[2]+.7))
        marker.pose.orientation.w=1.;marker.scale.z=.25
        marker.color.r=marker.color.g=marker.color.b=marker.color.a=1.;marker.text=title
        markers.append(marker)
        if 'core_removed_fraction' in rows[0]:
            from itertools import product
            bounds=[rows[0]['core_lower'],rows[0]['core_upper']]
            corners=np.array(list(product(*zip(*bounds))))
            outline=Marker(header=Header(frame_id='recon_'+key,stamp=stamp),ns='core_bounds',id=i,type=Marker.LINE_LIST,action=Marker.ADD)
            outline.pose.orientation.w=1.;outline.scale.x=.012
            outline.color.r=1.;outline.color.g=.7;outline.color.b=.05;outline.color.a=.65
            for j,v in enumerate(corners):
                for w in corners[j+1:]:
                    if np.count_nonzero(v!=w)==1:
                        outline.points.extend([Point(x=float(q[0]),y=float(q[1]),z=float(q[2])) for q in [v,w]])
            markers.append(outline)
    broadcaster=StaticTransformBroadcaster(node);broadcaster.sendTransform(transforms)
    def publish(topic,message):
        pub=node.create_publisher(PointCloud2,topic,qos);pub.publish(message);pubs.append(pub)
        print(f'{topic}: {message.width:,} points',flush=True)
    def display(name,topic,color,enabled=True,size=2):
        d=cloud_display(name,topic,enabled=enabled,flat=True)
        d.update({'Color':color,'Alpha':1.,'Size (Pixels)':size,'Selectable':False})
        return d
    topic='/pure3d/recon_subtraction/complete'
    publish(topic,pcd_to_pointcloud2(folder/'complete.pcd',Header(frame_id='recon_gt',stamp=stamp)))
    displays.append(display('Complete GT | ALL original points',topic,'155; 170; 185'))
    for row in rows:
        prefix=row['input_prefix'];base=f'/pure3d/recon_subtraction/{prefix}'
        with np.load(folder/f'{prefix}_indices.npz') as indices:removed=xyz[indices['removed_indices']]
        publish(base+'/removed',xyz_message(removed,Header(frame_id='recon_gt',stamp=stamp)))
        publish(base+'/input',pcd_to_pointcloud2(folder/f'{prefix}.pcd',Header(frame_id='recon_input',stamp=stamp)))
        publish(base+'/kept',pcd_to_pointcloud2(folder/f'{prefix}_kept.pcd',Header(frame_id='recon_kept',stamp=stamp)) if a.third=='filter' else xyz_message(removed,Header(frame_id='recon_kept',stamp=stamp)))
        children=[display('RED | deleted from complete GT',base+'/removed','255; 45; 45',size=3),
                  display('BLUE | exact input subset',base+'/input','75; 135; 255'),
                  display('GREEN | frozen Filter retained' if a.third=='filter' else 'RED | deleted targets only',base+'/kept','45; 220; 135' if a.third=='filter' else '255; 65; 65')]
        displays.append({'Class':'rviz_common/Group','Name':f'{prefix} | core {row["core_removed_fraction"]:.1%} / all {row["removed_fraction"]:.1%}' if 'core_removed_fraction' in row else f'{prefix} | removed {row["removed_fraction"]:.2%}',
                         'Enabled':prefix==a.variant,'Displays':children})
    label_topic='/pure3d/recon_subtraction/titles'
    pub=node.create_publisher(MarkerArray,label_topic,qos);pub.publish(MarkerArray(markers=markers));pubs.append(pub)
    displays.append({'Class':'rviz_default_plugins/MarkerArray','Name':'Column titles','Enabled':True,
                     'Topic':{'Value':label_topic,'Depth':1,'Reliability Policy':'Reliable','Durability Policy':'Transient Local'}})
    config=view_config(dict(bounds_world=[lo+[-spacing,0,0],hi+[spacing,0,1]],frame_id='recon_comparison'),displays)
    camera=config['Visualization Manager']['Views']['Current']
    camera.update(Distance=spacing*2.6,Pitch=.7,Yaw=1.5707963)
    saved=[]
    for key,title,offset in columns:
        close=dict(camera,Name=title,Distance=float(np.max(hi-lo)*1.5))
        close['Focal Point']=dict(X=float((lo[0]+hi[0])/2+offset),Y=float((lo[1]+hi[1])/2),Z=float((lo[2]+hi[2])/2))
        saved.append(close)
    if a.terrain:
        terrain=workspace_path(a.terrain)
        topic='/pure3d/recon_subtraction/large_terrain_context'
        context=read_pcd_xyz(terrain/'terrain_display.pcd')
        values=np.column_stack([context,context[:,2]]).astype('<f4')
        message=PointCloud2(header=Header(frame_id='recon_comparison',stamp=stamp),height=1,width=len(values),
            fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z','height'])],
            is_bigendian=False,point_step=16,row_step=len(values)*16,is_dense=True,data=values.tobytes())
        publish(topic,message)
        large=display('LARGE TERRAIN | height colors | display subset',topic,'125; 150; 175',enabled=a.overview,size=2)
        large.update({'Color Transformer':'Intensity','Channel Name':'height','Use rainbow':True,
                      'Min Intensity':float(context[:,2].min()),'Max Intensity':float(context[:,2].max())})
        if a.overview:
            for d in displays:
                d['Enabled']=False
                if 'Value' in d:d['Value']=False
        displays.append(large)
        bounds=np.asarray(read_json(terrain/'scene.json')['bounds_world'])
        overview=dict(camera,Name='LARGE TERRAIN overview',Distance=float(np.max(bounds[1]-bounds[0])*1.6),Pitch=.8,Yaw=.8)
        overview['Focal Point']=dict(zip(['X','Y','Z'],map(float,bounds.mean(0))))
        saved.append(overview)
        if a.overview:config['Visualization Manager']['Views']['Current']=overview
    config['Visualization Manager']['Views']['Saved']=saved
    config['Window Geometry'].update(Width=1650,Height=950)
    path=out/'subtraction.rviz';path.write_text(yaml.safe_dump(config,sort_keys=False))
    viewer=subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(path),'--ros-args','-r','__node:=recon_subtraction_rviz'])
    write_json(out/'session.json',dict(scene=str(folder),variant=a.variant,viewer_pid=viewer.pid,publisher_pid=os.getpid(),
                                      all_points=not a.overview,overview=a.overview,terrain=str(workspace_path(a.terrain)) if a.terrain else None,display_offsets_only=True,training=False))
    print(f'RViz opened: {folder.name}, variant {a.variant}, pid {viewer.pid}',flush=True)
    while viewer.poll() is None and rclpy.ok():rclpy.spin_once(node,timeout_sec=.2)
    node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
