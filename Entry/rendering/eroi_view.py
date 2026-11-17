"""Inspect complete motion-and-hold point clouds, one colour per node."""
from Entry.utils.paths import CONFIG_DIR
import fcntl
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import yaml


PALETTE=['#ef4444','#f59e0b','#84cc16','#14b8a6','#06b6d4','#3b82f6','#8b5cf6','#d946ef','#f472b6','#cbd5e1']


def rgb_value(index):
    value=PALETTE[index%len(PALETTE)]
    return tuple(int(value[i:i+2],16) for i in (1,3,5))


def run(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from sensor_msgs.msg import PointCloud2,PointField
    from std_msgs.msg import Header,String,Int32
    from visualization_msgs.msg import Marker,MarkerArray,InteractiveMarker,InteractiveMarkerControl,InteractiveMarkerFeedback
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from tf2_ros import StaticTransformBroadcaster
    from common.unitree_gazebo.scripts.frame_config import rotation_matrix
    from common.unitree_gazebo.scripts.map_anchor import stamped_transform
    from Entry.utils.io import atomic_json
    from Entry.utils.io import cloud_message
    from Entry.rendering.visualization import block_markers, node_markers, marker, point

    mission=Path(args.mission).resolve()
    if mission.is_file():mission=mission.parent
    directory=Path(args.output).resolve() if args.output else mission/'offline'
    metadata=json.loads((directory/'nodes/nodes.json').read_text())
    nodes=metadata['nodes'];frames=json.loads((directory/'reconstruction/frames.json').read_text())
    rotation=rotation_matrix(np.array(frames['world_map_quaternion_xyzw']));translation=np.array(frames['world_map_translation'])
    world_positions=np.array([n['body_pose']['position'] for n in nodes])@rotation.T+translation
    odometry=json.loads((directory/'reconstruction/odometry.json').read_text())
    samples=[];erois=[]
    for n in nodes:
        coord=np.load(n['xyz'],mmap_mode='r')
        rows=np.arange(len(coord)) if len(coord)<=args.max_points else np.linspace(0,len(coord)-1,args.max_points,dtype=np.int64)
        samples.append(np.asarray(coord[rows]))
        erois.append(json.loads(Path(n['eroi']).read_text()))
    config=yaml.safe_load((CONFIG_DIR / 'entry.rviz').read_text());manager=config['Visualization Manager']
    manager['Global Options']['Fixed Frame']='world'
    from copy import deepcopy
    template=manager['Displays'][1]
    def cloud_display(name,topic,enabled,color='150; 170; 190',rgb=False):
        item=deepcopy(template);item.update(Name=name,Enabled=enabled,Value=enabled)
        item['Topic']['Value']=topic;item['Color Transformer']='RGB8' if rgb else 'FlatColor'
        item['Color']=color;item['Alpha']=1.;item['Size (Pixels)']=2
        return item
    def marker_display(name,topic,enabled):
        return dict(Class='rviz_default_plugins/MarkerArray',Name=name,Enabled=enabled,Value=enabled,
            Topic={'Value':topic,'Depth':1,'History Policy':'Keep Last','Reliability Policy':'Reliable','Durability Policy':'Transient Local'})
    manager['Displays']=[manager['Displays'][0],
        cloud_display('ACTIVE node - complete motion + hold scans','/entry/node_eroi/active_cloud',True,rgb=True),
        marker_display('ACTIVE node EROI spatial blocks','/entry/node_eroi/active_blocks',True),
        marker_display('All nodes + actual short paths','/entry/node_eroi/markers',True),
        dict(Class='rviz_default_plugins/InteractiveMarkers',Name='Click a node to inspect its EROI',Enabled=True,Value=True,
             **{'Interactive Markers Namespace':'/entry_node_eroi_controls'})]
    for i,n in enumerate(nodes):
        color='; '.join(map(str,rgb_value(i)))
        manager['Displays'].append(cloud_display(f'{n["id"]} independently - {n["points"]:,} points',f'/entry/node_eroi/cloud/{n["id"]}',False,color))
    manager['Tools']=manager['Tools'][:3]
    center=world_positions.mean(0)
    view=manager['Views']['Current'];view.update(**{'Target Frame':'world','Focal Point':dict(zip(('X','Y','Z'),map(float,center)))},
        Distance=float(max(22.,np.linalg.norm(np.ptp(world_positions,axis=0))*1.8)),Pitch=.9,Yaw=-1.1)
    config_file=directory/'nodes_eroi.rviz';config_file.write_text(yaml.safe_dump(config,sort_keys=False))
    lock=Path(f'/tmp/pure3d_entry_{os.getuid()}_domain_{int(os.environ.get("ROS_DOMAIN_ID","0"))}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init(args=[]);node=Node('entry_node_eroi_view')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    active_pub=node.create_publisher(PointCloud2,'/entry/node_eroi/active_cloud',qos)
    blocks_pub=node.create_publisher(MarkerArray,'/entry/node_eroi/active_blocks',qos)
    markers_pub=node.create_publisher(MarkerArray,'/entry/node_eroi/markers',qos)
    status_pub=node.create_publisher(String,'/entry/status',qos)
    pubs=[]
    stamp=node.get_clock().now().to_msg()
    static=StaticTransformBroadcaster(node)
    static.sendTransform([stamped_transform('world','map',stamp,(translation,np.array(frames['world_map_quaternion_xyzw']))),
        stamped_transform('map','camera_init',stamp,(np.zeros(3),np.array([0.,0.,0.,1.])))])
    for i,n in enumerate(nodes):
        pub=node.create_publisher(PointCloud2,f'/entry/node_eroi/cloud/{n["id"]}',qos);pub.publish(cloud_message(samples[i],stamp));pubs.append(pub)
    current=dict(index=0,automatic=args.cycle_seconds>0)
    def select(index):
        current['index']=int(index)%len(nodes);i=current['index'];n=nodes[i];red,green,blue=rgb_value(i)
        coord=samples[i];data=np.empty((len(coord),4),np.float32);data[:,:3]=coord
        data[:,3].view(np.uint32)[:]=(red<<16)|(green<<8)|blue
        cloud=PointCloud2(header=Header(frame_id='map',stamp=stamp),height=1,width=len(coord),is_bigendian=False,
            fields=[PointField(name=name,offset=4*k,datatype=PointField.FLOAT32,count=1) for k,name in enumerate(['x','y','z','rgb'])],
            point_step=16,row_step=16*len(coord),is_dense=True,data=data.tobytes())
        active_pub.publish(cloud)
        blocks_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        blocks=[dict(key=b['key'],marker_id=j) for j,b in enumerate(erois[i]['core_blocks'])]
        array=block_markers(blocks,[4.,4.,3.],stamp)
        for item in array.markers:
            item.header.frame_id='world';item.color.r=red/255.;item.color.g=green/255.;item.color.b=blue/255.
            item.color.a=.04 if item.type==Marker.CUBE else .3
        blocks_pub.publish(array)
        array=node_markers([dict(id=n['id'],position=world_positions[j]) for j,n in enumerate(nodes)],stamp)
        for item in array.markers:item.header.frame_id='world'
        for item in array.markers:
            if item.ns=='nodes':
                r,g,b=rgb_value(item.id);item.color.r=r/255.;item.color.g=g/255.;item.color.b=b/255.
                item.scale.x=item.scale.y=item.scale.z=.35 if item.id==i else .22
            if item.ns=='labels':item.scale.z=.35
        title=marker('active_title',0,Marker.TEXT_VIEW_FACING,stamp);title.header.frame_id='world'
        title.pose.position=point(world_positions[i]+[0.,0.,1.2]);title.scale.z=.38
        title.color.r=red/255.;title.color.g=green/255.;title.color.b=blue/255.;title.color.a=1.
        title.text=f'{n["id"]}: {n["points"]:,} points | '+('initial stationary' if i==0 else 'motion + hold')
        array.markers.append(title)
        line=marker('actual_trajectory',0,Marker.LINE_STRIP,stamp);line.header.frame_id='world';line.scale.x=.045
        line.color.r,line.color.g,line.color.b,line.color.a=.35,.65,1.,1.
        line.points=[point(np.array(o['position'])@rotation.T+translation) for o in odometry]
        array.markers.append(line);markers_pub.publish(array)
        status=dict(state='viewing_node_eroi',active_node=n['id'],node_points=n['points'],display_points=len(coord),
            nodes=len(nodes),eroi_blocks=len(blocks),automatic=current['automatic'],cycle_seconds=args.cycle_seconds,
            ownership=metadata['ownership'],cropped=False,first_scan_stamp=n['first_scan_stamp'],last_scan_stamp=n['last_scan_stamp'],
            hold_seconds=n.get('hold_seconds'),rviz_config=str(config_file))
        status_pub.publish(String(data=json.dumps(status)));atomic_json(directory/'node_eroi_view.json',status)
    def selection(message):
        if message.data<0:current['automatic']=True
        elif message.data<len(nodes):current['automatic']=False;select(message.data)
    subscription=node.create_subscription(Int32,'/entry/node_eroi/select',selection,10)
    server=InteractiveMarkerServer(node,'entry_node_eroi_controls')
    def clicked(feedback):
        if feedback.event_type==InteractiveMarkerFeedback.BUTTON_CLICK:
            current['automatic']=False;select(int(feedback.marker_name.split('_')[-1]))
    for i,n in enumerate(nodes):
        item=InteractiveMarker();item.header.frame_id='world';item.name=f'node_{i}';item.description=n['id'];item.scale=.6
        item.pose.position=point(world_positions[i]+[0.,0.,.3]);item.pose.orientation.w=1.
        control=InteractiveMarkerControl();control.name='select_eroi';control.interaction_mode=InteractiveMarkerControl.BUTTON;control.always_visible=True
        sphere=Marker(type=Marker.SPHERE);sphere.pose.orientation.w=1.;sphere.scale.x=sphere.scale.y=sphere.scale.z=.3
        r,g,b=rgb_value(i);sphere.color.r=r/255.;sphere.color.g=g/255.;sphere.color.b=b/255.;sphere.color.a=1.
        control.markers.append(sphere);item.controls.append(control);server.insert(item,feedback_callback=clicked)
    server.applyChanges()
    select(0)
    timer=node.create_timer(max(.5,args.cycle_seconds),lambda:select(current['index']+1) if current['automatic'] else None)
    viewer=subprocess.Popen(['rviz2','-d',str(config_file),'--ros-args','-r','__node:=entry_eroi_rviz','-p','use_sim_time:=false'],start_new_session=True) if args.rviz else None
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        lock.close()
