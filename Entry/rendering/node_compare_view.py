"""Cycle independent node cost-head/VST comparisons and their stitched result."""
from Entry.utils.paths import CONFIG_DIR
import fcntl
import json
import os
from pathlib import Path
import subprocess
from copy import deepcopy
import numpy as np
import yaml


def run(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String,Int32
    from visualization_msgs.msg import Marker,MarkerArray,InteractiveMarker,InteractiveMarkerControl,InteractiveMarkerFeedback
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from tf2_ros import StaticTransformBroadcaster
    from Entry.utils.io import atomic_json
    from Entry.utils.io import cloud_message
    from Entry.rendering.visualization import marker, point
    from Entry.rendering.eroi_view import rgb_value
    from common.unitree_gazebo.scripts.frame_config import rotation_matrix
    from common.unitree_gazebo.scripts.map_anchor import stamped_transform

    mission=Path(args.mission).resolve()
    if mission.is_file():mission=mission.parent
    directory=Path(args.output).resolve() if args.output else mission/'offline'
    output=directory/'vst_nodewise';report=json.loads((output/'comparison.json').read_text())
    if report['state']!='complete':raise ValueError('Node comparison is incomplete')
    nodes=json.loads((directory/'nodes/nodes.json').read_text())['nodes']
    cost_dir=directory/('node_cost_'+report['variant'])
    world=np.load(cost_dir/'world_xyz.npy',mmap_mode='r')
    full_cost=np.load(directory/('merged_'+report['variant'])/'cost.npy',mmap_mode='r')
    lower,upper=world.min(0),world.max(0);width=float(upper[0]-lower[0]+8.)
    shifts=[np.array([-width/2,0.,0.]),np.array([width/2,0.,0.])]
    frames=json.loads((directory/'reconstruction/frames.json').read_text())
    rotation=rotation_matrix(np.array(frames['world_map_quaternion_xyzw']));translation=np.array(frames['world_map_translation'])
    positions=np.array([n['body_pose']['position'] for n in nodes])@rotation.T+translation
    odometry=json.loads((directory/'reconstruction/odometry.json').read_text())
    trajectory=np.array([o['position'] for o in odometry])@rotation.T+translation
    config=yaml.safe_load((CONFIG_DIR / 'entry.rviz').read_text());manager=config['Visualization Manager']
    manager['Global Options']['Fixed Frame']='world'
    def display(name,topic,enabled=True,maximum=1.):
        item=deepcopy(manager['Displays'][2]);item.update(Name=name,Enabled=enabled,Value=enabled)
        item['Topic']['Value']=topic;item['Max Intensity']=maximum;item['Size (Pixels)']=3
        return item
    grid=manager['Displays'][0];grid['Cell Size']=2.;grid['Plane Cell Count']=100
    manager['Displays']=[grid,
        display('LEFT - Cost head on SAME node supports','/entry/node_compare/head'),
        display('RIGHT - Native VST on SAME node supports','/entry/node_compare/vst'),
        display('LEFT alternative - Absolute head VST error','/entry/node_compare/error',False),
        display('LEFT alternative - All raw node cost points','/entry/node_compare/raw_head',False),
        dict(Class='rviz_default_plugins/MarkerArray',Name='Node IDs and actual trajectory',Enabled=True,Value=True,
            Topic={'Value':'/entry/node_compare/markers','Depth':1,'History Policy':'Keep Last',
                   'Reliability Policy':'Reliable','Durability Policy':'Transient Local'}),
        dict(Class='rviz_default_plugins/InteractiveMarkers',Name='Interact - select node or merged',Enabled=True,Value=True,
             **{'Interactive Markers Namespace':'/entry_node_compare_controls'})]
    manager['Tools']=manager['Tools'][:3]
    view=manager['Views']['Current'];view.update(Distance=float(width*2.05),Pitch=1.2,Yaw=-1.5707963,
        **{'Target Frame':'world','Focal Point':{'X':float((lower[0]+upper[0])/2),
            'Y':float((lower[1]+upper[1])/2),'Z':0.}})
    config['Window Geometry'].update(Width=1800,Height=980)
    config_file=output/'node_compare.rviz';config_file.write_text(yaml.safe_dump(config,sort_keys=False))
    lock=Path(f'/tmp/pure3d_entry_{os.getuid()}_domain_{int(os.environ.get("ROS_DOMAIN_ID","0"))}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init(args=[]);node=Node('entry_node_cost_comparison')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs={name:node.create_publisher(PointCloud2,'/entry/node_compare/'+name,qos) for name in ('head','vst','error','raw_head')}
    marker_pub=node.create_publisher(MarkerArray,'/entry/node_compare/markers',qos)
    status_pub=node.create_publisher(String,'/entry/status',qos)
    stamp=node.get_clock().now().to_msg();current=dict(index=0,automatic=args.cycle_seconds>0)
    static=StaticTransformBroadcaster(node)
    static.sendTransform([stamped_transform('world','map',stamp,(translation,np.array(frames['world_map_quaternion_xyzw']))),
        stamped_transform('map','camera_init',stamp,(np.zeros(3),np.array([0.,0.,0.,1.])))])
    def sample(rows):
        return rows if len(rows)<=args.max_points else rows[np.linspace(0,len(rows)-1,args.max_points,dtype=np.int64)]
    def publish(name,coord,cost):
        message=cloud_message(coord,stamp,cost);message.header.frame_id='world';pubs[name].publish(message)
    def select(index):
        i=int(index)%(len(nodes)+1);current['index']=i;merged=i==len(nodes)
        name='merged' if merged else nodes[i]['id']
        source=np.load(output/'merged_world.npy' if merged else report['nodes'][i]['shared'])
        rows=sample(np.flatnonzero(source[:,5]<=.15));shared=source[rows]
        publish('head',shared[:,:3]+shifts[0],shared[:,3]);publish('vst',shared[:,:3]+shifts[1],shared[:,4])
        publish('error',shared[:,:3]+shifts[0],np.abs(shared[:,3]-shared[:,4]))
        first,last=(0,len(world)) if merged else (nodes[i]['first_point'],nodes[i]['end_point'])
        raw_rows=sample(np.arange(first,last));publish('raw_head',world[raw_rows]+shifts[0],full_cost[raw_rows])
        array=MarkerArray()
        for pane,shift in enumerate(shifts):
            title=marker('pane_title',pane,Marker.TEXT_VIEW_FACING,stamp);title.header.frame_id='world'
            title.pose.position=point([(lower[0]+upper[0])/2+shift[0],upper[1]+3.,2.]);title.scale.z=1.2
            title.color.r=title.color.g=title.color.b=title.color.a=1.
            title.text=f'{name} | '+('Cost head' if pane==0 else 'Native VST')
            array.markers.append(title)
            line=marker('actual_path',pane,Marker.LINE_STRIP,stamp);line.header.frame_id='world';line.scale.x=.09
            line.color.r,line.color.g,line.color.b,line.color.a=.25,.6,1.,1.
            line.points=[point(x+shift) for x in trajectory];array.markers.append(line)
            for j,n in enumerate(nodes):
                label=marker('node_labels',pane*len(nodes)+j,Marker.TEXT_VIEW_FACING,stamp);label.header.frame_id='world'
                label.pose.position=point(positions[j]+shift+[0.,0.,1.]);label.scale.z=.45 if j==i else .32
                r,g,b=rgb_value(j);label.color.r=r/255.;label.color.g=g/255.;label.color.b=b/255.;label.color.a=1.;label.text=n['id']
                array.markers.append(label)
        marker_pub.publish(array)
        metric=report['merged_latest_support'] if merged else report['nodes'][i]['all_observed_support']
        status=dict(state='viewing_node_cost_comparison',active_node=name,automatic=current['automatic'],
            raw_points=last-first,matched_support_points=int(np.sum(source[:,5]<=.15)),display_support_points=len(shared),
            inference_frame='world',input_scope=report['input_scope'],metrics=metric,rviz_config=str(config_file))
        status_pub.publish(String(data=json.dumps(status)));atomic_json(output/'view.json',status)
    def selection(message):
        if message.data<0:current['automatic']=True
        elif message.data<=len(nodes):current['automatic']=False;select(message.data)
    subscription=node.create_subscription(Int32,'/entry/node_compare/select',selection,10)
    server=InteractiveMarkerServer(node,'entry_node_compare_controls')
    def clicked(feedback):
        if feedback.event_type==InteractiveMarkerFeedback.BUTTON_CLICK:
            current['automatic']=False;select(int(feedback.marker_name.split('_')[-1]))
    for i in range(len(nodes)+1):
        item=InteractiveMarker();item.header.frame_id='world';item.name=f'select_{i}';item.description=nodes[i]['id'] if i<len(nodes) else 'MERGED'
        item.scale=1.;item.pose.orientation.w=1.
        item.pose.position=point((positions[i] if i<len(nodes) else positions.mean(0)+[0.,-3.,0.])+shifts[0]+[0.,0.,.3])
        control=InteractiveMarkerControl();control.name='select';control.interaction_mode=InteractiveMarkerControl.BUTTON;control.always_visible=True
        sphere=Marker(type=Marker.SPHERE);sphere.pose.orientation.w=1.;sphere.scale.x=sphere.scale.y=sphere.scale.z=.45
        r,g,b=rgb_value(i);sphere.color.r=r/255.;sphere.color.g=g/255.;sphere.color.b=b/255.;sphere.color.a=1.
        control.markers.append(sphere);item.controls.append(control);server.insert(item,feedback_callback=clicked)
    server.applyChanges();select(0)
    timer=node.create_timer(max(.5,args.cycle_seconds),lambda:select(current['index']+1) if current['automatic'] else None)
    viewer=subprocess.Popen(['rviz2','-d',str(config_file),'--ros-args','-r','__node:=entry_node_compare_rviz','-p','use_sim_time:=false'],start_new_session=True) if args.rviz else None
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        lock.close()
