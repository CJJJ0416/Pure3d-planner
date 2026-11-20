"""RViz side-by-side comparison on shared native VST support positions."""
from common.paths import CONFIG_DIR
import fcntl
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import yaml


def run(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import MarkerArray,Marker
    from std_msgs.msg import String
    from common.io import atomic_json
    from common.io import cloud_message
    from common.visualization.visualization import marker, point, node_markers
    from Simulator.unitree_gazebo.scripts.frame_config import rotation_matrix
    from Simulator.unitree_gazebo.scripts.map_anchor import stamped_transform
    from tf2_ros import StaticTransformBroadcaster

    mission=Path(args.mission).resolve()
    if mission.is_file():mission=mission.parent
    directory=Path(args.output).resolve() if args.output else mission/'offline'
    comparison=directory/'vst_comparison'
    report=json.loads((comparison/'comparison.json').read_text())
    world=np.load(directory/'node_cost_gravity/world_xyz.npy',mmap_mode='r')
    old=np.load(directory/'merged/cost.npy',mmap_mode='r')
    aligned=np.load(directory/'merged_gravity/cost.npy',mmap_mode='r')
    vst=np.load(comparison/'merged_world.npy')
    index=np.load(comparison/'nearest_raw_index.npy')
    distance=np.load(comparison/'nearest_raw_distance.npy')
    low,high=np.array(report['display_roi_world_xy'])
    mask=((vst[:,:2]>=low)&(vst[:,:2]<=high)).all(1)&(vst[:,2]>=-1.5)&(distance<=.15)
    selected=np.flatnonzero(mask)
    if len(selected)>args.max_points:selected=selected[np.linspace(0,len(selected)-1,args.max_points,dtype=int)]
    support=vst[selected,:3]
    matched=index[selected]
    width=float(high[0]-low[0]+5.)
    shifts=[np.array([-width/2,0.,0.]),np.array([width/2,0.,0.])]
    model_xyz=support+shifts[0];vst_xyz=support+shifts[1]
    frames=json.loads((directory/'reconstruction/frames.json').read_text())
    rotation=rotation_matrix(np.array(frames['world_map_quaternion_xyzw']))
    translation=np.array(frames['world_map_translation'])
    nodes=json.loads((directory/'nodes/nodes.json').read_text())['nodes']
    odometry=json.loads((directory/'reconstruction/odometry.json').read_text())
    config=yaml.safe_load((CONFIG_DIR / 'whole_map.rviz').read_text())
    manager=config['Visualization Manager'];manager['Global Options']['Fixed Frame']='world'
    grid=manager['Displays'][0];grid['Cell Size']=2.;grid['Plane Cell Count']=80
    template=manager['Displays'][2]
    def display(name,topic,enabled=True,maximum=1.):
        from copy import deepcopy
        result=deepcopy(template);result['Name']=name;result['Topic']['Value']=topic
        result.update(Enabled=enabled,Value=enabled);result['Max Intensity']=maximum
        result['Size (Pixels)']=3
        return result
    manager['Displays']=[grid,
        display('LEFT - Gravity model on VST supports','/entry/comparison/gravity'),
        display('RIGHT - Native VST stitched','/entry/comparison/vst'),
        display('LEFT alternative - Original tilted-map model','/entry/comparison/original',False),
        display('LEFT alternative - Absolute VST difference','/entry/comparison/error',False,.5),
        display('LEFT alternative - All raw model points','/entry/comparison/raw_gravity',False),
        dict(Class='rviz_default_plugins/MarkerArray',Name='Node labels and actual paths',Enabled=True,Value=True,
             Topic={'Value':'/entry/comparison/markers','Depth':1,'History Policy':'Keep Last',
                    'Reliability Policy':'Reliable','Durability Policy':'Transient Local'})]
    view=manager['Views']['Current'];view.update(Distance=86.,Pitch=1.12,Yaw=-1.5707963,
        **{'Target Frame':'world','Focal Point':{'X':float((low[0]+high[0])/2),
            'Y':float((low[1]+high[1])/2),'Z':1.5}})
    config['Window Geometry'].update(Width=1800,Height=980)
    config_file=comparison/'compare.rviz';config_file.write_text(yaml.safe_dump(config,sort_keys=False))
    lock=Path(f'/tmp/pure3d_entry_{os.getuid()}_domain_{int(os.environ.get("ROS_DOMAIN_ID","0"))}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init(args=[]);node=Node('entry_vst_comparison')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs={name:node.create_publisher(PointCloud2,'/entry/comparison/'+name,qos)
          for name in ('gravity','vst','original','error','raw_gravity')}
    marker_pub=node.create_publisher(MarkerArray,'/entry/comparison/markers',qos)
    status_pub=node.create_publisher(String,'/entry/status',qos)
    static=StaticTransformBroadcaster(node);stamp=node.get_clock().now().to_msg()
    static.sendTransform([stamped_transform('world','map',stamp,(translation,np.array(frames['world_map_quaternion_xyzw']))),
        stamped_transform('map','camera_init',stamp,(np.zeros(3),np.array([0.,0.,0.,1.])))])
    def publish(name,xyz,values):
        message=cloud_message(xyz,stamp,values);message.header.frame_id='world';pubs[name].publish(message)
    publish('gravity',model_xyz,aligned[matched]);publish('vst',vst_xyz,vst[selected,3])
    publish('original',model_xyz,old[matched]);publish('error',model_xyz,np.abs(aligned[matched]-vst[selected,3]))
    raw_mask=((world[:,:2]>=low)&(world[:,:2]<=high)).all(1)&(world[:,2]>=-1.5)
    raw_rows=np.flatnonzero(raw_mask)
    if len(raw_rows)>args.max_points:raw_rows=raw_rows[np.linspace(0,len(raw_rows)-1,args.max_points,dtype=int)]
    publish('raw_gravity',world[raw_rows]+shifts[0],aligned[raw_rows])
    markers=MarkerArray()
    for pane,shift in enumerate(shifts):
        title=marker('titles',pane,Marker.TEXT_VIEW_FACING,stamp);title.header.frame_id='world'
        title.pose.position=point([shift[0],high[1]+2.,2.]);title.scale.z=1.
        title.color.r=title.color.g=title.color.b=title.color.a=1.
        title.text='Gravity-aligned model' if pane==0 else 'VST / same LIO input'
        markers.markers.append(title)
        values=[dict(id=n['id'],position=np.array(n['body_pose']['position'])@rotation.T+translation+shift) for n in nodes]
        node_array=node_markers(values,stamp)
        for item in node_array.markers:
            item.header.frame_id='world';item.ns=f'pane_{pane}_'+item.ns
            if item.type==Marker.SPHERE:item.scale.x=item.scale.y=item.scale.z=.35
            if item.type==Marker.TEXT_VIEW_FACING:item.scale.z=.5
        markers.markers.extend(node_array.markers)
        line=marker(f'pane_{pane}_actual',0,Marker.LINE_STRIP,stamp);line.header.frame_id='world'
        line.scale.x=.08;line.color.r,line.color.g,line.color.b,line.color.a=.25,.55,1.,1.
        line.points=[point(np.array(o['position'])@rotation.T+translation+shift) for o in odometry]
        markers.markers.append(line)
    marker_pub.publish(markers)
    status=dict(state='viewing_vst_comparison',fixed_frame='world',shared_support_points=len(selected),
        original_points=len(world),scene_roi=report['display_roi_world_xy'],
        display_min_world_z=-1.5,
        left='gravity model sampled at same VST support geometry',right='native VST, node deltas stitched',
        comparison=str(comparison/'comparison.json'),rviz_config=str(config_file))
    atomic_json(comparison/'view.json',status);status_pub.publish(String(data=json.dumps(status)))
    viewer=subprocess.Popen(['rviz2','-d',str(config_file),'--ros-args','-r','__node:=entry_compare_rviz',
        '-p','use_sim_time:=false'],start_new_session=True) if args.rviz else None
    node.get_logger().info(json.dumps(status))
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        lock.close()
