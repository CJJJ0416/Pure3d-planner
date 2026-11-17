"""Publish saved offline geometry, cost, spatial blocks and goal nodes for RViz."""
from Entry.utils.paths import CONFIG_DIR
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
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as PathMessage
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    from tf2_ros import StaticTransformBroadcaster
    from common.unitree_gazebo.scripts.map_anchor import stamped_transform
    from Entry.utils.io import atomic_json
    from Entry.utils.core import partition
    from Entry.utils.io import cloud_message
    from Entry.rendering.visualization import block_markers, node_markers
    from Entry.rendering.display_sampling import display_rows

    mission = Path(args.mission).resolve()
    if mission.is_file(): mission=mission.parent
    directory = Path(args.output).resolve() if args.output else mission/'offline'
    suffix='_'+args.variant if args.variant else ''
    merged = json.loads((directory/f'merged{suffix}/merge.json').read_text())
    if merged['state'] != 'complete': raise ValueError('Merged cost is incomplete')
    xyz = np.load(directory/'reconstruction/map.npy',mmap_mode='r')
    cost = np.load(directory/f'merged{suffix}/cost.npy',mmap_mode='r')
    if len(xyz) != len(cost) or len(xyz) != merged['points']: raise ValueError('Geometry/cost point count mismatch')
    nodes = json.loads((directory/'nodes/nodes.json').read_text())['nodes']
    frames = json.loads((directory/'reconstruction/frames.json').read_text())
    odometry = json.loads((directory/'reconstruction/odometry.json').read_text())
    rows,_=display_rows(xyz,cost,voxel_m=.1,max_points=args.max_points)
    shown = np.asarray(xyz[rows])
    encoding_frame=merged.get('inference_frame','map')
    encoding_xyz=np.load(directory/f'node_cost{suffix}/world_xyz.npy',mmap_mode='r') if encoding_frame=='world' else xyz
    blocks = [dict(key=list(key),marker_id=i) for i,(key,_) in enumerate(partition(encoding_xyz,[4.,4.,3.]))]
    display_nodes=[dict(id=n['id'],position=n['body_pose']['position']) for n in nodes]
    config=yaml.safe_load((CONFIG_DIR / 'entry.rviz').read_text())
    config['Visualization Manager']['Global Options']['Fixed Frame']=encoding_frame
    for display in config['Visualization Manager']['Displays']:
        if display['Name']=='Existing manual goal':display.update(Enabled=False,Value=False)
    config['Visualization Manager']['Displays'].append(dict(Class='rviz_default_plugins/Path',Name='Recorded LIO trajectory',
        Enabled=True,Value=True,Topic={'Value':'/entry/trajectory','Depth':1,'History Policy':'Keep Last',
        'Reliability Policy':'Reliable','Durability Policy':'Transient Local'},
        **{'Line Style':'Lines','Line Width':.04,'Color':'80; 150; 255','Alpha':1.0,'Buffer Length':1}))
    low,high=np.percentile(encoding_xyz[rows],[1,99],axis=0)
    center=(low+high)/2
    view=config['Visualization Manager']['Views']['Current']
    view['Focal Point']=dict(zip(('X','Y','Z'),map(float,center)))
    view['Distance']=float(max(10.,np.linalg.norm(high-low)*1.25))
    view['Pitch']=.85
    view['Target Frame']=encoding_frame
    config_file=directory/'offline_view.rviz'
    config_file.write_text(yaml.safe_dump(config,sort_keys=False))
    lock=Path(f'/tmp/pure3d_entry_{os.getuid()}_domain_{int(os.environ.get("ROS_DOMAIN_ID","0"))}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init(args=[]);node=Node('entry_offline_view')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs={key:node.create_publisher(kind,topic,qos) for key,kind,topic in [
        ('cloud',PointCloud2,'/entry/rviz/cloud'),('cost',PointCloud2,'/entry/rviz/cost_cloud'),
        ('blocks',MarkerArray,'/entry/blocks'),('nodes',MarkerArray,'/entry/nodes'),
        ('path',PathMessage,'/entry/trajectory'),('status',String,'/entry/status')]}
    static=StaticTransformBroadcaster(node)
    stamp=node.get_clock().now().to_msg()
    static.sendTransform([stamped_transform('world','map',stamp,
        (np.array(frames['world_map_translation']),np.array(frames['world_map_quaternion_xyzw']))),
        stamped_transform('map','camera_init',stamp,(np.zeros(3),np.array([0.,0.,0.,1.])))])
    path=PathMessage();path.header.frame_id='map';path.header.stamp=stamp
    for pose in odometry:
        item=PoseStamped();item.header=path.header
        item.pose.position.x,item.pose.position.y,item.pose.position.z=map(float,pose['position'])
        item.pose.orientation.x,item.pose.orientation.y,item.pose.orientation.z,item.pose.orientation.w=map(float,pose['quaternion_xyzw'])
        path.poses.append(item)
    pubs['cloud'].publish(cloud_message(shown,stamp))
    pubs['cost'].publish(cloud_message(shown,stamp,cost[rows]))
    block_array=block_markers(blocks,[4.,4.,3.],stamp)
    for item in block_array.markers:item.header.frame_id=encoding_frame
    pubs['blocks'].publish(block_array)
    pubs['nodes'].publish(node_markers(display_nodes,stamp))
    pubs['path'].publish(path)
    status=dict(state='viewing_saved_result',frame_id='map',fixed_frame=encoding_frame,points=len(xyz),display_points=len(rows),
        nodes=len(nodes),blocks=len(blocks),display_voxel_m=.1,prediction=merged['prediction'],rviz_config=str(config_file))
    pubs['status'].publish(String(data=json.dumps(status)))
    atomic_json(directory/'view.json',status)
    viewer=subprocess.Popen(['rviz2','-d',str(config_file),'--ros-args','-r','__node:=entry_rviz',
        '-p','use_sim_time:=false'],start_new_session=True) if args.rviz else None
    node.get_logger().info(json.dumps(status))
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        lock.close()
