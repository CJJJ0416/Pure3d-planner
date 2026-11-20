"""View saved surface/DC comparison in an isolated ROS domain; publishes no commands."""
from common.paths import CONFIG_DIR
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--map',type=int,default=8209)
    p.add_argument('--query',type=int,default=23)
    p.add_argument('--rviz',action='store_true')
    args=p.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Path as PathMessage
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from common.io import atomic_json
    from common.visualization.display_sampling import display_rows
    from common.io import cloud_message
    from common.visualization.visualization import point

    folder=args.run/str(args.map)/'surface/repeat_1'
    query=folder/'queries'/f'{args.query:06d}'
    record=json.loads((query/'query.json').read_text())
    config=yaml.safe_load((CONFIG_DIR / 'navigation.rviz').read_text())
    manager=config['Visualization Manager'];manager['Global Options']['Fixed Frame']='world'
    for display in manager['Displays']:
        name=display.get('Name','')
        if name.startswith('SCAN ') or display['Class']=='rviz_default_plugins/InteractiveMarkers' or name in ('Global path labels','EDEN style blocks'):
            display.update(Enabled=False,Value=False)
        if name=='Surface A* cells - height':display.update(Enabled=True,Value=True)
        if name=='Traversability cost':
            display.update(Enabled=True,Value=True,Alpha=.4,Style='Points',**{'Size (Pixels)':1,'Use rainbow':False})
    manager['Displays'].append(dict(Class='rviz_default_plugins/Path',Name='DC GPMP reference - orange',Enabled=True,Value=True,
        Topic={'Value':'/entry/dc_gpmp_path','Depth':1,'History Policy':'Keep Last','Reliability Policy':'Reliable','Durability Policy':'Transient Local'},
        **{'Line Style':'Lines','Line Width':.04,'Color':'255; 165; 45','Alpha':1.,'Buffer Length':1,'Pose Style':'None'}))
    view=manager['Views']['Current'];center=(np.array(record['start_body'])+record['goal_body'])/2
    view.update(**{'Target Frame':'world','Distance':max(10.,float(np.linalg.norm(np.array(record['start_body'])-record['goal_body']))*2),
        'Pitch':.9,'Focal Point':dict(zip(('X','Y','Z'),map(float,center)))})
    config_path=args.run/f'surface_view_{args.map}_{args.query}.rviz'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    rclpy.init();node=Node('entry_surface_offline_view')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    messages=[];cloud_counts={}
    stamp=node.get_clock().now().to_msg()
    for name,topic in [('full_voxel','/entry/rviz/cost_cloud'),('surface','/entry/surface_cloud')]:
        xyz=np.load(folder/(name+'_xyz.npy'));cost=np.load(folder/(name+'_cost.npy'))
        rows,_=display_rows(xyz,cost,voxel_m=.1,max_points=200000)
        messages.append((node.create_publisher(PointCloud2,topic,qos),cloud_message(xyz[rows],stamp,cost[rows],frame='world')))
        cloud_counts[topic]=dict(input_points=len(xyz),display_points=len(rows))
    path_counts={}
    for name,source in [('astar',query/'astar_world.npy'),('gpmp',query/'gpmp_world.npy'),
        ('dc_gpmp',args.run/str(args.map)/'dc/repeat_1/queries'/f'{args.query:06d}'/'gpmp_world.npy')]:
        if not source.exists():continue
        path=PathMessage();path.header.frame_id='world';path.header.stamp=stamp
        xyz=np.load(source)
        for v in xyz:
            pose=PoseStamped();pose.header=path.header;pose.pose.position=point(v);pose.pose.orientation.w=1.;path.poses.append(pose)
        topic='/entry/'+name+'_path'
        messages.append((node.create_publisher(PathMessage,topic,qos),path));path_counts[topic]=len(xyz)
    import os
    report=dict(state='viewing_saved_surface_comparison',map_key=args.map,query_id=args.query,
        domain=int(os.environ.get('ROS_DOMAIN_ID','0')),clouds=cloud_counts,paths=path_counts,
        record_success=record['success'],actual_execution=False,traversability_policy='cost_only',
        config=str(config_path.resolve()))
    publisher=node.create_publisher(String,'/entry/surface_view/status',qos)
    messages.append((publisher,String(data=json.dumps(report))))
    def publish():
        for pub,msg in messages:pub.publish(msg)
    publish();timer=node.create_timer(1.,publish)
    viewer=subprocess.Popen(['rviz2','-d',str(config_path),'--ros-args','-r','__node:=entry_surface_compare_rviz','-p','use_sim_time:=false']) if args.rviz else None
    if viewer is not None:report['rviz_pid']=viewer.pid
    atomic_json(args.run/'view_status.json',report)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        timer.cancel();node.destroy_node()
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
