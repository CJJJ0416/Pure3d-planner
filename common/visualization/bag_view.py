"""Inspect matched raw/KNN/cost snapshots saved by the JEPLO replay observer."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from common.paths import CONFIG_DIR


def live_config(run):
    cfg = yaml.safe_load((CONFIG_DIR/'local_cost.rviz').read_text())
    manager = cfg['Visualization Manager']
    names = {'World grid','Box LIO voxels','Local traversability cost','KNN removed observations','Entry LIO box'}
    manager['Displays'] = [d for d in manager['Displays'] if d['Name'] in names]
    for d in manager['Displays']:
        if d['Name'] == 'KNN removed observations': d.update(Enabled=True, Value=True)
        if d['Name'] == 'Local traversability cost': d['Use rainbow'] = False
    source=yaml.safe_load((CONFIG_DIR/'local_cost.rviz').read_text())['Visualization Manager']['Displays']
    path=deepcopy(next(d for d in source if d.get('Class')=='rviz_default_plugins/Path'))
    path.update(Name='Faster-LIO body trajectory',Enabled=True,Value=True)
    path['Topic'].update(Value='/faster_lio/path', **{'Durability Policy':'Volatile','Reliability Policy':'Reliable'})
    path['Color']='50; 200; 255'
    manager['Displays'].append(path)
    manager['Global Options']['Fixed Frame'] = 'world'
    manager['Views']['Current'].update({'Target Frame':'body','Distance':10.,'Pitch':.65,'Yaw':.9,
                                       'Focal Point':{'X':0.,'Y':0.,'Z':0.}})
    cfg['Window Geometry'].pop('QMainWindow State', None)
    cfg['Window Geometry'].update(Width=1500, Height=900, X=10, Y=35)
    path = Path(run)/'live.rviz'; path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def load_snapshot(run, index):
    audit = json.loads((run/'audit.json').read_text())
    item = audit['snapshots'][index]
    with np.load(run/item['path']) as data:
        raw, filtered, cost = [np.ascontiguousarray(data[key]) for key in ('raw','filtered','cost')]
    dtype = np.dtype((np.void, 12))
    kept = np.isin(raw.view(dtype).ravel(), filtered.view(dtype).ravel())
    if int(kept.sum()) != len(filtered):
        raise ValueError('Saved filtered points are not an exact subset of their matched raw snapshot')
    return raw, filtered, cost, raw[~kept], item


def plot(run, index=-1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    raw, filtered, cost, removed, item = load_snapshot(run,index)
    cmap = LinearSegmentedColormap.from_list('cost', ['#14c83c','#ffe45b','#f02d23'])
    low, high = np.quantile(raw,.01,axis=0), np.quantile(raw,.99,axis=0)
    center = (low+high)/2; radius = max(high-low)/2+.15
    fig = plt.figure(figsize=(16,5.6),layout='constrained')
    for i,(xyz,title) in enumerate([(raw,f'Local voxels: {len(raw):,}'),(filtered,f'KNN retained: {len(filtered):,}'),(filtered,'Tiny traversability cost')]):
        ax = fig.add_subplot(1,3,i+1,projection='3d')
        rows = np.arange(len(xyz))
        if len(rows)>35000: rows=rows[np.linspace(0,len(rows)-1,35000,dtype=int)]
        scatter = ax.scatter(*xyz[rows].T,s=1.,c=cost[rows] if i==2 else xyz[rows,2],
            cmap=cmap if i==2 else 'viridis',vmin=0 if i==2 else low[2],vmax=1 if i==2 else high[2],
            linewidths=0,rasterized=True,depthshade=False)
        if i==1 and len(removed):
            ax.scatter(*removed.T,s=5,c='#d525df',linewidths=0,label=f'Removed: {len(removed):,}',depthshade=False)
            ax.legend(loc='upper right',fontsize=9)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
               zlim=(center[2]-radius,center[2]+radius),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title=title)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-125)
        if i==2: fig.colorbar(scatter,ax=ax,shrink=.62,label='cost [0, 1]')
    first=json.loads((run/'preflight.json').read_text())['/livox/lidar']['first_ns']
    label=Path(json.loads((run/'run.json').read_text())['bag']).name
    fig.suptitle(f'{label} | matched snapshot at {(item["stamp_ns"]-first)/1e9:.1f} s | gravity-aligned world',fontsize=14)
    path=run/'comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    return path


def view(run, index, cycle, rviz):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import Marker,MarkerArray
    from common.io import cloud_message,atomic_json
    rclpy.init();node=Node('entry_bag_snapshot_view')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publishers={k:node.create_publisher(PointCloud2,'/entry/bag_view/'+k,qos) for k in ('raw','filtered','cost','removed')}
    labels=node.create_publisher(MarkerArray,'/entry/bag_view/labels',qos)
    cfg=yaml.safe_load(live_config(run).read_text());manager=cfg['Visualization Manager']
    template=next(d for d in manager['Displays'] if d['Name']=='Local traversability cost')
    displays=[manager['Displays'][0]]
    for label,name in [('raw','LEFT - local voxels'),('filtered','MIDDLE - KNN retained'),('cost','RIGHT - Tiny cost'),('removed','KNN removed from local box')]:
        d=deepcopy(template);d.update(Name=name,Enabled=True,Value=True);d['Topic']['Value']='/entry/bag_view/'+label
        if label in ('raw','filtered'):d.update({'Color Transformer':'AxisColor','Autocompute Value Bounds':{'Value':True}})
        if label=='removed':d.update({'Color Transformer':'FlatColor','Color':'210; 40; 220','Size (Pixels)':5})
        displays.append(d)
    displays.append({'Class':'rviz_default_plugins/MarkerArray','Name':'Snapshot labels','Enabled':True,'Value':True,
        'Topic':{'Value':'/entry/bag_view/labels','Depth':1,'Reliability Policy':'Reliable','Durability Policy':'Transient Local'}})
    manager['Displays']=displays
    manager['Views']['Current'].update({'Target Frame':'world','Distance':24.,'Pitch':1.1,'Yaw':1.57,'Focal Point':{'X':0.,'Y':0.,'Z':0.}})
    config=run/'snapshots.rviz';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    audit=json.loads((run/'audit.json').read_text());number=len(audit['snapshots'])
    current=(index%number);last=-1;changed=time.monotonic();viewer=None
    try:
        while rclpy.ok():
            if current!=last:
                raw,filtered,cost,removed,item=load_snapshot(run,current)
                center=(raw.min(0)+raw.max(0))/2
                stamp=node.get_clock().now().to_msg();arr=MarkerArray()
                for pane,(key,xyz,values,shift) in enumerate([('raw',raw,None,-7.),('filtered',filtered,None,0.),('cost',filtered,cost,7.),('removed',removed,None,0.)]):
                    publishers[key].publish(cloud_message(xyz-center+[shift,0,0],stamp,values,frame='world'))
                    if key=='removed':continue
                    m=Marker();m.header.frame_id='world';m.header.stamp=stamp;m.ns='snapshot_labels';m.id=pane
                    m.type=Marker.TEXT_VIEW_FACING;m.action=Marker.ADD;m.pose.orientation.w=1.
                    m.pose.position.x=shift;m.pose.position.y=3.7;m.pose.position.z=1.
                    m.scale.z=.35;m.color.r=m.color.g=m.color.b=m.color.a=1.
                    m.text={'raw':f'Local voxels: {len(raw):,}','filtered':f'KNN: {len(filtered):,} / removed {len(removed):,}','cost':'Tiny cost: green 0 -> red 1'}[key]
                    arr.markers.append(m)
                labels.publish(arr);last=current
                atomic_json(run/'view.json',dict(state='viewing',snapshot=current,source=item,display_only_x_offsets=[-7.,0.,7.]))
                if rviz and viewer is None:viewer=subprocess.Popen(['rviz2','-d',str(config),'--ros-args','-p','use_sim_time:=false'])
            if viewer is not None and viewer.poll() is not None:break
            rclpy.spin_once(node,timeout_sec=.1)
            if cycle>0 and time.monotonic()-changed>=cycle:current=(current+1)%number;changed=time.monotonic()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if viewer is not None and viewer.poll() is None:viewer.terminate();viewer.wait(timeout=10)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--snapshot',type=int,default=-1)
    parser.add_argument('--plot',action='store_true');parser.add_argument('--rviz',action='store_true')
    parser.add_argument('--cycle-seconds',type=float,default=0.)
    args=parser.parse_args();run=args.run.resolve()
    if args.plot:print(plot(run,args.snapshot))
    else:view(run,args.snapshot,args.cycle_seconds,args.rviz)


if __name__=='__main__':main()
