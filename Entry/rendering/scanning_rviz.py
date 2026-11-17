"""RViz playback of Tiny/Small costs from Entry's causal sliding-box replay."""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from PyQt5 import QtCore, QtWidgets
import yaml
from Entry.utils.io import atomic_json
from Entry.rendering.display_sampling import display_rows
from Entry.utils.io import cloud_message
from Entry.rendering.visualization import marker, point
from planner.visualization.rviz import cloud_display, marker_display
from utils.pointcloud import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    args=parser.parse_args()
    replay=args.replay.resolve();output=replay/'rviz';output.mkdir(exist_ok=True)
    index=read_json(replay/'index.json');records=index['records']
    times=np.array([row['stamp_s'] for row in records])
    root=Path(index['source_root']);trajectory=dict(np.load(replay/'trajectory.npz'))
    raw=np.load(root/index['profile']/'arrays_000/coord.npy',mmap_mode='r')
    background_rows,_=display_rows(raw,voxel_m=.1,max_points=200000)
    background_rows=np.sort(background_rows)
    rclpy.init();node=Node('entry_scanning_cost_playback')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
    prefix='/pure3d/cost_replay'
    cost_pub=node.create_publisher(PointCloud2,prefix+'/cost',qos)
    raw_pub=node.create_publisher(PointCloud2,prefix+'/observed',qos)
    marks_pub=node.create_publisher(MarkerArray,prefix+'/markers',qos)
    tf=TransformBroadcaster(node)
    cost_display=cloud_display('Tiny predicted cost | all local points',prefix+'/cost')
    cost_display.update({'Alpha':1.,'Size (Pixels)':4,'Selectable':False})
    raw_display=cloud_display('Past LIO observations | display 10 cm',prefix+'/observed',True,True)
    raw_display.update({'Alpha':.22,'Size (Pixels)':1,'Selectable':False})
    camera={'Class':'rviz_default_plugins/Orbit','Name':'Follow moving sensor','Distance':12.,
        'Pitch':.65,'Yaw':.85,'Target Frame':'cost_replay_sensor','Near Clip Distance':.01,
        'Focal Point':{'X':0.,'Y':0.,'Z':-.3}}
    config={'Panels':[{'Class':'rviz_common/Displays','Name':'Displays'},
                      {'Class':'rviz_common/Views','Name':'Views'}],
        'Visualization Manager':{'Class':'','Enabled':True,
            'Displays':[cost_display,raw_display,marker_display('ROI, context, pose and trajectory',prefix+'/markers')],
            'Global Options':{'Fixed Frame':'map','Background Color':'24; 29; 38','Frame Rate':30},
            'Tools':[{'Class':'rviz_default_plugins/'+s} for s in ['Interact','MoveCamera','Select','FocusCamera']],
            'Views':{'Current':camera,'Saved':[camera,{**camera,'Name':'Top view','Pitch':1.55}]}},
        'Window Geometry':{'Width':1550,'Height':950},'Preferences':{'PromptSaveOnExit':False}}
    cfg_path=output/'dynamic_cost.rviz';cfg_path.write_text(yaml.safe_dump(config,sort_keys=False))
    log=(output/'rviz.log').open('w')
    viewer=subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(cfg_path),
        '--ros-args','-r','__node:=entry_scanning_cost_rviz'],stdout=log,stderr=subprocess.STDOUT)
    (output/'viewer.pid').write_text(str(viewer.pid)+'\n')
    (output/'publisher.pid').write_text(str(os.getpid())+'\n')
    app=QtWidgets.QApplication([])
    panel=QtWidgets.QWidget();panel.setWindowTitle('Tiny cost | 55-85 s playback')
    layout=QtWidgets.QVBoxLayout(panel)
    label=QtWidgets.QLabel();layout.addWidget(label)
    slider=QtWidgets.QSlider(QtCore.Qt.Horizontal);slider.setRange(round(times[0]*1000),round(times[-1]*1000))
    slider.setValue(round(times[0]*1000));layout.addWidget(slider)
    play=QtWidgets.QPushButton('Pause');layout.addWidget(play)
    restart=QtWidgets.QPushButton('Restart');layout.addWidget(restart)
    loop=QtWidgets.QCheckBox('Loop this segment');loop.setChecked(True);layout.addWidget(loop)
    halo=QtWidgets.QCheckBox('Show retained context bounds');layout.addWidget(halo)
    speed=QtWidgets.QComboBox();speed.addItems(['0.5x','1x','2x']);speed.setCurrentIndex(1);layout.addWidget(speed)
    state=dict(seconds=float(times[0]),playing=True,last=-1,dirty=True)
    slider.valueChanged.connect(lambda value:state.update(seconds=value/1000,dirty=True))
    def toggle():
        state['playing']=not state['playing'];play.setText('Pause' if state['playing'] else 'Play')
    play.clicked.connect(toggle)
    restart.clicked.connect(lambda:slider.setValue(round(times[0]*1000)))
    halo.toggled.connect(lambda _:state.update(dirty=True))
    panel.resize(425,225);panel.move(1150,720);panel.show()
    corners=np.array(list(itertools.product([0,1],repeat=3)))
    edges=[(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in [1,2,4]]
    current_position=trajectory['position'][0]
    tick=time.perf_counter();subscription_checked=False

    def publish(chosen):
        nonlocal current_position
        row=records[chosen];stamp=node.get_clock().now().to_msg()
        with np.load(replay/row['path']) as data:
            current_position=data['position'].copy()
            cost_pub.publish(cloud_message(data['xyz'],stamp,data['cost']))
            visible=background_rows[background_rows<int(data['raw_point_count'])]
            raw_pub.publish(cloud_message(raw[visible],stamp))
            markers=MarkerArray()
            for number,(lo,hi,color) in enumerate([
                (data['lower'],data['upper'],[1.,.65,.03]),
                (data['context_lower'],data['context_upper'],[.5,.6,.7])]):
                outline=marker('bounds',number,Marker.LINE_LIST,stamp)
                outline.scale.x=.035 if number==0 else .015
                outline.color.r,outline.color.g,outline.color.b=map(float,color)
                outline.color.a=1. if number==0 or halo.isChecked() else 0.
                xyz=lo+corners*(hi-lo)
                outline.points=[point(xyz[i]) for edge in edges for i in edge]
                markers.markers.append(outline)
            sphere=marker('sensor',0,Marker.SPHERE,stamp);sphere.pose.position=point(current_position)
            sphere.scale.x=sphere.scale.y=sphere.scale.z=.16
            sphere.color.g=.7;sphere.color.b=1.;sphere.color.a=1.;markers.markers.append(sphere)
            heading=marker('sensor',1,Marker.ARROW,stamp)
            heading.points=[point(current_position),point(current_position+.8*data['rotation'][:,0])]
            heading.scale.x=.04;heading.scale.y=.1;heading.scale.z=.1
            heading.color.g=.7;heading.color.b=1.;heading.color.a=1.;markers.markers.append(heading)
            path=marker('trajectory',0,Marker.LINE_STRIP,stamp);path.scale.x=.035
            path.color.g=.7;path.color.b=1.;path.color.a=1.
            past=trajectory['position'][trajectory['stamp_s']<=row['stamp_s']+1e-6]
            path.points=[point(p) for p in past];markers.markers.append(path)
            hud=marker('timing',0,Marker.TEXT_VIEW_FACING,stamp)
            hud.pose.position=point(current_position+[0.,0.,3.])
            hud.scale.z=.22;hud.color.r=hud.color.g=hud.color.b=hud.color.a=1.
            hud.text=(f"Tiny best49 | {row['stamp_s']:.1f} s | ROI 6x6x5 m\n"
                f"{row['points']:,} cost points | {row['snapshot_to_cost_s']*1000:.0f} ms compute\n"
                f"Entry stable_core | green 0 -> red 1")
            markers.markers.append(hud);marks_pub.publish(markers)
            atomic_json(output/'state.json',dict(pid=os.getpid(),viewer_pid=viewer.pid,status='open',
                elapsed_s=row['stamp_s'],snapshot=chosen,model=index['model'],
                cost_points=row['points'],background_display_points=len(visible),
                raw_observation_points=int(data['raw_point_count']),lower=data['lower'].tolist(),
                upper=data['upper'].tolist(),snapshot_to_cost_s=row['snapshot_to_cost_s'],
                playback='saved actual Tiny inference snapshots, only causal observations used'))
        label.setText(f"{root.name} | {row['stamp_s']:.2f} / {times[-1]:.2f} s\n"
            f"Tiny best49 | {row['points']:,} local cost points\n"
            f"Compute {row['snapshot_to_cost_s']*1000:.1f} ms | encoder calls {row['encoder_calls']}\n"
            'Actual Entry inference snapshots. Orange: perception box.')

    def update():
        nonlocal tick,subscription_checked
        now=time.perf_counter();dt=now-tick;tick=now
        if viewer.poll() is not None:
            app.quit();return
        if state['playing'] and not slider.isSliderDown():
            value=state['seconds']+dt*[.5,1.,2.][speed.currentIndex()]
            if value>times[-1]:
                if loop.isChecked():value=float(times[0])
                else:value=float(times[-1]);state['playing']=False;play.setText('Play')
            slider.blockSignals(True);slider.setValue(round(value*1000));slider.blockSignals(False)
            state.update(seconds=value,dirty=True)
        chosen=max(0,int(np.searchsorted(times,state['seconds'],side='right'))-1)
        if chosen!=state['last'] or (state['dirty'] and not state['playing']):
            publish(chosen);state['last']=chosen;state['dirty']=False
        transform=TransformStamped();transform.header.frame_id='map'
        transform.header.stamp=node.get_clock().now().to_msg();transform.child_frame_id='cost_replay_sensor'
        transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z=map(float,current_position)
        transform.transform.rotation.w=1.;tf.sendTransform(transform)
        rclpy.spin_once(node,timeout_sec=0.)
        if not subscription_checked and all(p.get_subscription_count() for p in [cost_pub,raw_pub,marks_pub]):
            atomic_json(output/'display_check.json',dict(all_layers_subscribed=True,cost=cost_pub.get_subscription_count(),
                observed=raw_pub.get_subscription_count(),markers=marks_pub.get_subscription_count()))
            subscription_checked=True

    timer=QtCore.QTimer();timer.timeout.connect(update);timer.start(33)
    print('RVIZ_DYNAMIC_COST_READY',viewer.pid,flush=True)
    try:app.exec_()
    finally:
        timer.stop();node.destroy_node();rclpy.shutdown();log.close()


if __name__=='__main__':
    main()
