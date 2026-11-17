"""Downsampled RViz topics, independent of full inference/planning clouds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from Entry.utils.io import atomic_json
from Entry.rendering.display_sampling import display_rows
from Entry.utils.io import cloud_message


class DisplayPublisher:
    """One worker, coalesced pending inputs, no full-map work in ROS callbacks."""
    def __init__(self,node,voxel_m=.1,max_points=200000,hz=1.,status_path=None,topic_prefix='/entry/rviz'):
        if not np.isfinite([voxel_m,hz]).all() or voxel_m<=0 or hz<=0 or max_points<=0:raise ValueError('Invalid display settings')
        self.node=node;self.voxel_m=voxel_m;self.max_points=max_points;self.period=1/hz
        self.pending={};self.queued_at={};self.future=None;self.last_started={};self.stats={};self.status_path=status_path
        self.generations={};self.future_generation=None
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='entry_rviz_sampling')
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publishers={name:node.create_publisher(PointCloud2,topic_prefix+'/'+name,qos) for name in ['cloud','cost_cloud']}
        self.status_pub=node.create_publisher(String,topic_prefix+'/status',qos)

    def queue(self,name,xyz,stamp,cost=None,frame='map'):
        self.pending[name]=(xyz,stamp,cost,frame)
        self.queued_at[name]=time.perf_counter()

    def queue_message(self,name,message):
        self.pending[name]=message;self.queued_at[name]=time.perf_counter()

    def invalidate(self,name):
        self.generations[name]=self.generations.get(name,0)+1
        self.pending.pop(name,None);self.queued_at.pop(name,None);self.last_started.pop(name,None)

    def compute(self,name,source,queued_at=None):
        started=time.perf_counter()
        if isinstance(source,PointCloud2):
            from Entry.utils.io import cloud_xyz
            fields={f.name:f for f in source.fields};packed=(source.height==1 and not source.is_bigendian and
                all(k in fields and fields[k].datatype==7 and fields[k].count==1 and fields[k].offset==4*i for i,k in enumerate(('x','y','z'))))
            xyz=(np.ndarray((source.width,3),dtype='<f4',buffer=source.data,strides=(source.point_step,4)) if packed else cloud_xyz(source))
            stamp=source.header.stamp;frame=source.header.frame_id;cost=None
            if name=='cost_cloud':
                field=next(f for f in source.fields if f.name=='trav_cost')
                if field.datatype!=7 or field.count!=1:raise ValueError('Expected float32 trav_cost')
                dtype=np.dtype(dict(names=['cost'],formats=[('>' if source.is_bigendian else '<')+'f4'],offsets=[field.offset],itemsize=source.point_step))
                array=np.ndarray((source.height,source.width),dtype=dtype,buffer=source.data,strides=(source.row_step,source.point_step))
                cost=array['cost'][0] if source.height==1 else array['cost'].ravel()
        else:xyz,stamp,cost,frame=source
        rows,voxel_points=display_rows(xyz,cost,self.voxel_m,self.max_points)
        message=cloud_message(xyz[rows],stamp,None if cost is None else cost[rows]);message.header.frame_id=frame
        stats=dict(input_points=len(xyz),voxel_points=voxel_points,published_points=len(rows),voxel_m=self.voxel_m,
            max_points=self.max_points,processing_s=time.perf_counter()-started,frame_id=frame,
            source_stamp=stamp.sec+stamp.nanosec*1e-9,_received_wall=queued_at,
            cost_aggregation='maximum cost original point per voxel' if cost is not None else None)
        return name,message,stats

    def poll(self):
        if self.future is not None and self.future.done():
            future,self.future=self.future,None
            try:
                name,message,stats=future.result()
                if self.future_generation != self.generations.get(name,0):return
                self.publishers[name].publish(message)
                received=stats.pop('_received_wall',None)
                if received is not None:stats['receive_to_publish_s']=time.perf_counter()-received
                self.stats[name]=stats
                self.status_pub.publish(String(data=json.dumps(self.stats)))
                if self.status_path is not None:atomic_json(self.status_path,self.stats)
            except Exception as error:self.node.get_logger().error(f'RViz downsampling: {error}')
        if self.future is None:
            now=time.monotonic()
            for name in ['cost_cloud','cloud']:
                if name=='cloud' and self.publishers[name].get_subscription_count()==0:continue
                if name in self.pending and now-self.last_started.get(name,0)>=self.period:
                    source=self.pending.pop(name);self.last_started[name]=now
                    self.future_generation=self.generations.get(name,0)
                    self.future=self.pool.submit(self.compute,name,source,self.queued_at.pop(name,None));break

    def close(self):self.pending.clear();self.pool.shutdown(wait=True)


class DisplayRelay(Node):
    def __init__(self,args):
        super().__init__('entry_rviz_downsample')
        self.display=DisplayPublisher(self,args.voxel,args.max_points,args.hz,args.status,getattr(args,'topic_prefix','/entry/rviz'))
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cost_sub=self.create_subscription(PointCloud2,'/entry/cost_cloud',lambda msg:self.display.queue_message('cost_cloud',msg),qos)
        self.geometry_sub=None;self.source_qos=qos
        self.background_clock=Clock(clock_type=ClockType.STEADY_TIME)
        self.display_timer=self.create_timer(.1,self.poll,clock=self.background_clock)

    def poll(self):
        # Geometry is disabled in the default RViz. Avoid transporting another
        # full cumulative map unless somebody actually displays that layer.
        needed=self.display.publishers['cloud'].get_subscription_count()>0
        if needed and self.geometry_sub is None:
            self.geometry_sub=self.create_subscription(PointCloud2,'/entry/cloud',lambda msg:self.display.queue_message('cloud',msg),self.source_qos)
        elif not needed and self.geometry_sub is not None:
            self.destroy_subscription(self.geometry_sub);self.geometry_sub=None;self.display.pending.pop('cloud',None)
        self.display.poll()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--voxel',type=float,default=.1)
    parser.add_argument('--max-points',type=int,default=200000);parser.add_argument('--hz',type=float,default=1.)
    parser.add_argument('--status',type=Path)
    parser.add_argument('--topic-prefix',default='/entry/rviz')
    args=parser.parse_args()
    rclpy.init(args=[]);node=DisplayRelay(args)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.display.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
