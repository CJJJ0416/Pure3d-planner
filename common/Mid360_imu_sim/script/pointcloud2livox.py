#!/usr/bin/env python3
"""ROS2 PointCloud2 -> Livox CustomMsg; preserve simulation stamps/offsets."""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from livox_ros_driver2.msg import CustomMsg, CustomPoint

class Converter(Node):
    def __init__(self):
        super().__init__('pointcloud2livox')
        input_topic=self.declare_parameter('input_topic','/scan').value
        output_topic=self.declare_parameter('output_topic','/livox/lidar2_converted').value
        self.pub=self.create_publisher(CustomMsg,output_topic,10)
        self.sub=self.create_subscription(PointCloud2,input_topic,self.convert,qos_profile_sensor_data)
        self.get_logger().info(f'{input_topic} -> {output_topic}; preserving source frame and simulation timestamp')

    def convert(self,cloud):
        result=CustomMsg();result.header=cloud.header
        result.timebase=cloud.header.stamp.sec*1_000_000_000+cloud.header.stamp.nanosec
        result.lidar_id=1
        names=[f.name for f in sorted(cloud.fields,key=lambda f:f.offset) if f.name in ('x','y','z','intensity','offset_time','line','tag','reflectivity')]
        if not {'x','y','z'}<=set(names):
            self.get_logger().error('PointCloud2 is missing xyz fields');return
        for values in point_cloud2.read_points(cloud,field_names=names,skip_nans=True):
            v=dict(zip(names,values))
            if not all(math.isfinite(v[k]) for k in ('x','y','z')):continue
            p=CustomPoint();p.x=float(v['x']);p.y=float(v['y']);p.z=float(v['z'])
            p.reflectivity=max(0,min(255,int(v.get('reflectivity',v.get('intensity',0)))))
            p.offset_time=max(0,min(2**32-1,int(v.get('offset_time',0))))
            p.line=max(0,min(255,int(v.get('line',0))));p.tag=max(0,min(255,int(v.get('tag',0x10))))
            result.points.append(p)
        result.point_num=len(result.points);self.pub.publish(result)

def main():
    rclpy.init();node=Converter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
