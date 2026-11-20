"""Convert the official ROS1 Avia bag to ROS2 Humble without altering sensor payloads.

The conversion process uses rosbags from the workspace cache. A separate system
Python process writes with the installed ROS2 Humble storage implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
FRAME = struct.Struct('<IQI')


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        part = stream.read(size-len(result))
        if not part: raise EOFError('Truncated conversion pipe')
        result.extend(part)
    return bytes(result)


def write_native(destination):
    import rosbag2_py
    stream = sys.stdin.buffer
    descriptions = json.loads(stream.readline())
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(destination),storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('',''))
    for item in descriptions:
        writer.create_topic(rosbag2_py.TopicMetadata(name=item['topic'],type=item['msgtype'],serialization_format='cdr'))
    while True:
        header = stream.read(FRAME.size)
        if not header: break
        if len(header) < FRAME.size: header += read_exact(stream,FRAME.size-len(header))
        idx,stamp,size = FRAME.unpack(header)
        writer.write(descriptions[idx]['topic'],read_exact(stream,size),stamp)
    del writer


def convert(source,destination):
    sys.path.insert(0,str(ROOT/'.cache/bag_tools'))
    import numpy as np
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores,get_types_from_msg,get_typestore
    from rosbags.convert.converter import generate_message_converter
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    source_store=get_typestore(Stores.ROS1_NOETIC)
    dest_store=get_typestore(Stores.ROS2_HUMBLE)
    with Reader(source) as reader:
        connections=[c for c in reader.connections if c.topic in ('/livox/lidar','/livox/imu')]
        if {c.topic for c in connections}!={'/livox/lidar','/livox/imu'}:raise ValueError('Official sensor topics missing')
        for c in connections:
            source_store.register(get_types_from_msg(c.msgdef.data,c.msgtype))
            if c.msgtype.startswith('livox_ros_driver/'):
                definitions=get_types_from_msg(c.msgdef.data.replace('livox_ros_driver/','livox_ros_driver2/'),
                    c.msgtype.replace('livox_ros_driver/','livox_ros_driver2/'))
                definitions.pop('std_msgs/msg/Header',None)
                dest_store.register(definitions)
        imu=[c for c in connections if c.topic=='/livox/imu']
        offsets=[]
        for _,stamp,data in reader.messages(connections=imu):
            sec,ns=struct.unpack_from('<II',data,4)
            offsets.append(stamp-sec*10**9-ns)
        # Preserve event ordering and every relative record interval. Only translate
        # the host recording epoch onto the native sensor clock (seconds since boot).
        epoch_offset=int(np.sort(np.asarray(offsets,dtype=np.int64))[len(offsets)//2])
        descriptions=[];converters={};indices={};cache={}
        for c in connections:
            name=c.msgtype.replace('livox_ros_driver/','livox_ros_driver2/')
            indices[c.id]=len(descriptions);descriptions.append(dict(topic=c.topic,msgtype=name,source_msgtype=c.msgtype))
            converters[c.id]=generate_message_converter(source_store,dest_store,c.msgtype,name,cache,src_is2=False,dst_is2=True)
        child=subprocess.Popen(['/usr/bin/python3',str(Path(__file__).resolve()),'--writer','--destination',str(destination)],stdin=subprocess.PIPE)
        counts={c.topic:0 for c in connections};samples=[];started=time.monotonic();last=started
        try:
            child.stdin.write((json.dumps(descriptions)+'\n').encode())
            for connection,stamp,data in reader.messages(connections=connections):
                converted=converters[connection.id](data)
                idx=indices[connection.id]
                child.stdin.write(FRAME.pack(idx,stamp-epoch_offset,len(converted)))
                child.stdin.write(converted)
                counts[connection.topic]+=1
                if counts[connection.topic] in (1,1000,2000):
                    old=source_store.deserialize_ros1(data,connection.msgtype)
                    new=dest_store.deserialize_cdr(converted,descriptions[idx]['msgtype'])
                    assert (old.header.stamp.sec,old.header.stamp.nanosec)==(new.header.stamp.sec,new.header.stamp.nanosec)
                    if connection.topic=='/livox/lidar':
                        assert old.timebase==new.timebase and old.point_num==new.point_num
                        for a,b in zip(old.points,new.points,strict=True):
                            assert all(getattr(a,key)==getattr(b,key) for key in ('offset_time','x','y','z','reflectivity','tag','line'))
                    samples.append(dict(topic=connection.topic,message=counts[connection.topic],sensor_fields_equal=True))
                if time.monotonic()-last>10:
                    print('Converted',counts,'elapsed',round(time.monotonic()-started,1),'s',flush=True);last=time.monotonic()
            child.stdin.close()
            if child.wait(timeout=30)!=0:raise RuntimeError('ROS2 native writer failed')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
        if counts!={c.topic:c.msgcount for c in connections}:raise ValueError('Message count mismatch')
    report=dict(source=str(source),destination=str(destination),writer='installed ROS2 Humble rosbag2_py SQLite3',
        counts=counts,type_rename={'livox_ros_driver/msg/CustomMsg':'livox_ros_driver2/msg/CustomMsg'},
        record_time_translation_ns=-epoch_offset,relative_record_intervals_preserved=True,
        sensor_header_stamps_preserved=True,lidar_timebase_and_point_offset_times_preserved=True,
        reason='Official bag record time uses Unix host epoch; sensor headers use seconds since sensor boot',
        checked_samples=samples)
    (destination/'conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--writer',action='store_true');args=parser.parse_args()
    if args.writer:write_native(args.destination.resolve())
    else:
        if args.source is None:parser.error('--source is required')
        convert(args.source.resolve(),args.destination.resolve())


if __name__=='__main__':main()
