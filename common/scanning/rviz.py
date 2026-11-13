"""ROS-only scored point-cloud views. No thresholding is applied to model input."""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from common.pointcloud_io import read_pcd_xyz


class Publisher(Node):
    def __init__(self, root, profile, overlay=False):
        super().__init__('pure3d_scan_dataset_viewer')
        folder = root/profile
        clean = read_pcd_xyz(root/'clean.pcd')
        if overlay:
            xyz = read_pcd_xyz(folder/'rough.pcd')
            views = dict(rough=(xyz, np.ones(len(xyz))), clean=(clean, np.ones(len(clean))))
        else:
            arrays = dict(np.load(folder/'attributes.npz'))
            xyz, target = arrays['coord'], arrays['keep_target']
            roles = arrays['source_role']
            views = dict(rough=(xyz, target), truth_fused=(arrays['truth_coord'], np.ones(len(xyz))),
                clean=(clean, np.ones(len(clean))),
                dynamic=(xyz[roles==7], target[roles==7]),
                edge_returns=(xyz[arrays['edge_mixed']], target[arrays['edge_mixed']]),
                static_controls=(xyz[roles==5], target[roles==5]))
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.publishers_, self.messages = [], []
        for name, (points, score) in views.items():
            values = np.column_stack([points, score]).astype('<f4')
            msg = PointCloud2(header=Header(frame_id='map'), height=1, width=len(points),
                fields=[PointField(name=key, offset=4*i, datatype=PointField.FLOAT32, count=1)
                        for i, key in enumerate(['x','y','z','keep_target'])],
                is_bigendian=False, point_step=16, row_step=16*len(points), data=values.tobytes(), is_dense=True)
            publisher = self.create_publisher(PointCloud2, '/pure3d/scanning/'+name, qos)
            publisher.publish(msg)
            self.publishers_.append(publisher)
            self.messages.append(msg)
            print(name, len(points), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', default='coverage')
    parser.add_argument('--overlay', action='store_true', help='Publish only complete scan and Mesh-sampled GT for a constant-colour overlay')
    args = parser.parse_args()
    rclpy.init()
    node = Publisher(args.root, args.profile, args.overlay)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
