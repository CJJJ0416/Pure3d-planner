"""读取训练或诊断用的 Tomogram XYZ。部署推理使用建筑自身 XYZ，不需要此入口。"""
import argparse
from pathlib import Path

import numpy as np

from common.pointcloud_io import pointcloud2_xyz, read_pcd_xyz
from utils.pointcloud import ROOT, write_json


def save_queries(path, coord, source, frame_id=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coord)
    write_json(path.with_suffix(".json"), dict(
        source=source, points=len(coord), shape=list(coord.shape), fields=["x", "y", "z"],
        frame_id=frame_id, coordinates="unchanged source XYZ in metres, original point order",
    ))
    print(f"{path}: {len(coord):,} query positions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, help="读取已保存的 Tomogram PCD；省略时订阅话题")
    parser.add_argument("--topic", default="/tomogram")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tomogram_queries/query_coord.npy")
    args = parser.parse_args()
    if args.pcd:
        save_queries(args.output, read_pcd_xyz(args.pcd), str(args.pcd.resolve()))
        return

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2

    rclpy.init()
    node = Node("tomogram_query_xyz")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    received = False

    def callback(message):
        nonlocal received
        save_queries(args.output, pointcloud2_xyz(message), args.topic, message.header.frame_id)
        received = True

    subscription = node.create_subscription(PointCloud2, args.topic, callback, qos)
    print(f"Waiting for {args.topic} ...", flush=True)
    while not received:
        rclpy.spin_once(node)
    node.destroy_subscription(subscription)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
