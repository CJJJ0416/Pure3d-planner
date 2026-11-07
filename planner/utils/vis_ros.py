from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def traj2ros(traj, stamp=None):
    path_msg = Path()
    path_msg.header.frame_id = "map"
    # ROS2 消息用原生 float，并可传入节点时钟；原路径坐标/单位四元数保持不变。
    if stamp is not None:
        path_msg.header.stamp = stamp

    for waypoint in traj:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = path_msg.header.stamp
        pose.pose.position.x = float(waypoint[0])
        pose.pose.position.y = float(waypoint[1])
        pose.pose.position.z = float(waypoint[2])
        pose.pose.orientation.w = 1.0
        path_msg.poses.append(pose)

    return path_msg
