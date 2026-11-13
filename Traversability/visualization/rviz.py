"""用 RViz 显示完整预测点云，或已有 Tomogram 监督与原生切片。

默认发布全部条件的归一化代价（0–1）；--raw-cost 时在发布阶段乘回 cost_scale
（通常为 50），并把 RViz 色标同步为 0–50，便于对照归一化前的代价。
"""
import argparse
import os
from pathlib import Path
import pickle
import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from common.pointcloud_io import pcd_to_pointcloud2, read_pcd_xyz
from utils.pointcloud import ROOT, read_json


def layer_message(xy, height, cost, header):
    """把一张原生水平切片转换为 XYZ + trav_cost 点云消息。"""
    # 展开 XY 网格，拼接每层 ground 高度与常量/原始代价
    values = np.column_stack((xy, height.ravel(), np.broadcast_to(cost, height.shape).ravel()))
    # 过滤 NaN/Inf（无有效 ground 的单元不发布）
    values = values[np.isfinite(values).all(1)].astype("<f4")
    return PointCloud2(
        header=header, height=1, width=len(values),
        fields=[PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(["x", "y", "z", "trav_cost"])],
        is_bigendian=False, point_step=16, row_step=16 * len(values),
        is_dense=True, data=values.tobytes(),
    )


def cloud_display(name, topic, enabled=False, flat=False, size=.085, cost_max=1.0):
    """生成 RViz2 PointCloud2 显示项配置。

    flat=True 时用固定颜色显示建筑几何；否则按 trav_cost 强度着色，
    绿色低代价、红色高代价，色标上限为 cost_max。
    """
    return {
        "Class": "rviz_default_plugins/PointCloud2", "Name": name,
        "Enabled": enabled, "Value": enabled, "Alpha": .7 if flat else 1.0,
        "Position Transformer": "XYZ", "Color Transformer": "FlatColor" if flat else "Intensity",
        "Color": "150; 170; 190", "Channel Name": "trav_cost",
        "Autocompute Intensity Bounds": False, "Min Intensity": 0.0, "Max Intensity": cost_max,
        "Use rainbow": False, "Min Color": "20; 200; 60", "Max Color": "240; 45; 35",
        "Style": "Points" if flat else "Flat Squares", "Size (Pixels)": 1 if flat else 3,
        "Size (m)": size, "Decay Time": 0, "Use Fixed Frame": True, "Selectable": True,
        "Topic": {"Value": topic, "Depth": 1, "History Policy": "Keep Last",
                  "Reliability Policy": "Reliable", "Durability Policy": "Transient Local",
                  "Filter size": 10},
    }


def rviz_config(scene, teacher, raw_cost=False):
    """生成 RViz 配置：全部条件代价点云、建筑几何和 nominal 原生层。"""
    profiles = teacher["profiles"]
    # raw_cost=True 时色标上限使用教师原始代价尺度，否则为归一化后的 1.0
    cost_max = teacher["cost_scale"] if raw_cost else 1.0
    mode = f"Cost 0-{cost_max:g} | p={teacher['teacher_parameters']['trav']['adjust_cost_p']}"
    # 默认只启用第一组条件（nominal），其余在 RViz 中手动切换
    displays = [cloud_display(f"{mode} | {p['id']}", f"/pure3d/tomogram/{p['id']}",
                              enabled=i == 0, cost_max=cost_max)
                for i, p in enumerate(profiles)]
    displays.append(cloud_display("Building PCD | geometry", "/pure3d/building", flat=True))
    # nominal 的原生 ground/ceiling 切片放在可折叠分组中
    layers = []
    for layer in range(teacher["shape"][0]):
        layers.append(cloud_display(f"Ground {layer:02d}", f"/pure3d/nominal/layer_G_{layer}", cost_max=cost_max))
        layers.append(cloud_display(f"Ceiling {layer:02d}", f"/pure3d/nominal/layer_C_{layer}", flat=True))
    displays.append({"Class": "rviz_common/Group", "Name": "Nominal native layers",
                     "Enabled": True, "Displays": layers})
    return view_config(scene, displays)


def view_config(scene, displays):
    """保持地图原点坐标，根据实际建筑包围盒设置初始相机。"""
    # 根据场景包围盒设置初始相机距离与焦点，使整栋建筑可见
    bounds = np.asarray(scene["bounds_world"])
    center = bounds.mean(0)
    return {
        "Panels": [{"Class": "rviz_common/Displays", "Name": "Displays"},
                   {"Class": "rviz_common/Views", "Name": "Views"}],
        "Visualization Manager": {
            "Class": "", "Enabled": True, "Displays": displays,
            "Global Options": {"Fixed Frame": scene["frame_id"], "Background Color": "24; 29; 38",
                               "Frame Rate": 30},
            "Tools": [{"Class": f"rviz_default_plugins/{tool}"}
                      for tool in ["Interact", "MoveCamera", "Select", "FocusCamera"]],
            "Views": {"Current": {
                "Class": "rviz_default_plugins/Orbit", "Name": "Whole building",
                "Distance": float(np.max(bounds[1, :2] - bounds[0, :2]) * 1.35),
                "Pitch": .6, "Yaw": .8, "Target Frame": "<Fixed Frame>", "Near Clip Distance": .01,
                "Focal Point": dict(zip(["X", "Y", "Z"], map(float, center))),
            }},
        },
        "Window Geometry": {"Width": 1440, "Height": 980},
        "Preferences": {"PromptSaveOnExit": False},
    }


class PredictionViewer(Node):
    """一次发布完整预测与原始建筑，不降采样、不分层或重新赋色。"""
    def __init__(self, prediction, metadata, planning=None, query=None):
        super().__init__("pure3d_prediction_clouds")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        header = Header(frame_id=metadata["coordinate_frame"], stamp=self.get_clock().now().to_msg())
        self.static_tf = StaticTransformBroadcaster(self)
        transform = TransformStamped(header=Header(frame_id="world", stamp=header.stamp),
                                     child_frame_id=header.frame_id)
        transform.transform.rotation.w = 1.0
        self.static_tf.sendTransform(transform)
        self.cloud_publishers = []
        clouds = [("/pure3d/prediction", prediction)]
        if query is None:
            clouds.append(("/pure3d/prediction_source", metadata["source_pcd"]))
        for topic, path in clouds:
            message = pcd_to_pointcloud2(path, header)
            publisher = self.create_publisher(PointCloud2, topic, qos)
            publisher.publish(message)
            self.cloud_publishers.append(publisher)
            print(f"{topic}: {message.width:,} points | frame={header.frame_id}", flush=True)
        self.frame_id = metadata['coordinate_frame']
        self.path_publishers = {
            name: self.create_publisher(RosPath, f'/pure3d/planning/{name}', qos)
            for name in ('astar', 'gpmp')
        }
        self.endpoint_publisher = self.create_publisher(MarkerArray, '/pure3d/planning/endpoints', qos)
        if planning is not None:
            self.publish_planning(planning)
        if query is not None:
            self.publish_query(query)

    def publish_query(self, query):
        """直接显示输入查询坐标，支持 A* 未找到路径时检查起终点。"""
        header = Header(frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        markers = []
        for name, key, color in [
            ('START', 'start', (0., 1., 1.)), ('GOAL', 'goal', (1., .1, .8)),
        ]:
            x, y, z = map(float, query[key])
            top_z = query['marker_top_z']
            sphere = Marker(header=header, ns=name, id=0, type=Marker.SPHERE, action=Marker.ADD)
            sphere.pose.position = Point(x=x, y=y, z=z)
            sphere.pose.orientation.w = 1.
            sphere.scale.x = sphere.scale.y = sphere.scale.z = .5
            sphere.color.r, sphere.color.g, sphere.color.b = color
            sphere.color.a = 1.
            text = Marker(header=header, ns=name, id=1, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            text.pose.position = Point(x=x, y=y, z=top_z+.6)
            text.pose.orientation.w = 1.
            text.scale.z = .9
            text.color = sphere.color
            text.text = f'{name} ({x:.2f}, {y:.2f}, {z:.2f}) m'
            arrow = Marker(header=header, ns=name, id=2, type=Marker.ARROW, action=Marker.ADD)
            arrow.points = [Point(x=x, y=y, z=top_z), Point(x=x, y=y, z=z)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = .12, .3, .5
            arrow.color = sphere.color
            nearby = Marker(header=header, ns=name, id=3, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            nearby.pose.position = Point(x=x, y=y, z=z+.5)
            nearby.pose.orientation.w = 1.
            nearby.scale.z = .23
            nearby.color = sphere.color
            nearby.text = name
            markers.extend([sphere, text, arrow, nearby])
            print(f'{name}: world XYZ = ({x}, {y}, {z}) m', flush=True)
        self.endpoint_publisher.publish(MarkerArray(markers=markers))

    def clear_planning(self):
        """起终点变化时清除旧路径，避免把旧结果显示为新查询的路径。"""
        header = Header(frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        for publisher in self.path_publishers.values():
            publisher.publish(RosPath(header=header))
        self.endpoint_publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))

    def publish_planning(self, planning):
        """复用 publisher 更新路径和法向偏移标记，不重复发布整栋建筑。"""
        header = Header(frame_id=self.frame_id, stamp=self.get_clock().now().to_msg())
        for name, filename in [('astar', 'astar_path.csv'), ('gpmp', 'gpmp_trajectory.csv')]:
            rows = np.genfromtxt(planning / filename, delimiter=',', names=True)
            message = RosPath(header=header)
            for x, y, z in zip(rows['x'], rows['y'], rows['z']):
                pose = PoseStamped(header=header)
                pose.pose.position = Point(x=float(x), y=float(y), z=float(z))
                pose.pose.orientation.w = 1.0
                message.poses.append(pose)
            self.path_publishers[name].publish(message)
            print(f'/pure3d/planning/{name}: {len(message.poses)} poses', flush=True)
        markers = MarkerArray()
        for index, (name, row, color) in enumerate([
            ('START', 0, (0.0, 1.0, 1.0)), ('GOAL', -1, (1.0, 0.1, 0.8)),
        ]):
            for kind in [Marker.SPHERE, Marker.TEXT_VIEW_FACING]:
                marker = Marker(header=header, ns=name, id=2*index+int(kind == Marker.TEXT_VIEW_FACING),
                                type=kind, action=Marker.ADD)
                marker.pose.position = Point(x=float(rows['x'][row]), y=float(rows['y'][row]),
                                             z=float(rows['z'][row]) + (0.3 if kind == Marker.TEXT_VIEW_FACING else 0.0))
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = .2
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = 1.0
                marker.text = name
                markers.markers.append(marker)
        surface = np.genfromtxt(planning / 'support_path.csv', delimiter=',', names=True)
        for index, row in enumerate([0, -1]):
            arrow = Marker(header=header, ns='endpoint_gravity_offset', id=index,
                           type=Marker.ARROW, action=Marker.ADD)
            arrow.points = [Point(x=float(data['x'][row]), y=float(data['y'][row]),
                                  z=float(data['z'][row])) for data in (surface, rows)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = .025, .06, .1
            arrow.color.r, arrow.color.b, arrow.color.a = 1.0, 1.0, 1.0
            markers.markers.append(arrow)
        self.endpoint_publisher.publish(markers)


def show_prediction(path, planning=None, interactive=False, query=None):
    """打开全部建筑原始点的模型预测，默认只有预测显示项开启。"""
    path = path.resolve()
    metadata = read_json(path.with_suffix(".json"))
    coord = read_pcd_xyz(path)
    scene = dict(bounds_world=[coord.min(0), coord.max(0)], frame_id=metadata["coordinate_frame"])
    prediction_display = cloud_display("PTv3 prediction | ALL points | cost 0-1",
                                       "/pure3d/prediction", enabled=True)
    prediction_display.update({"Style": "Points", "Size (Pixels)": 1, "Selectable": False})
    displays = [prediction_display, cloud_display("Source building | ALL points",
                                                  "/pure3d/prediction_source", flat=True)]
    query_path = query.resolve() if query is not None else None
    query_data = read_json(query_path) if query_path is not None else None
    if query_data is not None:
        query_data = dict(query_data, marker_top_z=float(coord[:, 2].max()+2.))
        xyz = np.asarray([query_data['start'], query_data['goal']], dtype=np.float64)
        scene['bounds_world'] = [xyz.min(0)-1.5, xyz.max(0)+1.5]
        prediction_display.update(Name=f"Query PCD | {metadata['points']:,} points | map XYZ", Style='Flat Squares',
                                  Alpha=.8, **{'Size (m)': .085})
        displays = [prediction_display]
    if planning is not None:
        planning = planning.resolve()
        route = np.genfromtxt(planning / 'gpmp_trajectory.csv', delimiter=',', names=True)
        xyz = np.column_stack([route[name] for name in ('x', 'y', 'z')])
        scene['bounds_world'] = [xyz.min(0) - 1.5, xyz.max(0) + 1.5]
        for name, color, width in [('astar', '255; 230; 80', .08), ('gpmp', '0; 220; 255', .045)]:
            displays.append({
                'Class': 'rviz_default_plugins/Path', 'Name': name.upper() + ' | map XYZ',
                'Enabled': True, 'Color': color, 'Alpha': 1.0, 'Buffer Length': 1,
                'Line Style': 'Billboards', 'Line Width': width, 'Pose Style': 'None',
                'Topic': {'Value': f'/pure3d/planning/{name}', 'Depth': 1,
                          'Reliability Policy': 'Reliable', 'Durability Policy': 'Transient Local',
                          'History Policy': 'Keep Last'},
            })
    if planning is not None or query_data is not None:
        displays.append({
            'Class': 'rviz_default_plugins/MarkerArray', 'Name': 'START / GOAL', 'Enabled': True,
            'Topic': {'Value': '/pure3d/planning/endpoints', 'Depth': 1,
                      'Reliability Policy': 'Reliable', 'Durability Policy': 'Transient Local',
                      'History Policy': 'Keep Last'},
        })
    output = ROOT / "runs/prediction_rviz" if planning is None else planning / 'rviz'
    if query_path is not None:
        output = query_path.parent / 'rviz'
    if interactive:
        from planner.interactive import InteractivePlanning, MARKER_NAMESPACE
        output = planning / 'interactive' / 'rviz'
        displays.append({
            'Class': 'rviz_default_plugins/InteractiveMarkers',
            'Name': 'Drag START / GOAL', 'Enabled': True, 'Value': True,
            'Interactive Markers Namespace': MARKER_NAMESPACE,
            'Show Descriptions': True, 'Show Axes': False, 'Show Visual Aids': True,
        })
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(output / "ros_logs"))
    config = output / "view.rviz"
    configuration = view_config(scene, displays)
    if planning is not None or query_data is not None:
        camera = configuration['Visualization Manager']['Views']['Current']
        camera['Name'] = 'Query input START / GOAL' if query_data is not None else 'Planning route'
        bounds = np.asarray(scene['bounds_world'])
        camera['Distance'] = float(np.max(bounds[1] - bounds[0]) * 1.6)
        if query_data is not None:
            camera['Yaw'] = .25
        saved_views = []
        for name, point in [('START', xyz[0]), ('GOAL', xyz[-1])]:
            closeup = dict(camera)
            closeup.update(Name=name, Distance=5.0, **{'Focal Point': dict(zip(['X', 'Y', 'Z'], map(float, point)))})
            if query_data is not None:
                closeup.update(Distance=4., Pitch=.22)
            saved_views.append(closeup)
        configuration['Visualization Manager']['Views']['Saved'] = saved_views
    config.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    rclpy.init()
    node = PredictionViewer(path, metadata, planning, query_data)
    controller = InteractivePlanning(node, path, planning) if interactive else None
    viewer = subprocess.Popen(["/opt/ros/humble/bin/rviz2", "-d", str(config),
                               "--ros-args", "-r", "__node:=pure3d_prediction_rviz"])
    (output / 'viewer.pid').write_text(str(viewer.pid)+'\n')
    print(f"RViz pid={viewer.pid} | prediction={path} | all {metadata['points']:,} points", flush=True)
    if interactive:
        # 静态查看器原先只 wait；交互模式必须处理 feedback、service 和完成回调。
        while viewer.poll() is None and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
        controller.shutdown()
    result = viewer.wait()
    print(f"RViz exited with code {result}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return result


class TomogramViewer(Node):
    """ROS 2 发布节点：把代价点云、建筑 PCD 与 nominal 原生切片发布到 RViz。"""

    def __init__(self, folder, scene, teacher, raw_cost=False):
        """发布静态 TF、全部条件代价点云、建筑 PCD 与 nominal 原生层。"""
        super().__init__("pure3d_tomogram_clouds")
        # TRANSIENT_LOCAL 让晚启动的 RViz 也能收到已发布的点云
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.cloud_publishers = []
        header = Header(frame_id=scene["frame_id"], stamp=self.get_clock().now().to_msg())
        # 发布 world -> 场景坐标系 的静态 TF，RViz 固定帧使用 scene.frame_id
        self.static_tf = StaticTransformBroadcaster(self)
        transform = TransformStamped(header=Header(frame_id="world", stamp=header.stamp),
                                     child_frame_id=scene["frame_id"])
        transform.transform.rotation.w = 1.0
        self.static_tf.sendTransform(transform)

        def publish(topic, message):
            """创建持久 publisher 并立即发布一条点云消息。"""
            # 保留 publisher 引用，防止局部变量销毁后消息被丢弃
            publisher = self.create_publisher(PointCloud2, topic, qos)
            publisher.publish(message)
            self.cloud_publishers.append(publisher)
            print(f"{topic}: {message.width:,} points", flush=True)

        # raw_cost 模式只影响发布时的数值与色标，不改磁盘上的归一化文件
        multiplier = teacher["cost_scale"] if raw_cost else 1.0
        print(f"Teacher distance curve p={teacher['teacher_parameters']['trav']['adjust_cost_p']}", flush=True)
        print(f"Display cost range: 0-{multiplier if raw_cost else 1:g}", flush=True)
        for profile in teacher["profiles"]:
            publish(f"/pure3d/tomogram/{profile['id']}",
                    pcd_to_pointcloud2(folder / profile["tomogram"], header, cost_multiplier=multiplier))
        publish("/pure3d/building", pcd_to_pointcloud2(folder / teacher["pcd"], header))
        # 读取 nominal 的 VST pickle，发布原生 ground/ceiling 水平切片
        with (folder / teacher["profiles"][0]["vst"]).open("rb") as stream:
            payload = pickle.load(stream)
        data = payload["data"]
        nx, ny = data.shape[2:]
        # 将整数网格索引换算为世界 XY；Z 使用每层 ground 高度
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        xy = np.column_stack(((x.ravel() - nx // 2) * payload["resolution"] + payload["center"][0],
                              (y.ravel() - ny // 2) * payload["resolution"] + payload["center"][1]))
        for layer in range(data.shape[1]):
            # ground 层按 trav_cost 着色；ceiling 层用固定代价 1.0 仅作几何参考
            publish(f"/pure3d/nominal/layer_G_{layer}",
                    layer_message(xy, data[3, layer], data[0, layer].astype(np.float32)
                                  / (1.0 if raw_cost else payload["cost_scale"]), header))
            publish(f"/pure3d/nominal/layer_C_{layer}", layer_message(xy, data[4, layer], 1.0, header))


def main():
    """生成 RViz 配置、启动发布节点，并等待 RViz 退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "datasets/wfc_capability_dr_stand_v3/scene_20261109")
    parser.add_argument("--prediction", type=Path, help="完整 prediction.pcd，按 trav_cost 0–1 显示全部点")
    parser.add_argument("--planning", type=Path, help="同时显示该规划目录的 A*、GPMP 路径和起终点")
    parser.add_argument("--query", type=Path, help="直接显示 query.json 的世界系起终点，不要求已有成功路径")
    parser.add_argument("--interactive", action="store_true", help="拖动起终点后重新规划，需同时指定 --prediction 和 --planning")
    parser.add_argument("--raw-cost", action="store_true", help="显示归一化之前的 VST 原始代价 [0, 50]。")
    args = parser.parse_args()
    if args.interactive and (args.prediction is None or args.planning is None):
        parser.error('--interactive requires --prediction and --planning')
    if args.prediction:
        raise SystemExit(show_prediction(args.prediction, args.planning, args.interactive, args.query))
    folder = args.scene.resolve()
    output = ROOT / "runs/tomogram_rviz"
    output.mkdir(parents=True, exist_ok=True)
    # 把 ROS 日志限制在 runs 目录，避免污染工作区其他位置
    os.environ.setdefault("ROS_LOG_DIR", str(output / "ros_logs"))
    scene, teacher = read_json(folder / "scene.json"), read_json(folder / "teacher.json")
    # 将 RViz 配置写到固定路径，方便复现与手工检查
    config = output / "view.rviz"
    config.write_text(yaml.safe_dump(rviz_config(scene, teacher, args.raw_cost), sort_keys=False), encoding="utf-8")
    rclpy.init()
    node = TomogramViewer(folder, scene, teacher, raw_cost=args.raw_cost)
    # 使用系统 RViz2 打开配置，并重命名节点避免冲突
    viewer = subprocess.Popen(["/opt/ros/humble/bin/rviz2", "-d", str(config),
                               "--ros-args", "-r", "__node:=pure3d_rviz"])
    print(f"RViz pid={viewer.pid}, config={config}", flush=True)
    result = viewer.wait()
    print(f"RViz exited with code {result}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
