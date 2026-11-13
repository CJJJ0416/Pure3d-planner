"""在 RViz 中随机查看一个已生成的 Filter PCD 盒，保留地图坐标。"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import secrets
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from common.pointcloud_io import pcd_to_pointcloud2, read_pcd_xyz
from Traversability.visualization.rviz import cloud_display, view_config
from utils.pointcloud import ROOT, read_json, write_json, workspace_path


ROLE_COLORS = np.array([[55, 155, 125], [60, 140, 200], [235, 151, 35],
                       [150, 100, 205], [207, 97, 45], [216, 65, 126], [182, 116, 34], [220, 45, 45]], dtype=np.uint32)


def colored_cloud(coord, roles, header, palette=ROLE_COLORS):
    """Map source_role to RGB for inspection, without predicting keep_score."""
    color = palette[roles]
    rgb = (color[:, 0] << 16) | (color[:, 1] << 8) | color[:, 2]
    values = np.empty((len(coord), 4), dtype="<f4")
    values[:, :3] = coord
    values[:, 3] = rgb.astype("<u4").view("<f4")
    return PointCloud2(header=header, height=1, width=len(coord),
        fields=[PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(["x", "y", "z", "rgb"])],
        is_bigendian=False, point_step=16, row_step=16 * len(coord), is_dense=True, data=values.tobytes())


SUPERVISION_DISPLAYS = [
    ("gt", "GT | continuous keep_target | ALL input points"),
    ("dynamic", "Dynamic object trails | remove entire trajectory"),
    ("registration", "Registration ghosts | geometric quality"),
    ("static_controls", "Static objects and disconnected structures | retain"),
    ("retained", "Retained by GT >= 0.5"),
]


def scored_cloud(coord, score, header):
    values = np.column_stack((coord, score)).astype("<f4")
    return PointCloud2(header=header, height=1, width=len(coord),
        fields=[PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, name in enumerate(["x", "y", "z", "keep_target"])],
        is_bigendian=False, point_step=16, row_step=16 * len(coord), is_dense=True, data=values.tobytes())


def supervision_clouds(coord, attributes, header):
    role, score = attributes["source_role"], attributes["keep_target"]
    masks = dict(gt=np.ones(len(coord), bool), dynamic=role == 7,
        registration=np.isin(role, [2, 6]), static_controls=(role == 5) | attributes["is_disconnected_control"],
        retained=score >= .5)
    return {name: scored_cloud(coord[mask], score[mask], header) for name, mask in masks.items()}


def boundary_marker(lower, upper, name, color, header, identifier):
    corners = np.array([[x, y, z] for x in [lower[0], upper[0]]
                        for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]])
    edges = [(i, i ^ axis) for i in range(8) for axis in [1, 2, 4] if i < (i ^ axis)]
    marker = Marker(header=header, ns=name, id=identifier, type=Marker.LINE_LIST, action=Marker.ADD)
    marker.pose.orientation.w = 1.
    marker.scale.x = .015
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.points = [Point(x=float(corners[i, 0]), y=float(corners[i, 1]), z=float(corners[i, 2]))
                     for edge in edges for i in edge]
    return marker


def window_attributes(box, metadata):
    """Read the shared-array interval for one displayed context window."""
    folder=box.parents[2]
    variant=metadata['variant']
    offsets=np.load(folder / f'windows_{variant:03d}_offsets.npy',mmap_mode='r')
    first,last=offsets[metadata['id']:metadata['id']+2]
    point_index=np.load(folder / f'windows_{variant:03d}_index.npy',mmap_mode='r')[first:last]
    with np.load(folder / f'rough_attributes_{variant:03d}.npz') as archive:
        attributes={name:archive[name][point_index] for name in archive.files}
    attributes['point_index']=np.array(point_index)
    attributes['core_mask']=np.load(folder / f'windows_{variant:03d}_core.npy',mmap_mode='r')[first:last]
    return attributes


class BoxPublisher(Node):
    def __init__(self, box, attributes, output):
        super().__init__("pure3d_filter_dataset")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        header = Header(frame_id="map", stamp=self.get_clock().now().to_msg())
        self.transform = StaticTransformBroadcaster(self)
        transform = TransformStamped(header=Header(frame_id="world", stamp=header.stamp), child_frame_id="map")
        transform.transform.rotation.w = 1.
        self.transform.sendTransform(transform)
        self.publishers_ = []
        rough = attributes["coord"]
        metadata = read_json(box / "box.json")
        core = attributes["core_mask"]
        clean = read_pcd_xyz(box.parents[2] / 'clean.pcd')
        clean = clean[((clean >= metadata['context_lower']) & (clean < metadata['context_upper'])).all(1)]
        clouds = {
            "roles": colored_cloud(rough, attributes["source_role"], header),
            "rough": colored_cloud(rough, np.zeros(len(rough),dtype=np.int64), header),
            "clean": colored_cloud(clean, np.zeros(len(clean),dtype=np.int64), header),
            "core": colored_cloud(rough[core], attributes["source_role"][core], header),
            **supervision_clouds(rough, attributes, header),
        }
        for name, cloud in clouds.items():
            publisher = self.create_publisher(PointCloud2, f"/pure3d/filter_dataset/{name}", qos)
            publisher.publish(cloud)
            self.publishers_.append(publisher)
            print(f"{name}: {cloud.width:,} points | map XYZ", flush=True)
        markers = [
            boundary_marker(metadata["core_lower"], metadata["core_upper"], "core", (0., .9, 1., .85), header, 0),
            boundary_marker(metadata["context_lower"], metadata["context_upper"], "context", (.7, .75, .85, .25), header, 1),
        ]
        publisher = self.create_publisher(MarkerArray, "/pure3d/filter_dataset/bounds", qos)
        publisher.publish(MarkerArray(markers=markers))
        self.publishers_.append(publisher)


class ScenePublisher(Node):
    """Publish full scene clouds once, with optional floor-band views."""
    def __init__(self, folder, variant, output):
        super().__init__("pure3d_filter_full_scene")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        header = Header(frame_id="map", stamp=self.get_clock().now().to_msg())
        self.transform = StaticTransformBroadcaster(self)
        transform = TransformStamped(header=Header(frame_id="world", stamp=header.stamp), child_frame_id="map")
        transform.transform.rotation.w = 1.
        self.transform.sendTransform(transform)
        self.publishers_ = []

        def publish(name, message):
            publisher = self.create_publisher(PointCloud2, f"/pure3d/filter_scene/{name}", qos)
            publisher.publish(message)
            self.publishers_.append(publisher)
            print(f"{name}: {message.width:,} points | map XYZ", flush=True)

        scene = read_json(folder / "scene.json")
        roles = read_json(folder / "roles.json")
        palette = np.zeros((max(roles["semantic_classes"].values()) + 1, 3), dtype=np.uint32)
        for instance in roles["instances"]:
            palette[instance["semantic_id"]] = instance["color_rgb"]
        clean = read_pcd_xyz(folder / "clean.pcd")
        with np.load(folder / "clean_attributes.npz") as values:
            semantic = values["semantic_id"]
        publish("clean", colored_cloud(clean, semantic, header, palette))
        heights = [floor["z_m"] for floor in scene["floors"]]
        for floor, height in enumerate(heights):
            low = -np.inf if floor == 0 else height - .15
            high = heights[floor + 1] - .15 if floor + 1 < len(heights) else np.inf
            selection = (clean[:, 2] >= low) & (clean[:, 2] < high)
            publish(f"floor_{floor}", colored_cloud(clean[selection], semantic[selection], header, palette))
        rough = read_pcd_xyz(folder / f"rough_{variant:03d}.pcd")
        with np.load(folder / f"rough_attributes_{variant:03d}.npz") as values:
            attributes = dict(values)
        publish("rough", colored_cloud(rough, attributes["source_role"], header))
        for name, message in supervision_clouds(rough, attributes, header).items():
            publish(name, message)
        self.counts = dict(clean=len(clean), rough=len(rough),
            dynamic=int((attributes["source_role"] == 7).sum()),
            registration=int(np.isin(attributes["source_role"], [2, 6]).sum()))



def full_scene_configuration(scene):
    def display(name, topic, enabled=False):
        result = cloud_display(name, topic, enabled=enabled, flat=True)
        result.update({"Color Transformer": "RGB8", "Style": "Points",
                       "Size (Pixels)": 1, "Alpha": 1., "Selectable": False})
        return result

    displays = [display("Whole clean scene | semantic colors | ALL points", "/pure3d/filter_scene/clean", True),
                display("Whole rough scene | source roles | ALL points", "/pure3d/filter_scene/rough")]
    for key, label in SUPERVISION_DISPLAYS:
        item = display(label, f"/pure3d/filter_scene/{key}")
        item.update({"Color Transformer": "Intensity", "Channel Name": "keep_target",
            "Autocompute Intensity Bounds": False, "Min Intensity": 0., "Max Intensity": 1.,
            "Use rainbow": False, "Min Color": "230; 40; 40", "Max Color": "30; 180; 90", "Size (Pixels)": 2})
        displays.append(item)
    displays.append({"Class": "rviz_common/Group", "Name": "Clean floor bands | toggle whole clean off first",
        "Enabled": True, "Displays": [display(f"Floor {floor} | map Z band", f"/pure3d/filter_scene/floor_{floor}")
                                        for floor in range(len(scene["floors"]))]})
    result = view_config(scene, displays)
    result["Visualization Manager"]["Views"]["Current"]["Name"] = "Whole WFC scene"
    result["Visualization Manager"]["Views"]["Saved"] = []
    bounds = np.asarray(scene["bounds_world"])
    for floor in scene["floors"]:
        view = dict(result["Visualization Manager"]["Views"]["Current"])
        view.update(Name=f"Floor {floor['floor']}", Distance=25., Pitch=.7,
            **{"Focal Point": {"X": float(bounds[:, 0].mean()), "Y": float(bounds[:, 1].mean()), "Z": float(floor["z_m"] + 1.)}})
        result["Visualization Manager"]["Views"]["Saved"].append(view)
    return result


def show_full_scene(folder, variant, output):
    scene = read_json(folder / "scene.json")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = workspace_path(output) if output else ROOT / "runs/filter_dataset_rviz" / f"full_scene_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(output / "ros_logs"))
    path = output / "filter_full_scene.rviz"
    path.write_text(yaml.safe_dump(full_scene_configuration(scene), sort_keys=False))
    rclpy.init()
    node = ScenePublisher(folder, variant, output)
    write_json(output / "scene_view.json", dict(scene=str(folder), seed=scene["seed"],
        variant=variant, coordinate_frame="map", additional_downsampling=False,
        floor_bands="display-only Z intervals, not VST slices", points=node.counts))
    viewer = subprocess.Popen(["/opt/ros/humble/bin/rviz2", "-d", str(path),
        "--ros-args", "-r", "__node:=pure3d_filter_full_scene_rviz"])
    (output / "viewer.pid").write_text(str(viewer.pid) + "\n")
    (output / "publisher.pid").write_text(str(os.getpid()) + "\n")
    print(f"Full-scene RViz PID={viewer.pid} | {folder}", flush=True)
    while viewer.poll() is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=.2)
    node.destroy_node()
    rclpy.shutdown()
    print(f"RViz exited with code {viewer.returncode}", flush=True)


def configuration(metadata, attributes):
    displays = []
    for key, label, enabled, color in [
        ("roles", "Rough | source roles | ALL input points", True, "130; 175; 240"),
        ("rough", "Rough | uniform blue", False, "130; 175; 240"),
        ("clean", "Clean reference | toggle to compare", False, "120; 235; 170"),
        ("core", "Training core | source roles", False, "130; 175; 240"),
    ]:
        display = cloud_display(label, f"/pure3d/filter_dataset/{key}", enabled=enabled, flat=True)
        display.update({"Style": "Points", "Size (Pixels)": 3, "Size (m)": .025,
                        "Alpha": 1., "Color": color, "Selectable": False})
        if key in ("roles", "core"):
            display["Color Transformer"] = "RGB8"
        displays.append(display)
    for key, label in SUPERVISION_DISPLAYS:
        item = cloud_display(label, f"/pure3d/filter_dataset/{key}", enabled=False, flat=True)
        item.update({"Style": "Points", "Size (Pixels)": 3, "Selectable": False,
            "Color Transformer": "Intensity", "Channel Name": "keep_target",
            "Autocompute Intensity Bounds": False, "Min Intensity": 0., "Max Intensity": 1.,
            "Use rainbow": False, "Min Color": "230; 40; 40", "Max Color": "30; 180; 90"})
        displays.append(item)
    displays.append({"Class": "rviz_default_plugins/MarkerArray", "Name": "Cyan: core | grey: context",
        "Enabled": True, "Topic": {"Value": "/pure3d/filter_dataset/bounds", "Depth": 1,
            "Reliability Policy": "Reliable", "Durability Policy": "Transient Local", "History Policy": "Keep Last"}})
    result = view_config(dict(bounds_world=[metadata["context_lower"], metadata["context_upper"]], frame_id="map"), displays)
    result["Visualization Manager"]["Views"]["Current"]["Name"] = f"Random box {metadata['variant']:03d}/{metadata['id']:05d}"
    return result


def show_prediction(path,output,selection=None):
    path=workspace_path(path)
    out=workspace_path(output) if output else path.parent/'rviz'
    out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('ROS_LOG_DIR',str(out/'ros_logs'))
    xyz=read_pcd_xyz(path)
    clouds=[('Prediction',path,True)]
    if selection is not None:
        selection=workspace_path(selection)
        clouds += [('Filtered',selection/'filtered.pcd',False),('Removed',selection/'removed.pcd',False)]
    displays=[]
    for name,file,enabled in clouds:
        d=cloud_display(name,f'/pure3d/filter_prediction/{name.lower()}',enabled=enabled,flat=name!='Prediction')
        if name=='Prediction':
            d.update({'Color Transformer':'Intensity','Channel Name':'keep_score','Autocompute Intensity Bounds':False,
                'Min Intensity':0.,'Max Intensity':1.,'Use rainbow':False,'Min Color':'230; 40; 40','Max Color':'30; 180; 90'})
        d.update({'Style':'Points','Size (Pixels)':1,'Selectable':False})
        displays.append(d)
    config=view_config(dict(bounds_world=[xyz.min(0).tolist(),xyz.max(0).tolist()],frame_id='map'),displays)
    config_path=out/'prediction.rviz';config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    rclpy.init();node=Node('pure3d_filter_prediction')
    header=Header(frame_id='map',stamp=node.get_clock().now().to_msg())
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    publishers=[]
    for name,file,enabled in clouds:
        pub=node.create_publisher(PointCloud2,f'/pure3d/filter_prediction/{name.lower()}',qos)
        pub.publish(pcd_to_pointcloud2(file,header));publishers.append(pub)
    viewer=subprocess.Popen(['/opt/ros/humble/bin/rviz2','-d',str(config_path)])
    while viewer.poll() is None and rclpy.ok(): rclpy.spin_once(node,timeout_sec=.2)
    node.destroy_node();rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction",type=Path,help="XYZ + keep_score prediction.pcd")
    parser.add_argument("--selection",type=Path,help="Optional directory exported by Filter.inference.threshold")
    parser.add_argument("--scene", type=Path, default=Path("datasets/filter_scoring_mesh_full_20261112/seed_821100"))
    parser.add_argument("--box", type=Path, help="View this box instead of a random selection.")
    parser.add_argument("--seed", type=int, help="Seed for reproducible uniform selection from boxes.json.")
    parser.add_argument("--with-trails", action="store_true", help="Randomly choose boxes containing dynamic or registration ghosts.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-scene", action="store_true", help="Display the complete scene instead of one box.")
    parser.add_argument("--variant", type=int, default=0, help="Whole-scene rough variant.")
    args = parser.parse_args()
    if args.prediction:
        return show_prediction(args.prediction,args.output,args.selection)
    scene = workspace_path(args.scene)
    if args.full_scene:
        return show_full_scene(scene, args.variant, args.output)
    seed = secrets.randbits(32) if args.seed is None else args.seed
    samples = read_json(scene / "boxes.json")["samples"]
    if args.with_trails:
        samples = [sample for sample in samples if sample["dynamic_points"] + sample["registration_points"] > 0]
    if args.box:
        box = workspace_path(args.box)
    else:
        selected = samples[int(np.random.default_rng(seed).integers(len(samples)))]
        box = scene / selected["path"]
    metadata = read_json(box / "box.json")
    attributes=window_attributes(box,metadata)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = workspace_path(args.output) if args.output else ROOT / "runs/filter_dataset_rviz" / stamp
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(output / "ros_logs"))
    selection = dict(random_seed=seed, selection=("uniform_over_trail_boxes" if args.with_trails else "uniform_over_all_boxes") if not args.box else "explicit_box",
        box=str(box), scene=str(scene), metadata=metadata,
        dynamic_points=int((attributes["source_role"] == 7).sum()),
        registration_points=int(np.isin(attributes["source_role"], [2, 6]).sum()))
    write_json(output / "selection.json", selection)
    path = output / "filter_random_box.rviz"
    path.write_text(yaml.safe_dump(configuration(metadata, attributes), sort_keys=False))
    print(f"Selected {box} | seed={seed} | output={output}", flush=True)
    rclpy.init()
    node = BoxPublisher(box, attributes, output)
    viewer = subprocess.Popen(["/opt/ros/humble/bin/rviz2", "-d", str(path),
        "--ros-args", "-r", "__node:=pure3d_filter_dataset_rviz"])
    (output / "viewer.pid").write_text(str(viewer.pid) + "\n")
    (output / "publisher.pid").write_text(str(os.getpid()) + "\n")
    print(f"RViz PID={viewer.pid}", flush=True)
    while viewer.poll() is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=.2)
    node.destroy_node()
    rclpy.shutdown()
    print(f"RViz exited with code {viewer.returncode}", flush=True)


if __name__ == "__main__":
    main()
