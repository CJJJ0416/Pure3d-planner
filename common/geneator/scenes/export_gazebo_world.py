#!/usr/bin/env python3
"""Export an existing triangular OBJ as an exact Gazebo Classic static mesh.

Only the container changes: vertices, face winding, metres and world origin are
preserved. No welding, decimation, convex hull, floor filling or terrain rebuild.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obj_triangles(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        if parts[0] == 'v':
            if len(parts) != 4:
                raise ValueError('Expected untransformed XYZ vertices')
            vertices.append([float(v) for v in parts[1:]])
        elif parts[0] == 'f':
            if len(parts) != 4:
                raise ValueError('Expected already triangulated OBJ')
            ids = [int(v.split('/')[0]) for v in parts[1:]]
            faces.append([v - 1 if v > 0 else len(vertices) + v for v in ids])
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise ValueError('Invalid mesh')
    return vertices, faces


def collada(path, vertices, faces):
    root = ET.Element('COLLADA', xmlns='http://www.collada.org/2005/11/COLLADASchema', version='1.4.1')
    asset = ET.SubElement(root, 'asset')
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for key in ('created', 'modified'):
        ET.SubElement(asset, key).text = stamp
    ET.SubElement(asset, 'unit', name='meter', meter='1')
    ET.SubElement(asset, 'up_axis').text = 'Z_UP'
    effect = ET.SubElement(ET.SubElement(root, 'library_effects'), 'effect', id='terrain-effect')
    technique = ET.SubElement(ET.SubElement(effect, 'profile_COMMON'), 'technique', sid='common')
    phong = ET.SubElement(technique, 'phong')
    for key, color in [('ambient', '.25 .25 .25 1'), ('diffuse', '.68 .72 .76 1'), ('specular', '.08 .08 .08 1')]:
        ET.SubElement(ET.SubElement(phong, key), 'color').text = color
    ET.SubElement(ET.SubElement(phong, 'shininess'), 'float').text = '12'
    material = ET.SubElement(ET.SubElement(root, 'library_materials'), 'material', id='terrain-material', name='Terrain')
    ET.SubElement(material, 'instance_effect', url='#terrain-effect')
    geometry = ET.SubElement(ET.SubElement(root, 'library_geometries'), 'geometry', id='terrain-geometry', name='Terrain')
    mesh = ET.SubElement(geometry, 'mesh')
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths > 1e-15, lengths, 1)[:, None]
    normals[lengths <= 1e-15] = [0, 0, 1]
    for name, values in [('positions', vertices), ('normals', normals)]:
        source = ET.SubElement(mesh, 'source', id='terrain-' + name)
        array_id = 'terrain-' + name + '-array'
        ET.SubElement(source, 'float_array', id=array_id, count=str(values.size)).text = ' '.join(format(float(v), '.17g') for v in values.ravel())
        accessor = ET.SubElement(ET.SubElement(source, 'technique_common'), 'accessor', source='#' + array_id, count=str(len(values)), stride='3')
        for axis in 'XYZ':
            ET.SubElement(accessor, 'param', name=axis, type='float')
    vertex = ET.SubElement(mesh, 'vertices', id='terrain-vertices')
    ET.SubElement(vertex, 'input', semantic='POSITION', source='#terrain-positions')
    tri = ET.SubElement(mesh, 'triangles', count=str(len(faces)), material='terrain-symbol')
    ET.SubElement(tri, 'input', semantic='VERTEX', source='#terrain-vertices', offset='0')
    ET.SubElement(tri, 'input', semantic='NORMAL', source='#terrain-normals', offset='1')
    indices = np.stack((faces, np.repeat(np.arange(len(faces))[:, None], 3, axis=1)), axis=-1)
    ET.SubElement(tri, 'p').text = ' '.join(map(str, indices.ravel()))
    scene = ET.SubElement(ET.SubElement(root, 'library_visual_scenes'), 'visual_scene', id='Scene', name='Scene')
    node = ET.SubElement(scene, 'node', id='Terrain', name='Terrain', type='NODE')
    instance = ET.SubElement(node, 'instance_geometry', url='#terrain-geometry')
    binding = ET.SubElement(ET.SubElement(instance, 'bind_material'), 'technique_common')
    ET.SubElement(binding, 'instance_material', symbol='terrain-symbol', target='#terrain-material')
    ET.SubElement(ET.SubElement(root, 'scene'), 'instance_visual_scene', url='#Scene')
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obj', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Gazebo package directory or staging directory')
    parser.add_argument('--name', required=True)
    parser.add_argument('--sha256', required=True, help='Expected frozen training OBJ SHA256')
    parser.add_argument('--spawn', type=float, nargs=4, required=True, metavar=('X', 'Y', 'Z', 'YAW'))
    parser.add_argument('--with-launch',action='store_true',help='Also create an optional A1 launch wrapper')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', args.name):
        parser.error('Use a simple model name')
    if digest(args.obj) != args.sha256:
        raise ValueError('Source mesh differs from frozen training manifest')
    scene = json.loads(args.scene.read_text())
    vertices, faces = obj_triangles(args.obj)
    model = args.output / 'models' / args.name
    if model.exists():
        raise FileExistsError(model)
    (model / 'meshes').mkdir(parents=True)
    (args.output / 'worlds').mkdir(exist_ok=True)
    if args.with_launch:
        (args.output / 'launch').mkdir(exist_ok=True)
    dae = model / 'meshes' / 'building.dae'
    collada(dae, vertices, faces)
    uri = f'model://{args.name}/meshes/building.dae'
    (model / 'model.config').write_text(f'''<?xml version="1.0"?>
<model><name>{args.name}</name><version>1.0</version><sdf version="1.7">model.sdf</sdf>
<description>Exact procedural training building, seed {scene['seed']}, metres and Z-up.</description></model>
''')
    (model / 'model.sdf').write_text(f'''<?xml version="1.0"?>
<sdf version="1.7"><model name="{args.name}"><static>true</static><pose>0 0 0 0 0 0</pose>
<link name="terrain"><visual name="visual"><geometry><mesh><uri>{uri}</uri><scale>1 1 1</scale></mesh></geometry><cast_shadows>true</cast_shadows></visual>
<collision name="collision"><geometry><mesh><uri>{uri}</uri><scale>1 1 1</scale></mesh></geometry>
<max_contacts>20</max_contacts><surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction>
<bounce><restitution_coefficient>0</restitution_coefficient><threshold>1000000</threshold></bounce>
<contact><ode><kp>1000000</kp><kd>100</kd><max_vel>0.1</max_vel><min_depth>0.001</min_depth></ode></contact></surface>
</collision></link></model></sdf>
''')
    world_name = args.name.lower().replace('-', '_')
    (args.output / 'worlds' / (world_name + '.world')).write_text(f'''<?xml version="1.0"?>
<sdf version="1.7"><world name="{world_name}">
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so"><update_rate>1000</update_rate></plugin>
<gravity>0 0 -9.81</gravity><physics name="ode" type="ode"><max_step_size>0.001</max_step_size><real_time_update_rate>1000</real_time_update_rate><real_time_factor>1</real_time_factor><ode><solver><type>quick</type><iters>50</iters></solver></ode></physics>
<scene><ambient>0.5 0.5 0.5 1</ambient><background>0.8 0.85 0.9 1</background><shadows>true</shadows></scene>
<light name="sun" type="directional"><cast_shadows>true</cast_shadows><pose>0 0 20 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse><specular>0.1 0.1 0.1 1</specular><direction>-0.5 0.2 -1</direction></light>
<gui fullscreen="false"><camera name="user_camera"><pose>12 -25 14 0 0.5 1.8</pose><view_controller>orbit</view_controller></camera></gui>
<!-- The existing mesh supplies both floors. A ground_plane at z=0 would fill real voids. -->
<include><uri>model://{args.name}</uri><name>{args.name}</name><pose>0 0 0 0 0 0</pose></include>
</world></sdf>
''')
    defaults = dict(zip(('x', 'y', 'height', 'yaw'), map(str, args.spawn)))
    defaults.update(gui='true', rviz='false', planner='None', device='cuda', mid360='true')
    launch = '''"""Generated building with the existing A1 controller and Gazebo launch."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    package = Path(get_package_share_directory('unitree_gazebo'))
    defaults = DEFAULTS
    args = {key: LaunchConfiguration(key) for key in defaults}
    args.update(world=str(package/'worlds'/'WORLD_FILE'), odom_x='0', odom_y='0', odom_z='0')
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k,v in defaults.items()],
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(package/'launch'/'gazebo.launch.py')),
                                 launch_arguments=args.items())])
'''.replace('DEFAULTS', repr(defaults)).replace('WORLD_FILE', world_name + '.world')
    if args.with_launch:
        (args.output / 'launch' / (world_name + '.launch.py')).write_text(launch)
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    tree = ET.parse(dae)
    restored = np.fromstring(tree.find('.//c:source[@id="terrain-positions"]/c:float_array', ns).text, sep=' ').reshape(-1, 3)
    restored_faces = np.fromstring(tree.find('.//c:triangles/c:p', ns).text, sep=' ', dtype=np.int64).reshape(-1, 3, 2)[:, :, 0]
    assert np.array_equal(restored, vertices) and np.array_equal(restored_faces, faces)
    report = dict(source_obj=str(args.obj.resolve()), source_scene=str(args.scene.resolve()),
        source_sha256=args.sha256, scene_sha256=digest(args.scene), dae_sha256=digest(dae),
        seed=scene['seed'], source_name=scene['name'], model_name=args.name,
        vertices=len(vertices), triangles=len(faces), bounds_m=[vertices.min(0).tolist(), vertices.max(0).tolist()],
        metre_scale=1, up_axis='Z_UP', world_transform='identity', vertex_roundtrip_max_error_m=0,
        triangle_indices_exact=True, source_degenerate_triangles=int((np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]]),axis=1)<1e-12).sum()),
        added_ground_plane=False, visual_collision_same_mesh=True,
        suggested_a1_spawn_xyz_yaw=args.spawn, gazebo_physics='ODE',
        source_m0_dynamics_transferred=False)
    (model / 'export_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    terrain_description=('连续平滑坡面、平台、孔洞和实体障碍；没有台阶、扶手或栏杆'
        if scene.get('stair_structure',{}).get('style')=='ramp' else '楼梯、楼板、栏杆、孔洞和实体障碍')
    if not args.with_launch:
        (model/'README.md').write_text(f'''# {args.name}

seed `{scene['seed']}`，包含{terrain_description}。
网格保持米制、Z向上和原世界坐标，显示与碰撞使用同一个DAE。
`model.sdf`/`model.config` 是静态模型描述；`worlds/{world_name}.world` 是完整场景。
没有添加补地平面，也没有加载机器人、导航策略或路线。
Gazebo加载前应将包内的 `models` 目录加入 `GAZEBO_MODEL_PATH`。
此导出仅提供场景文件，不会自动启动仿真。来源与精度见 `export_manifest.json`。
''')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return
    (model / 'README.md').write_text(f'''# {args.name}

当前单条上楼训练所用的整栋两层建筑，seed `{scene['seed']}`。
DAE 顶点、三角面、绕序和原点与源 OBJ 一致，单位米、Z 向上，显示与碰撞共用同一份网格。
包含{terrain_description}，不补地面、不缩放建筑。
材质使用统一灰色，静态网格碰撞使用 ODE；接触摩擦系数为 0.8。

在工作空间编译安装后启动：

```bash
cd /home/cjh/glocal-nav2
source /opt/ros/humble/setup.bash
colcon build --packages-select unitree_gazebo --symlink-install
source install/setup.bash
ros2 launch unitree_gazebo {world_name}.launch.py
```

默认启动 A1、MID360 和 Gazebo，机器人放在本次上楼训练起点附近；`planner:=None`，RViz 关闭。
可追加 `planner:=scan rviz:=true`，但新建筑需要重新建立对应地图，旧 Building 地图不适用。
`x/y/height/yaw/gui/rviz/device/mid360/planner` 可以覆盖。
出生高度 `{args.spawn[2]:.3f} m` 是 A1 基座的初始离地高度，不能沿用 M0 的悬高。
Gazebo 继续使用工作空间现有 A1 腿部控制器；本次仅导出场景，没有部署 M0 导航策略。

场景：`worlds/{world_name}.world`；网格：`models/{args.name}/meshes/building.dae`。
`export_manifest.json` 保存源网格指纹、坐标范围和格式转换校验。
''')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
