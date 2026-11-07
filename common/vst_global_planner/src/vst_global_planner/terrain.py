"""Combined metric terrain, using glocal2-nav's attributed geometry backend.

The retained upstream terrain-generator supplies the solid reference stairs.
Floating stairs, ramp profiles and convex prisms reuse glocal2-nav. Curved bridge,
arched tunnel and helical treads are extensions in this project, not upstream WFC.
"""
from collections import defaultdict
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import trimesh

from .layout import random_layout,transform_xy
from .domain_randomization import sample_geometry
from .terrain_shapes import add_curved_flight,cut_platform,union_buildings,slab_from_footprints


def load_backend(root):
    root = Path(root).resolve()
    for rel in ("src/stair_nav/scenes/geometry.py", "src/stair_nav/scenes/stair_structure.py",
                "src/stair_nav/vendor/eth_stairs.py"):
        if not (root / rel).is_file():
            raise FileNotFoundError(root / rel)
    sys.path.insert(0, str(root / "src"))
    geometry = importlib.import_module("stair_nav.scenes.geometry")
    structure = importlib.import_module("stair_nav.scenes.stair_structure")
    if not Path(geometry.__file__).resolve().is_relative_to(root):
        raise RuntimeError("A different stair_nav checkout is already imported")
    provenance = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in (
        "src/stair_nav/scenes/geometry.py", "src/stair_nav/scenes/stair_structure.py",
        "src/stair_nav/vendor/eth_stairs.py", "src/stair_nav/vendor/LICENSE.terrain-generator")}
    return geometry, structure, dict(root=str(root), files=provenance)


def create_scene(cfg, backend_root):
    cfg = sample_geometry(cfg)
    geo, st, provenance = load_backend(backend_root)
    width, depth = map(float, cfg["size_m"])
    height, slab = float(cfg["deck_height_m"]), float(cfg["slab_thickness_m"])
    if width < 48 or depth < 40:
        raise ValueError("The compact playground preset requires at least 48 x 40 m")
    if not 1.8 <= height <= 2.8:
        raise ValueError("The current layout supports deck heights between 1.8 and 2.8 m")
    if not .7 <= cfg["bridge_width_m"] <= 1.5:
        raise ValueError("Bridge width must be between 0.7 and 1.5 m")
    if not .02 <= cfg["sample_spacing_m"] <= cfg["vst"]["resolution"] / 2:
        raise ValueError("Point spacing must resolve the VST cells")
    rng = np.random.default_rng(cfg["seed"])
    parts, groups = [], defaultdict(list)
    scene = dict(schema_version=1, name=cfg["name"], seed=int(cfg["seed"]),
                 frame_id="map", units="m", z_axis="up", yaw=0.,
                 width=width, depth=depth, height=height, slab_thickness=slab,
                 layout="mixed_traversability_platform", capabilities=dict(random_map=False),
                 architecture=dict(family="mixed_platform", label="组合通行训练平台"),
                 config=dict(cfg), solids=[], support_surfaces=[], stair_flights=[],
                 floors=[dict(id=0, z=0.), dict(id=1, z=height)],
                 floor_graph=dict(nodes=[dict(id=0, z=0.), dict(id=1, z=height)], edges=[]),
                 queries=[], rooms=[], relief=[], landings=[], floor_features=[],
                 generator_provenance=provenance)
    scene["config"]["sample_spacing"] = cfg["sample_spacing_m"]
    scene["config"]["stair_structure"] = dict(style="floating", railings=False,
                                                tread_thickness=.055)
    landmarks, paths = [], {}
    platform_openings=[]

    def remember(meshes, group):
        for mesh in meshes:
            if not mesh.is_watertight or not mesh.is_volume:
                raise ValueError(f"Invalid closed component: {group}")
            parts.append(mesh)
            groups[group].append(mesh)

    def solid(lo, hi, group):
        mesh = geo.box(lo, hi)
        remember([mesh], group)
        scene["solids"].append(dict(kind=group, lo=list(lo), hi=list(hi)))

    def prism(xy, top, bottom, group):
        meshes = []
        st.add_prism(scene, meshes, xy, top, bottom, group)
        remember(meshes, group)

    def deck(lo, hi, group):
        if group=="underpass_landing":
            plate=geo.box([*lo,height-slab],[*hi,height])
            plate,openings=cut_platform(plate,lo,hi,height,slab,cfg["platform_opening_mode"],
                cfg["platform_hole_size_fraction"],cfg["platform_hole_offset_fraction"])
            remember([plate],group)
            for opening in openings:
                opening["region_id"]="raised_platform"
                platform_openings.append(opening)
        else:
            solid([*lo, height-slab], [*hi, height], group)
        for x in (lo[0]+.25, hi[0]-.25):
            for y in (lo[1]+.25, hi[1]-.25):
                solid([x-.11, y-.11, -.05], [x+.11, y+.11, height-slab+.02], group)

    # Build structure units in local construction coordinates, then separate them.
    west_width,west_depth=map(float,cfg["west_deck_size_m"])
    east_width,east_depth=map(float,cfg["east_deck_size_m"])
    deck([-8.-west_width,.5],[-8.,.5+west_depth],"west_deck")
    deck([-1.5,3.],[-1.5+east_width,3.+east_depth],"east_deck")

    # A true S-shaped narrow bridge, including exposed drop edges.
    t = np.linspace(0., 1., 97)
    amplitude = float(cfg["bridge_amplitude_m"] * rng.uniform(.92, 1.08))
    center = np.c_[-8.15+6.8*t, 5.+amplitude*np.sin(2*np.pi*t)]
    tangent = np.c_[np.full(len(t), 6.8), amplitude*2*np.pi*np.cos(2*np.pi*t)]
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    bw = float(cfg["bridge_width_m"])
    left, right = center+normal*bw/2, center-normal*bw/2
    footprints=[np.array([left[i],right[i],right[i+1],left[i+1]]) for i in range(len(t)-1)]
    remember([slab_from_footprints(footprints,height,slab)],"curved_narrow_bridge")
    paths["curved_bridge"] = np.c_[center, np.full(len(t), height)]
    landmarks.append(dict(name="Curved narrow bridge", xyz=[-4.75, 5., height+.25]))

    def ramp(x, y, run, width_m, name):
        flight = dict(id=name, x=x, y=y, direction=1, width=width_m,
                      n_steps=1, step_depth=run, riser=height, length=run,
                      end_y=y+run, base_z=0., top_z=height, from_floor=0, to_floor=1)
        q, z, blend, slope = st.ramp_profile(flight, transition=.25, spacing=.12)
        for a, b, za, zb in zip(q[:-1], q[1:], z[:-1], z[1:]):
            m = float((zb-za)/(b-a))
            top = [0., m, float(za-m*(y+a))]
            prism([[x-width_m/2,y+a], [x+width_m/2,y+a],
                   [x+width_m/2,y+b], [x-width_m/2,y+b]], top,
                  [0., m, top[2]-.18], name)
        flight.update(construction="ramp", profile_q_m=q.tolist(), profile_z_m=z.tolist(),
                      transition_length_m=blend, maximum_slope_rad=slope)
        scene["stair_flights"].append(flight)
        paths[name] = np.c_[np.full(len(q), x), y+q, z]
        landmarks.append(dict(name=name.replace("_", " ").title(), xyz=[x, y+run/2, height/2+.3]))

    ramp(-11.8,-9.,9.5,float(cfg["west_ramp_width_m"]),"west_ramp")
    landing_w,landing_d=map(float,cfg["ramp_landing_size_m"])
    deck([-11.8-landing_w/2,.5],[-11.8+landing_w/2,.5+landing_d],"ramp_landing")
    # The larger platform and its ramp form a building independent of the underpass.
    ramp(12.1,-10.3,9.1,float(cfg["east_ramp_width_m"]),"east_ramp")
    landing_width,landing_depth = map(float,cfg["underpass_landing_size_m"])
    if not 2.3<=landing_width<=8.0 or not 1.4<=landing_depth<=8.0:
        raise ValueError("Independent platform dimensions must fit the playground preset")
    platform_x0=12.1-landing_width/2
    deck([platform_x0,-1.2],[platform_x0+landing_width,-1.2+landing_depth],"underpass_landing")
    scene["underpass_landing_size_m"]=[landing_width,landing_depth]
    paths["platform_top"] = np.c_[np.linspace(platform_x0+.8,platform_x0+landing_width-.8,41),
        np.full(41,-1.2+.2),np.full(41,height)]
    landmarks[-1]=dict(name="Detached large platform / ramp",xyz=[12.1,-1.2+landing_depth/2,height+.3])

    # Open-riser stairs reuse glocal2-nav's thin treads and side stringers.
    n = int(cfg["open_stair_steps"])
    step_depth=float(cfg["open_stair_depth_m"])
    flight = dict(id="open_stairs", x=1., y=3.-n*step_depth, end_y=3., direction=1,
                  width=float(cfg["open_stair_width_m"]), step_depth=step_depth, n_steps=n, riser=height/n,
                  base_z=0., top_z=height, length=(n+1)*step_depth+.2,
                  landing_depth=.2, from_floor=0, to_floor=1)
    meshes,probes,attach,normal,tangent,top_footprints=add_curved_flight(scene,flight,
        cfg["open_stair_bend_degrees"],cfg["open_stair_width_ratios"],st.add_prism)
    remember(meshes,"open_riser_stairs")
    scene["stair_flights"].append(flight)
    landing_w,landing_d=map(float,cfg["open_stair_landing_size_m"])
    # The platform follows the final stair heading and overlaps the top tread.
    front=attach-normal*landing_w/2
    back=attach+normal*landing_w/2
    corners=np.array([front,back,back+tangent*landing_d,front+tangent*landing_d])
    remember([slab_from_footprints([*top_footprints,corners],height,slab)],"open_stair_landing")
    for side in (-1,1):
        for distance in (.25,landing_d-.25):
            p=attach+normal*side*(landing_w/2-.25)+tangent*distance
            solid([p[0]-.11,p[1]-.11,-.05],[p[0]+.11,p[1]+.11,height-slab+.02],"open_stair_landing")
    paths["open_stairs"]=probes
    landmarks.append(dict(name="Open-riser stairs", xyz=[1., -.1, height/2+.35]))

    # Two open-riser flights with an actual mid-height platform and 180-degree turn.
    count=int(cfg["switchback_steps_per_flight"])
    lower_depth,upper_depth=map(float,cfg["switchback_depth_m"])
    stair_width=float(cfg["switchback_width_m"])
    lower_x,lower_y=-12.65,7.75
    upper_x=lower_x+stair_width+.7
    turn_y=lower_y+count*lower_depth+.25
    landing_depth=float(cfg["switchback_landing_depth_m"])
    flights=[dict(id="switchback_lower",x=lower_x,y=lower_y,direction=1,step_depth=lower_depth,base_z=0.),
             dict(id="switchback_upper",x=upper_x,y=turn_y,direction=-1,step_depth=upper_depth,base_z=height/2)]
    for flight in flights:
        depth_m=flight["step_depth"]
        flight.update(width=stair_width,n_steps=count,riser=height/(2*count),
            end_y=flight["y"]+flight["direction"]*count*depth_m,
            top_z=flight["base_z"]+height/2,length=(count+1)*depth_m+.2,
            landing_depth=.2,from_floor=0 if flight["base_z"]==0 else "mid_landing",
            to_floor="mid_landing" if flight["base_z"]==0 else 1)
        ratios=np.interp(np.linspace(0,1,count+1),[0,.5,1],cfg["switchback_width_ratios"])
        flight["tread_widths"]=(stair_width*ratios).tolist()
        meshes=[]
        st.add_flight(scene,meshes,flight)
        remember(meshes,"switchback_stairs")
        scene["stair_flights"].append(flight)
        q=(np.arange(count+1)+.5)*depth_m
        paths[flight["id"]]=np.c_[np.full(count+1,flight["x"]),
            flight["y"]+flight["direction"]*q,flight["base_z"]+np.arange(count+1)*height/(2*count)]
    solid([lower_x-stair_width/2,turn_y,height/2-slab],
          [upper_x+stair_width/2,turn_y+landing_depth,height/2],"switchback_stairs")
    paths["switchback_turn"]=np.c_[np.linspace(lower_x,upper_x,41),
                                  np.full(41,turn_y+landing_depth/3),np.full(41,height/2)]
    scene["switchback"]=dict(turn_degrees=180.,mid_height_m=height/2,
        landing_size_m=[upper_x-lower_x+stair_width,landing_depth],steps_per_flight=count)
    top_width,top_depth=map(float,cfg["switchback_top_size_m"])
    top_y=flights[-1]["end_y"]+.02
    deck([upper_x-top_width/2,top_y-top_depth],[upper_x+top_width/2,top_y],"switchback_top_landing")
    landmarks.append(dict(name="U-turn stairs / mid landing",xyz=[-11.6,10.7,height/2+.6]))

    # Helical treads: >1 turn gives overlapping XY surfaces with true clearance.
    cx, cy, ri = 9., 7.5, .38
    ro=float(cfg["spiral_outer_radius_m"])
    walk_radius=ro-.85
    ns = int(cfg["spiral_steps"])
    turn = float(cfg["spiral_turn_degrees"])
    if turn != 450. or ns < 20 or height/ns <= .045:
        raise ValueError("The connected spiral preset uses 450 degrees and valid thin treads")
    angles = np.deg2rad(np.linspace(90., 90.+turn, ns+1))
    spiral_points = []
    for j, (a, b) in enumerate(zip(angles[:-1], angles[1:])):
        edge = np.linspace(a, b, 5)
        # One closed extrusion per tread avoids coplanar internal subdivision faces.
        outer = np.c_[cx+ro*np.cos(edge),cy+ro*np.sin(edge)]
        inner = np.c_[cx+ri*np.cos(edge),cy+ri*np.sin(edge)]
        xy = np.vstack([outer,inner])
        k = len(edge)
        faces = [[v for v in face] for u in range(k-1)
                 for face in ((u,u+1,k+u+1),(u,k+u+1,k+u))]
        tread = trimesh.creation.extrude_triangulation(xy,np.asarray(faces),height=.045)
        tread.apply_translation([0,0,(j+1)*height/ns-.045])
        remember([tread],"spiral_stairs")
        mid = (a+b)/2
        spiral_points.append([cx+walk_radius*np.cos(mid), cy+walk_radius*np.sin(mid), (j+1)*height/ns])
    column = trimesh.creation.cylinder(radius=.48, height=height+.1, sections=48)
    column.apply_translation([cx, cy, height/2-.05])
    remember([column], "spiral_stairs")
    connector_width,connector_depth=map(float,cfg["spiral_connector_size_m"])
    connector_right=cx-ro+.3
    deck([connector_right-connector_width,cy-connector_depth/2],
         [connector_right,cy+connector_depth/2],"spiral_connector")
    paths["spiral_stairs"] = np.asarray(spiral_points)
    exit_angle = (angles[-2]+angles[-1])/2
    radii = np.linspace(walk_radius,ro+.85,31)
    paths["spiral_exit"] = np.c_[cx+radii*np.cos(exit_angle),cy+radii*np.sin(exit_angle),
                                  np.full(len(radii),height)]
    scene["spiral"] = dict(center_xy=[cx,cy], inner_radius_m=ri, outer_radius_m=ro,
                            steps=ns, turn_degrees=turn, tread_thickness_m=.045,
                            rise_per_turn_m=height*360/turn)
    landmarks.append(dict(name="450-degree spiral stairs", xyz=[cx,cy,height+.6]))

    # Arched underpass: roof and support walls are actual collision geometry.
    ax, half, spring = 8.5, 1.4, .65
    inner_peak = height-.4
    solid([6., -5., -.02], [ax-half, -1., height], "arched_underpass")
    solid([ax+half, -5., -.02], [11., -1., height], "arched_underpass")
    xs = np.linspace(ax-half, ax+half, 33)
    zs = spring+(inner_peak-spring)*np.sqrt(np.maximum(0,1-((xs-ax)/half)**2))
    for a,b,za,zb in zip(xs[:-1],xs[1:],zs[:-1],zs[1:]):
        vertices = [[x,y,z] for y in (-5.,-1.) for x,z in ((a,za),(b,zb),(b,height),(a,height))]
        remember([trimesh.convex.convex_hull(vertices)], "arched_underpass")
    paths["underpass"] = np.c_[np.full(41,ax), np.linspace(-6.,0.,41), np.zeros(41)]
    scene["underpass"] = dict(center_x=ax, y_range=[-5.,-1.], clear_width_m=2*half,
                               central_clearance_m=inner_peak)
    landmarks.append(dict(name="Arched underpass / roof", xyz=[8.5,-3.,height+.4]))

    # An additional solid stair reference executes the retained ETH generator.
    solid_n=int(cfg["solid_stair_steps"])
    solid_depth=float(cfg["solid_stair_depth_m"])
    solid_y=-5.-(solid_n+1)*solid_depth
    solid_stairs = geo.stairs(float(cfg["solid_stair_width_m"]),solid_depth,solid_n,height,.5,slab,6.55,solid_y)
    remember([solid_stairs], "solid_reference_stairs")
    paths["solid_stairs"] = np.c_[np.full(solid_n+1,6.55),solid_y+(np.arange(solid_n+1)+.5)*solid_depth,
                                   np.arange(solid_n+1)*height/solid_n]

    # Randomized clutter uses a separate region, away from structure interfaces.
    occupied,obstacle_records = [],[]
    for i in range(int(cfg["obstacle_count"])):
        for _ in range(500):
            x,y = rng.uniform([-8.5,-10.7],[-2.3,-3.])
            if all(np.linalg.norm(np.array([x,y])-p)>1.25 for p in occupied):
                break
        else:
            raise ValueError("Not enough free clutter space for the obstacle count")
        occupied.append(np.array([x,y]))
        shape=str(rng.choice(cfg["obstacle_shapes"]))
        bounds=np.asarray(cfg["obstacle_size_range_m"],float)
        dx,dy,h=rng.uniform(bounds[:,0],bounds[:,1])
        yaw=float(rng.uniform(-np.pi,np.pi))
        if shape=="cylinder":
            mesh=trimesh.creation.cylinder(radius=.5,height=h,sections=24)
            mesh.apply_scale([dx,dy,1.])
        elif shape=="wedge":
            mesh=trimesh.convex.convex_hull([[-dx/2,-dy/2,-h/2],[dx/2,-dy/2,-h/2],
                [-dx/2,dy/2,-h/2],[dx/2,dy/2,-h/2],[-dx/2,dy/2,h/2],[dx/2,dy/2,h/2]])
        else:
            mesh=geo.box([-dx/2,-dy/2,-h/2],[dx/2,dy/2,h/2])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(yaw,[0,0,1]))
        mesh.apply_translation([x,y,h/2-.015])
        remember([mesh], "random_obstacles")
        obstacle_records.append(dict(shape=shape,size_m=[float(dx),float(dy),float(h)],
            construction_xy=[float(x),float(y)],construction_yaw_rad=yaw))
    west_obstacle_x=-8.-west_width+.35
    west_obstacle_y=.5+.7*west_depth
    solid([west_obstacle_x,west_obstacle_y,height-.01],
          [west_obstacle_x+.85,west_obstacle_y+.7,height+.8],"upper_deck_obstacles")
    east_obstacle_x=-1.5+.7*east_width
    east_obstacle_y=3.+.5*east_depth
    solid([east_obstacle_x,east_obstacle_y,height-.01],
          [east_obstacle_x+.75,east_obstacle_y+.8,height+.7],"upper_deck_obstacles")
    landmarks.append(dict(name="Random obstacle field", xyz=[-5.8,-7.,1.5]))

    # Compact low-wall passages and upright posts, inspired by the supplied RoM video.
    for lo,hi in [([-6.,-2.,-.02],[-1.,-1.7,1.1]),([.6,-2.,-.02],[5.5,-1.7,1.1]),
                  ([-6.,-2.,-.02],[-5.7,2.3,1.1]),([-6.,2.,-.02],[-1.2,2.3,1.1]),
                  ([.4,2.,-.02],[5.5,2.3,1.1]),([-3.8,.1,-.02],[-1.6,.4,.8]),
                  ([2.,-1.6,-.02],[2.3,.5,.8])]:
        solid(lo,hi,"low_playground_walls")
    prism([[5.2,-1.7],[5.5,-1.6],[4.6,2.2],[4.3,2.1]],
          [0,0,1.1],[0,0,-.02],"low_playground_walls")
    for x,y in ((-4.6,-.8),(3.6,.9)):
        post=trimesh.creation.cylinder(radius=.22,height=1.4,sections=24)
        post.apply_translation([x,y,.69])
        remember([post],"playground_posts")
    paths["low_wall_corridor"]=np.c_[np.full(61,-.2),np.linspace(-2.45,2.75,61),np.zeros(61)]
    landmarks.append(dict(name="Low-wall passages / posts",xyz=[0.,0.,1.6]))

    # Each unit gets an independent world-space footprint and planning boundary.
    zone_groups={
        "curved_bridge":["west_deck","east_deck","curved_narrow_bridge","upper_deck_obstacles"],
        "ramp":["west_ramp","ramp_landing"],
        "open_stairs":["open_riser_stairs","open_stair_landing"],
        "switchback_stairs":["switchback_stairs","switchback_top_landing"],
        "spiral_stairs":["spiral_stairs","spiral_connector"],
        "raised_platform":["east_ramp","underpass_landing"],
        "underpass":["arched_underpass","solid_reference_stairs"],
        "obstacles":["random_obstacles"],
        "low_walls":["low_playground_walls","playground_posts"],
    }
    path_zones={"curved_bridge":"curved_bridge","west_ramp":"ramp","open_stairs":"open_stairs",
        "switchback_lower":"switchback_stairs","switchback_upper":"switchback_stairs",
        "switchback_turn":"switchback_stairs","spiral_stairs":"spiral_stairs",
        "spiral_exit":"spiral_stairs","east_ramp":"raised_platform","platform_top":"raised_platform",
        "underpass":"underpass","solid_stairs":"underpass","low_wall_corridor":"low_walls"}
    landmark_zones=["curved_bridge","ramp","raised_platform","open_stairs","switchback_stairs",
                    "spiral_stairs","underpass","obstacles","low_walls"]
    zone_probe_points={name:np.vstack([p for key,p in paths.items() if path_zones[key]==name])
        for name in zone_groups if any(path_zones[key]==name for key in paths)}
    placements,layout_info=random_layout(groups,zone_groups,cfg,zone_probe_points)
    transforms,regions,building_meshes={},{},[]
    margin=float(cfg["planning_margin_m"])
    for name,names in zone_groups.items():
        zone_meshes=[m for group in names for m in groups[group]]
        matrix=placements[name]["matrix"]
        transforms[name]=matrix
        for m in zone_meshes:
            m.apply_transform(matrix)
        building_meshes.append(zone_meshes)
        bounds=np.array([m.bounds for m in zone_meshes])
        lo,hi=bounds[:,0,:2].min(axis=0),bounds[:,1,:2].max(axis=0)
        planning_lo,planning_hi=lo.copy(),hi.copy()
        if name in zone_probe_points:
            probe_world=transform_xy(zone_probe_points[name][:,:2],matrix)
            planning_lo=np.minimum(planning_lo,probe_world.min(axis=0))
            planning_hi=np.maximum(planning_hi,probe_world.max(axis=0))
        regions[name]=dict(id=name,feature_groups=names,structure_bounds_xy=[*lo.tolist(),*hi.tolist()],
            planning_bounds_xy=[*(planning_lo-margin).tolist(),*(planning_hi+margin).tolist()],
            construction_to_world=matrix.tolist(),yaw_rad=placements[name]["yaw_rad"],
            yaw_deg=float(np.degrees(placements[name]["yaw_rad"])),
            planning_rule="Choose both endpoints and constrain the entire teacher path to this region")
    roundtrip_errors=[]
    for name,p in paths.items():
        matrix=transforms[path_zones[name]]
        original=p[:,:2].copy()
        p[:,:2]=transform_xy(original,matrix)
        back=(p[:,:2]-matrix[:2,3])@matrix[:2,:2]
        roundtrip_errors.append(float(abs(back-original).max()))
    for item,name in zip(landmarks,landmark_zones):
        item["xyz"][:2]=transform_xy(item["xyz"][:2],transforms[name]).tolist()
    for flight in scene["stair_flights"]:
        name=("switchback_stairs" if flight["id"].startswith("switchback") else path_zones[flight["id"]])
        matrix=transforms[name]
        local_probes=flight.get("surface_probe_local_xyz",[[flight["x"],flight["y"],flight["base_z"]],
            [flight["x"],flight["end_y"],flight["top_z"]]])
        entry=transform_xy(np.asarray(local_probes)[0,:2],matrix)
        exit_xy=transform_xy(np.asarray(local_probes)[-1,:2],matrix)
        flight.update(region_id=name,geometry_frame="construction_local",construction_to_world=matrix.tolist(),
            entry_world_xyz=[*entry.tolist(),flight["base_z"]],exit_world_xyz=[*exit_xy.tolist(),flight["top_z"]],
            world_yaw_rad=placements[name]["yaw_rad"]+flight["direction"]*np.pi/2)
    scene["spiral"]["center_xy"]=transform_xy([cx,cy],transforms["spiral_stairs"]).tolist()
    scene["underpass"]["center_world_xy"]=transform_xy([ax,-3.],transforms["underpass"]).tolist()
    scene["obstacle_centers_xy"]=[transform_xy(p,transforms["obstacles"]).tolist() for p in occupied]
    for obstacle in obstacle_records:
        obstacle["world_xy"]=transform_xy(obstacle["construction_xy"],transforms["obstacles"]).tolist()
        obstacle["world_yaw_rad"]=obstacle["construction_yaw_rad"]+regions["obstacles"]["yaw_rad"]
    scene["obstacle_geometry"]=obstacle_records
    for opening in platform_openings:
        opening["world_center_xy"]=transform_xy(opening["construction_center_xy"],transforms[opening["region_id"]]).tolist()
    scene["platform_openings"]=platform_openings
    bounds=regions["obstacles"]["structure_bounds_xy"]
    perimeter_margin=min(.45,margin*.75)
    a,b=np.array(bounds[:2])-perimeter_margin,np.array(bounds[2:])+perimeter_margin
    paths["obstacle_perimeter_probe"]=np.vstack([
        np.c_[np.linspace(a[0],b[0],31),np.full(31,a[1]),np.zeros(31)],
        np.c_[np.full(31,b[0]),np.linspace(a[1],b[1],31),np.zeros(31)]])
    path_zones["obstacle_perimeter_probe"]="obstacles"
    scene["training_regions"],scene["path_regions"]=regions,path_zones
    scene["layout_generation"]=dict(layout_info,max_path_transform_roundtrip_error_m=max(roundtrip_errors))

    # Shared ground preserves the surrounding floor seen by the actual sensor.
    outline=[[-width/2+2,-depth/2],[width/2-2,-depth/2],[width/2,-depth/2+2],
        [width/2,depth/2-3],[width/2-3,depth/2],[-width/2+4,depth/2],
        [-width/2,depth/2-4],[-width/2,-depth/2+3]]
    base_meshes=[]
    st.add_prism(scene,base_meshes,outline,[0,0,0],[0,0,-slab],"base")
    matrix=transforms["curved_bridge"]
    pit_half_width=amplitude+bw/2+.2
    pit=geo.box([-7.25,5.-pit_half_width,-slab-.1],[-2.15,5.+pit_half_width,.1])
    pit.apply_transform(matrix)
    base=trimesh.boolean.difference([base_meshes[0],pit],engine="manifold")
    remember([base],"base_platform")
    pit_xy=transform_xy([[-7.25,5.-pit_half_width],[-2.15,5.-pit_half_width],
        [-2.15,5.+pit_half_width],[-7.25,5.+pit_half_width]],matrix)
    scene["base_openings"]=[dict(vertices_xy=pit_xy.tolist(),bottom="open")]
    scene["ground_outline_xy"]=outline
    scene["construction_records_frame"]="Pre-layout coordinates; apply training_regions transforms. Mesh is authoritative."
    groups={"base_platform":groups["base_platform"],**{k:v for k,v in groups.items() if k!="base_platform"}}

    mesh = union_buildings(building_meshes,base)
    for name, points in paths.items():
        for reverse in (False, True):
            a,b = (points[-1],points[0]) if reverse else (points[0],points[-1])
            direction=points[-2,:2]-points[-1,:2] if reverse else points[1,:2]-points[0,:2]
            scene["queries"].append(dict(id=name+("_reverse" if reverse else "_forward"),
                start_support_xyz=a.tolist(), goal_support_xyz=b.tolist(),
                start_yaw=float(np.arctan2(direction[1],direction[0])),
                region_id=path_zones[name],planning_bounds_xy=regions[path_zones[name]]["planning_bounds_xy"],
                role="geometry_probe_endpoints_not_validated_navigation_tasks"))
    scene.update(feature_groups={k:len(v) for k,v in groups.items()}, landmarks=landmarks,
                 randomized_bridge_amplitude_m=amplitude,
                 geometry_semantics="terrain.obj is authoritative, analytic scene solids are construction provenance")
    return scene, mesh, groups, paths


def vertical_hits(mesh, xy, upwards_only=False):
    """Independent vertical triangle intersections without optional rtree."""
    triangles = mesh.triangles
    a, e1, e2 = triangles[:,0], triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]
    d = e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
    valid = d > 1e-10 if upwards_only else abs(d)>1e-10
    a,e1,e2,d = a[valid],e1[valid],e2[valid],d[valid]
    out = []
    for p in np.asarray(xy):
        q = p[:2]-a[:,:2]
        u = (q[:,0]*e2[:,1]-q[:,1]*e2[:,0])/d
        v = (e1[:,0]*q[:,1]-e1[:,1]*q[:,0])/d
        inside = (u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)
        z = a[:,2]+u*e1[:,2]+v*e2[:,2]
        out.append(np.unique(np.round(z[inside],6)))
    return out


def validate(scene, mesh, paths):
    checks = {}
    checks["watertight"] = bool(mesh.is_watertight)
    checks["positive_volume"] = bool(mesh.is_volume)
    checks["single_connected_component"] = len(mesh.split()) == 1
    path_errors = {}
    for name, points in paths.items():
        hits = vertical_hits(mesh, points[:,:2], upwards_only=True)
        err = [min(abs(z-p[2])) if len(z) else float("inf") for z,p in zip(hits,points)]
        path_errors[name] = float(max(err))
        checks[name+"_support_surface"] = path_errors[name] < 2e-5
    # The bridge has no floor below its centre. The tunnel has free volume below the roof.
    bridge=paths["curved_bridge"]
    z = vertical_hits(mesh,[bridge[len(bridge)//2,:2]])[0]
    checks["real_void_below_curved_bridge"] = len(z)>0 and z.min()>1.
    tunnel=scene["underpass"]
    z = vertical_hits(mesh,[tunnel["center_world_xy"]])[0]
    checks["underpass_has_ground_and_overhead_surface"] = (np.any(abs(z)<1e-5)
        and np.any(abs(z-scene["underpass"]["central_clearance_m"])<1e-5))
    # A mid-flight point must retain free space beneath the open-riser stairs.
    flight = next(f for f in scene["stair_flights"] if f["id"]=="open_stairs")
    j = flight["n_steps"]//2
    xy = flight["surface_probe_local_xyz"][j][:2]
    xy=transform_xy(xy,flight["construction_to_world"])
    z = vertical_hits(mesh,[xy])[0]
    checks["open_stairs_not_filled_solid"] = np.any(abs(z)<1e-5) and np.any(abs(z-(j*flight["riser"]-.055))<1e-5)
    spiral = paths["spiral_stairs"]
    z = vertical_hits(mesh,[spiral[2,:2]],upwards_only=True)[0]
    checks["spiral_retains_multiple_heights_at_same_xy"] = np.sum(z>.01)>=2
    for i,opening in enumerate(scene["platform_openings"]):
        z=vertical_hits(mesh,[opening["world_center_xy"]])[0]
        checks[f"platform_opening_{i}_removes_upper_surface"]=not np.any(abs(z-scene["height"])<1e-5)
    regions=scene["training_regions"]
    distances=[]
    names=list(regions)
    for i,a in enumerate(names):
        ba=np.array(regions[a]["structure_bounds_xy"])
        for b in names[i+1:]:
            bb=np.array(regions[b]["structure_bounds_xy"])
            gap=np.maximum(0,np.maximum(ba[:2]-bb[2:],bb[:2]-ba[2:]))
            distances.append(float(np.linalg.norm(gap)))
    minimum_gap=min(distances)
    required_gap=float(scene["config"]["minimum_structure_gap_m"])
    checks["structure_gap_meets_configured_minimum"]=minimum_gap+1e-7>=required_gap
    checks["planning_bounds_do_not_overlap"]=True
    checks["planning_regions_inside_ground_bounds"]=True
    checks["planning_regions_inside_ground_outline"]=True
    polygon=np.asarray(scene["ground_outline_xy"])
    edges=np.roll(polygon,-1,axis=0)-polygon
    for i,name in enumerate(names):
        ba=np.array(regions[name]["planning_bounds_xy"])
        half=np.array([scene["width"],scene["depth"]])/2
        if np.any(ba[:2]<-half) or np.any(ba[2:]>half):
            checks["planning_regions_inside_ground_bounds"]=False
        corners=np.array([[ba[0],ba[1]],[ba[2],ba[1]],[ba[2],ba[3]],[ba[0],ba[3]]])
        delta=corners[:,None,:]-polygon[None,:,:]
        cross=edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
        if not (cross>=-1e-7).all():
            checks["planning_regions_inside_ground_outline"]=False
        for other in names[i+1:]:
            bb=np.array(regions[other]["planning_bounds_xy"])
            if np.all(ba[:2]<bb[2:]) and np.all(bb[:2]<ba[2:]):
                checks["planning_bounds_do_not_overlap"]=False
    angles=[r["yaw_deg"] for r in regions.values()]
    checks["all_buildings_have_distinct_seeded_yaw"]=len(set(round(a,6) for a in angles))==len(angles)
    checks["path_transforms_preserve_local_geometry"]=scene["layout_generation"]["max_path_transform_roundtrip_error_m"]<1e-10
    a=np.array(regions["raised_platform"]["structure_bounds_xy"])
    b=np.array(regions["underpass"]["structure_bounds_xy"])
    gap=np.maximum(0,np.maximum(a[:2]-b[2:],b[:2]-a[2:]))
    detached_gap=float(np.linalg.norm(gap))
    checks["large_platform_separated_from_underpass"]=detached_gap+1e-7>=required_gap
    for name,p in paths.items():
        bounds=np.array(regions[scene["path_regions"][name]]["planning_bounds_xy"])
        checks[name+"_inside_own_region"]=bool(((p[:,:2]>=bounds[:2]-1e-7)&(p[:,:2]<=bounds[2:]+1e-7)).all())
    checks = {name: bool(value) for name, value in checks.items()}
    report = dict(passed=all(checks.values()), checks=checks,
                  path_support_max_error_m=path_errors,
                  vertices=len(mesh.vertices), triangles=len(mesh.faces),
                  bounds_world=mesh.bounds.tolist(), components=len(mesh.split()),
                  minimum_structure_gap_m=minimum_gap,
                  platform_underpass_gap_m=detached_gap,
                  building_yaw_degrees={n:r["yaw_deg"] for n,r in regions.items()},
                  scope="Static mesh, open volumes and probe support surfaces. No policy, gait or dynamic traversability validation.")
    return report
