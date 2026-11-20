"""Scene assets and self-contained inspection views."""
import os
from pathlib import Path

import numpy as np
import trimesh

from .terrain import load_backend, validate


PALETTE = {
    "base_platform": "#cbd2d7", "west_deck": "#99a3ab", "east_deck": "#99a3ab",
    "curved_narrow_bridge": "#8aa8a0", "west_ramp": "#b6a685", "east_ramp": "#b6a685",
    "underpass_landing": "#99a3ab", "open_riser_stairs": "#ac8d72",
    "spiral_stairs": "#9c8fa7", "spiral_connector": "#99a3ab",
    "arched_underpass": "#a19a90", "solid_reference_stairs": "#96a1a5",
    "random_obstacles": "#79b2c2", "upper_deck_obstacles": "#78848d",
    "switchback_stairs":"#b29c82",
    "ramp_landing":"#99a3ab", "open_stair_landing":"#99a3ab", "switchback_top_landing":"#99a3ab",
    "low_playground_walls":"#9fa6ac", "playground_posts":"#79b2c2",
}
LABELS = {
    "base_platform":"主体平台", "west_deck":"曲桥端平台 A", "east_deck":"曲桥端平台 B",
    "curved_narrow_bridge":"S 形窄桥", "west_ramp":"独立斜坡", "east_ramp":"大平台坡道",
    "underpass_landing":"独立大平台", "open_riser_stairs":"空心直梯",
    "spiral_stairs":"450° 旋转楼梯", "spiral_connector":"楼梯上层连接道",
    "arched_underpass":"拱形桥洞", "solid_reference_stairs":"实心参考楼梯",
    "random_obstacles":"地面障碍物", "upper_deck_obstacles":"高台障碍物",
    "switchback_stairs":"U 形折返楼梯",
    "ramp_landing":"缓坡顶平台", "open_stair_landing":"直梯顶平台", "switchback_top_landing":"折返梯顶平台",
    "low_playground_walls":"低墙通道", "playground_posts":"通道立柱",
}
ZONE_LABELS={"curved_bridge":"弯曲窄桥区","ramp":"斜坡区","open_stairs":"空心直梯区",
    "switchback_stairs":"折返楼梯区","spiral_stairs":"旋转楼梯区","underpass":"桥洞区","obstacles":"障碍区",
    "low_walls":"低墙通道区", "raised_platform":"独立大平台区"}


def views(output, scene, groups):
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent.parent / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go

    meshes = {name: trimesh.util.concatenate(items) for name, items in groups.items()}
    fig = plt.figure(figsize=(15,10), facecolor="#f5f7fa")
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    for name, mesh in meshes.items():
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolor=PALETTE[name],
            edgecolor=(.12,.16,.2,.13), linewidths=.12,
            zorder=0 if name=="base_platform" else 1))
    for i, item in enumerate(scene["landmarks"],1):
        x,y,z = item["xyz"]
        ax.text(x,y,z+.18,str(i),ha="center",va="bottom",fontsize=12,weight="bold",color="#18293b",zorder=5)
    w,d,h = scene["width"],scene["depth"],scene["height"]
    ax.set(xlim=(-w/2,w/2),ylim=(-d/2,d/2),zlim=(-.35,h+1.1),
           xlabel="X [m]",ylabel="Y [m]",zlabel="Z [m]")
    ax.set_box_aspect((w,d,(h+1.45)*1.7))
    ax.set_zticks([0,h])
    ax.set_proj_type("ortho")
    ax.view_init(elev=33,azim=-65)
    ax.set_title(f"RoM-inspired mixed playground  |  seed {scene['seed']}\n"
                 f"{w:g} x {d:g} m  |  upper level {h:g} m  |  visual Z scale 1.7x",pad=20,fontsize=16)
    descriptions=[f"{i}. {v['name']}" for i,v in enumerate(scene["landmarks"],1)]
    text="    ".join(descriptions[:5])+"\n"+"    ".join(descriptions[5:])
    fig.text(.05,.028,text,fontsize=9,color="#33465b")
    fig.subplots_adjust(left=0,right=1,bottom=.09,top=.91)
    fig.savefig(output/"preview.png",dpi=150)
    plt.close(fig)

    count=len(scene["training_regions"])
    cols=3 if count>8 else 4
    rows=int(np.ceil(count/cols))
    fig=plt.figure(figsize=(cols*6,rows*5),facecolor="#f5f7fa")
    for index,(name,region) in enumerate(scene["training_regions"].items(),1):
        ax=fig.add_subplot(rows,cols,index,projection="3d")
        ax.computed_zorder=False
        lo=np.array(region["structure_bounds_xy"][:2])-1.
        hi=np.array(region["structure_bounds_xy"][2:])+1.
        clip=trimesh.creation.box([*(hi-lo),.6])
        clip.apply_translation([*((lo+hi)/2),-.1])
        ground=trimesh.boolean.intersection([meshes["base_platform"],clip],engine="manifold")
        ax.add_collection3d(Poly3DCollection(ground.triangles,facecolor=PALETTE["base_platform"],zorder=0))
        for group in region["feature_groups"]:
            ax.add_collection3d(Poly3DCollection(meshes[group].triangles,facecolor=PALETTE[group],
                edgecolor=(.1,.1,.1,.12),linewidths=.12,zorder=1))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(-.3,h+1.),
               xlabel="X [m]",ylabel="Y [m]",zlabel="Z [m]",title=name.replace("_"," ").title())
        ax.set_box_aspect((*(hi-lo),(h+1.3)*1.6))
        ax.set_proj_type("ortho")
        ax.view_init(elev=30,azim=(-135 if name=="switchback_stairs" else -62)+region["yaw_deg"])
    if len(scene["training_regions"])<8:
        ax=fig.add_subplot(2,4,8)
        ax.axis("off")
        ax.text(.05,.8,f"{scene['width']:g} x {scene['depth']:g} m shared ground\n\n"
            f"{len(scene['training_regions'])} training regions\n\nIndependent planning bounds\n\nShared sensor geometry",
            fontsize=15,va="top",color="#33465b")
    fig.tight_layout()
    fig.savefig(output/"details.png",dpi=140)
    plt.close(fig)

    figure = go.Figure()
    for name, mesh in meshes.items():
        v,f = mesh.vertices,mesh.faces
        figure.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            name=LABELS[name],color=PALETTE[name],flatshading=True,showlegend=True,
            legendgroup=name,lighting=dict(ambient=.7,diffuse=.8,specular=.05),
            hovertemplate="X %{x:.2f} m<br>Y %{y:.2f} m<br>Z %{z:.2f} m<extra>%{fullData.name}</extra>"))
    region_buttons=[dict(label="总览",method="relayout",args=[{
        "scene.xaxis.autorange":True,"scene.yaxis.autorange":True,"scene.zaxis.autorange":True,
        "scene.aspectratio":dict(x=1.,y=d/w,z=(h+1.45)*1.7/w),
        "scene.camera.eye":dict(x=.8,y=-1.,z=.75)}])]
    for name,region in scene["training_regions"].items():
        x0,y0,x1,y1=region["planning_bounds_xy"]
        yaw=region["yaw_rad"]
        rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        eye=rotation@np.array([-.8 if name=="switchback_stairs" else .8,-1.])
        region_buttons.append(dict(label=f"{ZONE_LABELS[name]} · {region['yaw_deg']:.0f}°",method="relayout",args=[{
            "scene.xaxis.range":[x0,x1],"scene.yaxis.range":[y0,y1],"scene.zaxis.range":[-.3,h+1.],
            "scene.aspectratio":dict(x=1.,y=(y1-y0)/(x1-x0),z=(h+1.3)*1.7/(x1-x0)),
            "scene.camera.eye":dict(x=float(eye[0]),y=float(eye[1]),z=.75)}]))
    figure.update_layout(title=dict(text=f"随机朝向 Playground · {w:g} × {d:g} m · seed {scene['seed']}<br>"
        "<sup>选择训练区查看细节，拖动旋转，滚轮缩放。显示高度放大不改变网格的米制坐标。</sup>",x=.03),
        margin=dict(l=5,r=5,b=5,t=90),paper_bgcolor="#f5f7fa",legend=dict(x=1.,y=1.),
        scene=dict(aspectmode="manual",aspectratio=dict(x=1.,y=d/w,z=(h+1.45)*1.7/w),
            xaxis_title="X / m",yaxis_title="Y / m",zaxis_title="Z / m",
            camera=dict(eye=dict(x=.8,y=-1.,z=.75)),dragmode="orbit"),
        updatemenus=[dict(type="dropdown",x=.04,y=1.03,buttons=region_buttons),
            dict(type="buttons",direction="left",x=.4,y=1.03,
            buttons=[dict(label="斜视",method="relayout",args=[{"scene.camera.eye":dict(x=.8,y=-1.,z=.75),"scene.camera.up":dict(x=0,y=0,z=1)}]),
                     dict(label="俯视",method="relayout",args=[{"scene.camera.eye":dict(x=0,y=0,z=1.7),"scene.camera.up":dict(x=0,y=1,z=0)}]),
                     dict(label="实际比例",method="relayout",args=[{"scene.aspectratio":dict(x=1.,y=d/w,z=(h+1.45)/w)}]),
                     dict(label="高度放大",method="relayout",args=[{"scene.aspectratio":dict(x=1.,y=d/w,z=(h+1.45)*1.7/w)}])])])
    html = figure.to_html(full_html=True,include_plotlyjs=True,
                          config=dict(displaylogo=False,responsive=True,scrollZoom=True))
    html = html.replace("<head>","<head><meta charset='utf-8'><title>Tomography · 组合通行平台</title>"
        "<style>html,body{margin:0;height:100%;background:#f5f7fa}</style>",1)
    (output/"index.html").write_text(html,encoding="utf-8")


def save(output, cfg, scene, mesh, groups, paths, backend_root, include_points=True):
    _, _, _ = load_backend(backend_root)
    from stair_nav.core.io import sha256, write_json, write_pcd
    from stair_nav.scenes.geometry import sample_surface
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = validate(scene,mesh,paths)
    write_json(output/"geometry_validation.json",report)
    if not report["passed"]:
        raise RuntimeError(f"Geometry validation failed: {report['checks']}")
    mesh.export(output/"terrain.obj")
    mesh.export(output/"terrain.stl")
    colored = trimesh.Scene()
    for name, items in groups.items():
        part = trimesh.util.concatenate(items)
        color = bytes.fromhex(PALETTE[name][1:])
        part.visual.face_colors = [*color,255]
        colored.add_geometry(part,node_name=name,geom_name=name)
    colored.export(output/"terrain.glb")
    scene.update(bounds_world=mesh.bounds.tolist(),mesh_watertight=True,
                 mesh_sha256=sha256(output/"terrain.obj"),pcd_ground_clip_z=0.,
                 tomography_status="not_generated",training_status="not_started")
    if include_points:
        points = sample_surface(mesh,cfg["sample_spacing_m"])
        points = points[points[:,2]>=-1e-6]
        write_pcd(output/"terrain.pcd",points)
        scene.update(point_count=len(points),pcd_sha256=sha256(output/"terrain.pcd"))
    write_json(output/"scene.json",scene)
    write_json(output/"config.resolved.json",scene["config"])
    write_json(output/"training_regions.json",dict(regions=scene["training_regions"],
        path_regions=scene["path_regions"],
        sensor_geometry="Use full terrain.obj; planning crops must not create artificial LiDAR floor edges"))
    write_json(output/"geometry_paths.json",dict(
        role="Surface probes and endpoint proposals, not teacher actions or certified robot routes",
        paths={name:points.tolist() for name,points in paths.items()}))
    views(output,scene,groups)
    return report

