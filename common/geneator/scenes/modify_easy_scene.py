#!/usr/bin/env python3
"""对已有两层示例做可追溯的小幅修改，保留原始 VST 调试图。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
os.environ.setdefault("CUDA_PATH","/usr/local/cuda")
os.environ.setdefault("CUPY_CACHE_DIR",str(ROOT/".cache/cupy"))
os.environ["PYTHONPATH"]=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH","")
import numpy as np
import trimesh
from stair_nav.scenes.geometry import box,rotation,sample_surface,to_world,to_local
from stair_nav.core.io import read_json,write_json,write_pcd,sha256
from stair_nav.scenes.stair_structure import add_flight


def subtract(rectangle,holes):
    """从矩形依次扣除矩形洞口，返回剩余互不重叠的矩形块列表。

    rectangle/holes 均采用局部 [x0,y0,x1,y1]。每次重叠区域最多切出四条边带，
    过滤零面积块；后续构造楼板时按这些剩余块建实体，洞口不会只停留在元数据。
    """
    result=[rectangle]
    for hole in holes:
        pieces=[]
        for x0,y0,x1,y1 in result:
            a,b,c,d=max(x0,hole[0]),max(y0,hole[1]),min(x1,hole[2]),min(y1,hole[3])
            if a>=c or b>=d:pieces.append([x0,y0,x1,y1]);continue
            for rect in ([x0,y0,a,y1],[c,y0,x1,y1],[a,y0,c,b],[a,d,c,y1]):
                if rect[0]<rect[2] and rect[1]<rect[3]:pieces.append(rect)
        result=pieces
    return result


def edit_geometry(source,neck,relief):
    """复制旧两层 easy 场景，在保留建筑基准尺寸下修改踏板宽度、孔洞和坡面。

    neck 为最窄梯宽、relief 为下层起伏高度，单位均为米；保留原 yaw、层高和
    楼梯中心线，在新 scene 中同步重建实体、支撑面与额外桥上/桥下查询。
    返回 (scene,世界网格)，不写源文件；仅接受旧版基础两层场景，避免误解释多层结构。
    """
    old=read_json(source/"scene.json")
    if "floors" in old or "stairs" not in old:raise ValueError("Expected the saved basic two-floor easy scene")
    scene=deepcopy(old);st=scene["stairs"];h,w,d=scene["height"],scene["width"],scene["depth"]
    slab=scene["slab_thickness"]
    if not 0<neck<=st["width"] or not 0<=relief<.4:raise ValueError("Neck must fit the original stair width; relief must be in [0,0.4)")
    holes=[scene["opening"],[scene["opening"][2]+.7,-d/2+1.2,w/2-2.,-1.],
           [scene["opening"][2]+.7,2.2,w/2-2.,d/2-1.2]]
    if any(a>=c or b>=e for a,b,c,e in holes):raise ValueError("Source geometry is too small for these local openings")
    scene.update(name=f"{old['name']}_edit01",layout="saved_easy_edit",source_scene=str(source),
        floors=[dict(id=0,z=0.,openings=[]),dict(id=1,z=h,openings=holes)],
        rooms=[],doorways=[],relief=[],support_surfaces=[],solids=[],
        modifications=dict(stair_neck_width_m=neck,relief_height_m=relief,new_upper_openings=holes[1:],
                           fixed="Source width/depth, yaw, level height, stair centerline, rise, tread depth, landing and outer walls"))
    meshes=[]
    def flat(lo,hi,kind,walkable=False,**extra):
        item=dict(lo=list(map(float,lo)),hi=list(map(float,hi)),kind=kind,**extra)
        scene["solids"].append(item);meshes.append(box(lo,hi))
        if walkable:
            scene["support_surfaces"].append(dict(kind=kind,vertices_xy=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],height_plane=[0.,0.,hi[2]]))
    for item in old["solids"]:
        if item["kind"] in ("stair","landing","upper_floor","stair_stringer","stair_rail","stair_baluster"):continue
        flat(item["lo"],item["hi"],item["kind"],item["kind"]=="lower_floor")
    for a,b,c,e in subtract([-w/2,-d/2,w/2,d/2],holes):flat([a,b,h-slab],[c,e,h],"upper_floor",True)
    widths=st["width"]-(st["width"]-neck)*np.sin(np.linspace(0,np.pi,st["n_steps"]+1))**2
    st["tread_widths"]=widths.tolist();st["neck_width_m"]=neck
    scene["stair_flights"]=[dict(**st,id=0,from_floor=0,to_floor=1,direction=1,
                                  base_z=0.,top_z=h,length=st["end_y"]-st["y"])]
    add_flight(scene, meshes, scene["stair_flights"][0])
    ys=np.array([-d/2+1.2,-2.4,-.4,1.2,3.2,d/2-1.2]);zs=np.array([0,relief,0,.8*relief,0,0])
    if relief:
        for ya,yb,za,zb in zip(ys[:-1],ys[1:],zs[:-1],zs[1:]):
            x0,x1=.0,2.0;poly=[[x0,ya],[x1,ya],[x1,yb],[x0,yb]]
            slope=(zb-za)/(yb-ya);plane=[0.,slope,za-slope*ya]
            xy=np.asarray(poly);top=np.c_[xy,np.ones(4)]@plane
            vertices=np.vstack([np.c_[xy,np.full(4,-slab)],np.c_[xy,top]])
            meshes.append(trimesh.convex.convex_hull(vertices))
            scene["solids"].append(dict(lo=vertices.min(axis=0).tolist(),hi=vertices.max(axis=0).tolist(),kind="terrain_ramp",
                 vertices_xy=poly,top_plane=plane,bottom_plane=[0.,0.,-slab]))
            scene["support_surfaces"].append(dict(kind="terrain_ramp",vertices_xy=poly,height_plane=plane))
    scene["lower_relief_profile"]=dict(x_bounds=[0.,2.],y=ys.tolist(),z=zs.tolist())
    for name,a,b,floor in (
        ("upper_bridge",to_local(old["queries"][0]["goal_support_xyz"],old),[w/2-.9,.7,h],1),
        ("under_bridge",[1.,ys[0]-.3,0.],[1.,ys[-1]+.3,0.],0)):
        scene["queries"].append(dict(id=name,start_support_xyz=to_world(a,scene).tolist(),goal_support_xyz=to_world(b,scene).tolist(),
            start_yaw=0.,required_transition="same_floor",start_floor=floor,goal_floor=floor))
    cfg=deepcopy(old["config"]);cfg.update(name=scene["name"],layout=scene["layout"],source_scene=str(source),modifications=scene["modifications"])
    scene["config"]=cfg
    mesh=trimesh.boolean.union(meshes,engine="manifold")
    transform=np.eye(4);transform[:3,:3]=rotation(scene["yaw"]);mesh.apply_transform(transform)
    if not mesh.is_volume or not mesh.is_watertight:raise ValueError("Edited geometry must remain a closed positive-volume solid")
    scene.update(mesh_watertight=True,bounds_world=mesh.bounds.tolist())
    return scene,mesh


def preview(source,output,scene,mesh):
    """为修改后的场景输出三维、梯宽和逐层路线对比，以及 VST 原始图对照。

    source/output 分别表示来源与新产物目录，传入 scene/mesh 已包含修改几何。
    仅绘制已经成功优化的路线；baseline/current JPEG 保持原始方向，不据切片编号
    推断简化层编号，也不把调试图当作完整碰撞验收。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(16,10));grid=fig.add_gridspec(2,2,left=.05,right=.97,bottom=.07,top=.91,hspace=.3,wspace=.25)
    ax=fig.add_subplot(grid[0,0],projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor=(.4,.58,.7,.18),edgecolor=(.25,.4,.5,.12),linewidth=.25))
    for item in scene.get("obstacles", []):
        xy=np.asarray(item["vertices_xy"])
        vertices=np.vstack([np.c_[xy,np.full(len(xy),item["lo"][2])],np.c_[xy,np.full(len(xy),item["hi"][2])]])
        obstacle=trimesh.convex.convex_hull(vertices)
        obstacle.vertices=to_world(obstacle.vertices,scene)
        color=np.asarray(item["color_rgb"])/255
        ax.add_collection3d(Poly3DCollection(obstacle.triangles,facecolor=color,edgecolor="none",alpha=.9))
    lo,hi=mesh.bounds
    ax.set(xlim=[lo[0],hi[0]],ylim=[lo[1],hi[1]],zlim=[0,hi[2]],xlabel="World X / m",ylabel="World Y / m",zlabel="Z / m",title="Saved building + local additions")
    ax.set_box_aspect(np.maximum(hi-lo,1));ax.view_init(elev=32,azim=-48)
    profile=fig.add_subplot(grid[0,1]);st=scene["stairs"]
    profile.plot(np.arange(len(st["tread_widths"]))*st["step_depth"],st["tread_widths"],"o-",color="#c87b32")
    profile.axhline(st["width"],color="#8494a5",ls="--",label="Original width")
    profile.set(xlabel="Distance along stair / m",ylabel="Width / m",title="Same stairs, gentle width change");profile.legend();profile.grid(alpha=.2)
    report=read_json(output/"paths.report.json") if (output/"paths.report.json").exists() else dict(queries=[])
    success={q["id"] for q in report["queries"] if q.get("optimized")}
    colors=["#d94e39","#176caa","#26996e","#a54dad"]
    for floor in scene["floors"]:
        plan=fig.add_subplot(grid[1,floor["id"]]);base=floor["z"];w,d=scene["width"],scene["depth"]
        plan.add_patch(Rectangle((-w/2,-d/2),w,d,facecolor="#e4edf4"))
        for a,b,c,e in floor["openings"]:plan.add_patch(Rectangle((a,b),c-a,e-b,facecolor="white",edgecolor="#a8b5c2",hatch="//",lw=.6))
        if floor["id"]==0:
            p=scene["lower_relief_profile"]
            for ya,yb,za,zb in zip(p["y"][:-1],p["y"][1:],p["z"][:-1],p["z"][1:]):
                plan.add_patch(Rectangle((p["x_bounds"][0],ya),2.,yb-ya,facecolor=plt.get_cmap("YlOrBr")(.2+(za+zb)),edgecolor="#bc9955",lw=.5))
        for j,width in enumerate(st["tread_widths"]):
            y=st["y"]+j*st["step_depth"]
            plan.add_patch(Rectangle((st["x"]-width/2,y),width,st["step_depth"],facecolor="#e3c49b",edgecolor="#b78a4b",lw=.5))
        for item in scene["solids"]:
            if item["kind"]!="wall":continue
            a,b=np.array(item["lo"]),np.array(item["hi"]);plan.add_patch(Rectangle(a[:2],*(b-a)[:2],color="#52616e"))
        for item in scene.get("obstacles", []):
            if item["floor"]!=floor["id"]:continue
            color=np.asarray(item["color_rgb"])/255
            plan.add_patch(Polygon(item["vertices_xy"],facecolor=color,edgecolor="#374858",lw=.7))
            plan.text(*item["center_xy"],item["id"],ha="center",va="center",fontsize=7,color="white",fontweight="bold")
        for query,color in zip(scene["queries"],colors):
            file=output/"paths"/f"{query['id']}.gpmp.csv"
            if query["id"] in success and file.exists():
                world=np.atleast_2d(np.loadtxt(file,delimiter=",",skiprows=1))[:,:3]
                xyz=to_local(world,scene)
                shown=(xyz[:,2]>=base-.3)&(xyz[:,2]<=base+.5)
                xy=np.where(shown[:,None],xyz[:,:2],np.nan)
                plan.plot(xy[:,0],xy[:,1],color=color,lw=1.5,label=query["id"])
                if floor["id"]==0:ax.plot(world[:,0],world[:,1],world[:,2]+.5,color=color,lw=1.5)
            for name,marker in (("start","o"),("goal","^")):
                if query[f"{name}_floor"]!=floor["id"]:continue
                p=to_local(query[f"{name}_support_xyz"],scene);plan.scatter(*p[:2],color=color,marker=marker,s=20,zorder=5)
        plan.set(xlim=[-w/2-.2,w/2+.2],ylim=[-d/2-.2,d/2+.2],aspect="equal",title=f"Floor {floor['id']+1} | local XY | z={base:.2f} m")
        if plan.lines:plan.legend(fontsize=7)
    addition=f" | {len(scene['obstacles'])} obstacles" if scene.get("obstacles") else ""
    fig.suptitle(f"{scene['name']}{addition} | building and stairs retained",fontsize=15)
    fig.savefig(output/"preview.png",dpi=145);plt.close(fig)
    # 另存原始调试图的并排对比，保留后端 JPEG 原件。
    from PIL import Image,ImageDraw
    baseline=output/"vst_debug/baseline"
    old_files=sorted(baseline.glob("w_local_map*_binary.jpg"))
    if old_files:
        sheet=Image.new("RGB",(len(old_files)*180,410),"#edf2f7");draw=ImageDraw.Draw(sheet)
        draw.text((8,5),"PREVIOUS SCENE (top) / CURRENT SCENE (bottom) - raw pre-skeleton binary, original orientation",fill="#304357")
        for i,file in enumerate(old_files):
            draw.text((i*180+10,25),file.stem.replace("w_local_map","slice ").replace("_binary",""),fill="#304357")
            for y,root in ((48,baseline),(230,output/"vst_debug")):
                candidate=root/file.name
                if candidate.exists():
                    pic=Image.open(candidate).convert("RGB");pic.thumbnail((165,165));sheet.paste(pic,(i*180+8,y))
        sheet.save(output/"vst_debug/comparison.png")


def main():
    """解析修改参数，在新输出目录生成可追溯产物并保持源目录所有文件指纹不变。

    --geometry-only 跳过 GPU 建图与规划；否则调用隔离 worker 重建新图和曲线。
    保存 source_snapshot/run 报告并在有成功路线时生成共享回放数据。
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"outputs/upstream_f5580008/easy_00043")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stair-neck-width",type=float,default=1.3)
    parser.add_argument("--relief-height",type=float,default=.18)
    parser.add_argument("--geometry-only",action="store_true",help="Skip GPU mapping/planning; baseline debug is still copied")
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    if output.exists() or source==output or source in output.parents:parser.error("Use a new output directory outside the source scene")
    fingerprints={str(f.relative_to(source)):sha256(f) for f in sorted(source.rglob("*")) if f.is_file()}
    scene,mesh=edit_geometry(source,args.stair_neck_width,args.relief_height)
    output.mkdir(parents=True,exist_ok=False)
    if (source/"vst_debug").exists():shutil.copytree(source/"vst_debug",output/"vst_debug/baseline")
    mesh.export(output/"terrain.obj")
    points=sample_surface(mesh,scene["config"]["sample_spacing"]);points=points[points[:,2]>=-1e-6]
    write_pcd(output/"terrain.pcd",points);scene.update(point_count=len(points),pcd_sha256=sha256(output/"terrain.pcd"))
    write_json(output/"scene.json",scene);write_json(output/"config.resolved.json",scene["config"])
    write_json(output/"source_snapshot.json",dict(source=str(source),sha256=fingerprints))
    status=dict(stage="modified_geometry",seed=scene["seed"],source=str(source),geometry_validation="disabled",success_criterion="gpmp_output")
    write_json(output/"run.json",status)
    if not args.geometry_only:
        from stair_nav.app.cli import worker
        import subprocess
        try:
            worker("tomography",output);worker("plan",output)
            report=read_json(output/"paths.report.json")
            status.update(stage="complete",optimized_count=report["optimized_count"],query_count=len(report["queries"]),
                          accepted=report["optimized_count"]==len(report["queries"]))
        except (RuntimeError,subprocess.TimeoutExpired) as exc:status.update(stage="generation_failed",accepted=False,error=str(exc))
    preview(source,output,scene,mesh)
    for name,digest in fingerprints.items():
        if sha256(source/name)!=digest:raise RuntimeError(f"Source artifact changed: {name}")
    status.update(source_unchanged=True,baseline_debug_files=len(list((output/"vst_debug/baseline").glob("*"))),
                  current_debug_files=len(list((output/"vst_debug").glob("w_local_map*_binary.jpg"))),preview="preview.png")
    if not args.geometry_only and (output/"paths.report.json").exists():
        from stair_nav.visualization.scene import build_scene_viewer
        if any(q.get("optimized") for q in read_json(output/"paths.report.json")["queries"]):
            build_scene_viewer(output)
            status["viewer"]="viewer_data.json"
    write_json(output/"run.json",status);print(json.dumps(status,indent=2),flush=True)
    return 2 if status.get("accepted") is False else 0


if __name__=="__main__":raise SystemExit(main())
