#!/usr/bin/env python3
"""Export an overview of the actual saved native GPMP paths."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    args=p.parse_args()
    scene=args.scene.resolve()
    model=json.loads((scene/"scene.json").read_text())
    records=json.loads((scene/"routes.report.json").read_text())["queries"]
    good=[r for r in records if r["optimized"]]
    names=list(dict.fromkeys(r["region_id"] for r in good))
    colors={name:plt.get_cmap("tab10")(i) for i,name in enumerate(names)}
    mesh=trimesh.load(scene/"terrain.obj",force="mesh",process=False)
    fig=plt.figure(figsize=(16,10),facecolor="#f5f7fa")
    ax=fig.add_subplot(111,projection="3d")
    ax.computed_zorder=False
    ground=mesh.triangles[:,:,2].max(axis=1)<=1e-5
    ax.add_collection3d(Poly3DCollection(mesh.triangles[ground],facecolor="#d5dce2",edgecolor="none",zorder=0))
    ax.add_collection3d(Poly3DCollection(mesh.triangles[~ground],facecolor="#9ca8b2",
        edgecolor=(.2,.25,.3,.08),linewidths=.12,zorder=1))
    labelled=set()
    for r in good:
        path=np.load(scene/"paths"/(r["id"]+".gpmp.npy"))
        name=r["region_id"]
        label=name.replace("_"," ").title() if name not in labelled else None
        ax.plot(path[:,0],path[:,1],path[:,2]+.08,color=colors[name],lw=2.4,label=label,zorder=5)
        ax.scatter(*path[0],color=colors[name],s=14,zorder=6)
        labelled.add(name)
    w,d=model["width"],model["depth"]
    ax.set(xlim=(-w/2,w/2),ylim=(-d/2,d/2),zlim=(-.3,3.7),xlabel="X [m]",ylabel="Y [m]",zlabel="Z [m]")
    ax.set_zticks([0,2.4])
    ax.set_box_aspect((w,d,4.*1.7))
    ax.set_proj_type("ortho")
    ax.view_init(elev=38,azim=-62)
    fig.suptitle(f"{len(good)} random native VST / A* / GPMP routes | {len(names)} building regions\n"
        "Offline mesh support and upper-body checks | GPU RGB / depth replay on port 8769",y=.965,fontsize=16)
    ax.legend(loc="upper left",bbox_to_anchor=(.02,.94),framealpha=.9,fontsize=10)
    fig.text(.08,.035,"Displayed paths are planned references. No student policy or robot dynamics rollout.",fontsize=10,color="#33465b")
    fig.subplots_adjust(left=0,right=1,bottom=.08,top=.88)
    fig.savefig(scene/"routes_overview.png",dpi=150)
    print(scene/"routes_overview.png")


if __name__=="__main__": main()
