"""Uniform map-wide endpoints and native VST A*/GPMP routes per building region."""
import os
from pathlib import Path
import pickle

import numpy as np

from .terrain import load_backend
from .supervision import write_json_utf8


LABELS={"curved_bridge":"曲桥","ramp":"独立斜坡","open_stairs":"空心直梯",
    "switchback_stairs":"折返楼梯","spiral_stairs":"旋转楼梯","raised_platform":"独立大平台",
    "underpass":"桥洞","obstacles":"障碍区","low_walls":"低墙通道"}


class RegionRoutes:
    def __init__(self,source,backend_root):
        self.source=Path(source).resolve()
        load_backend(backend_root)
        from stair_nav.core.io import read_json,sha256
        from stair_nav.planning.vst import check_sources
        self.scene=read_json(self.source/"scene.json")
        self.meta=read_json(self.source/"vst.meta.json")
        if self.meta["tomogram_sha256"]!=sha256(self.source/"vst.pickle"):
            raise ValueError("Source VST fingerprint changed")
        if self.meta["pcd_sha256"]!=sha256(self.source/"terrain.pcd"):
            raise ValueError("Source point cloud fingerprint changed")
        if self.meta["source"]["files"]!=check_sources()["files"]:
            raise ValueError("Source VST and current native planner versions differ")
        self.threshold=float(self.scene["config"]["planner"]["cost_threshold"])
        self.barrier=float(self.scene["config"]["vst"]["cost_barrier"])
        self.cache={}

    def region(self,name):
        if name not in self.cache:
            from stair_nav.planning.vst import make_planner
            with (self.source/"regions"/name/"vst.pickle").open("rb") as f:
                payload=pickle.load(f)
            data=np.asarray(payload["data"],float)
            valid=np.isfinite(data[3])&np.isfinite(data[0])&(data[0]>=0)&(data[0]<=self.threshold)
            cells=np.argwhere(valid)
            xy=(cells[:,1:]-np.array(data.shape[2:])//2)*payload["resolution"]+payload["center"]
            xyz=np.c_[xy,data[3,tuple(cells.T)[0],tuple(cells.T)[1],tuple(cells.T)[2]]]
            # Duplicate representations of the same support do not bias random sampling.
            _,ids=np.unique(np.round(xyz,4),axis=0,return_index=True)
            xyz=xyz[ids]
            self.cache[name]=dict(payload=payload,data=data,xyz=xyz,
                planner=make_planner(data,payload["resolution"],cost_barrier=self.barrier,cost_threshold=self.threshold))
        return self.cache[name]

    def plan(self,name,rng,reverse=False):
        from stair_nav.planning.vst import endpoint_index,optimized_world
        r=self.region(name)
        # Sample independently across the region's full traversable map, at all ground heights.
        start,goal=r["xyz"][rng.choice(len(r["xyz"]),size=2,replace=False)]
        if reverse: start,goal=goal,start
        if np.linalg.norm(start[:2]-goal[:2])<1.5:
            raise ValueError("Random endpoints are too close")
        payload,data,planner=r["payload"],r["data"],r["planner"]
        si=endpoint_index(start,data,payload["center"],payload["resolution"],payload["slice_dh"],self.threshold)
        gi=endpoint_index(goal,data,payload["center"],payload["resolution"],payload["slice_dh"],self.threshold)
        if not planner.plan(si,gi,False):
            raise RuntimeError("A* initialization failed")
        indices=np.asarray(planner.get_path_finder().get_result_matrix(),int)
        if not len(indices) or not np.array_equal(indices[-1,1:],gi[[2,1]]):
            raise RuntimeError("A* did not reach the sampled goal")
        if not planner.plan(si,gi,True):
            raise RuntimeError("Native GPMP optimization failed")
        path=optimized_world(planner,data,payload["center"],payload["resolution"])
        if path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.isfinite(path).all():
            raise ValueError("Invalid GPMP path")
        if (max(np.linalg.norm(path[0,:2]-start[:2]),np.linalg.norm(path[-1,:2]-goal[:2]))>.3
                or max(abs(path[0,2]-start[2]),abs(path[-1,2]-goal[2]))>.5):
            raise ValueError("GPMP did not connect the sampled endpoints at their height")
        bounds=np.array(self.scene["training_regions"][name]["planning_bounds_xy"])
        if np.any(path[:,:2]<bounds[:2]-1e-7) or np.any(path[:,:2]>bounds[2:]+1e-7):
            raise ValueError("GPMP left its own building region")
        return path,dict(region_id=name,start_support_xyz=start.tolist(),goal_support_xyz=goal.tolist(),
            reverse=reverse,start_index=si.tolist(),goal_index=gi.tolist(),
            endpoint_sampling="uniform_all_traversable_region_cells",
            planning_bounds_xy=bounds.tolist(),cost_threshold=self.threshold,
            backend="VST + native A* + native GPMP",height_contract="Native GPMP height reference, default +0.1m offset, unchanged")


def generate_routes(source,output,backend_root,seed=20261104,per_region=2,max_attempts=30):
    from stair_nav.planning.vst import save_path
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/"paths").mkdir()
    planner=RegionRoutes(source,backend_root)
    from .route_geometry import RouteGeometry
    geometry=RouteGeometry(planner.source/"terrain.obj",planner.scene["config"]["validation"])
    rng=np.random.default_rng(seed)
    records=[]
    original=Path.cwd()
    os.chdir(output/"paths")
    try:
        for name in planner.scene["training_regions"]:
            for number in range(per_region):
                query=f"{name}_{number+1:02d}"
                record=dict(id=query,region_id=name,optimized=False,failures=[])
                for attempt in range(1,max_attempts+1):
                    try:
                        path,meta=planner.plan(name,rng,reverse=bool(number%2))
                        validation=geometry.check(path)
                        save_path(path,Path(query+".gpmp"))
                        record.update(meta,optimized=True,accepted=True,path_csv=f"paths/{query}.gpmp.csv",
                            geometry_validation=validation,attempts=attempt,
                            length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))
                        break
                    except (ValueError,RuntimeError,IndexError) as exc:
                        record["failures"].append(str(exc))
                records.append(record)
                write_json_utf8(output/"routes.report.json",dict(seed=seed,source=str(planner.source),queries=records))
                print(f"{query}: {'planned' if record['optimized'] else 'failed'}",flush=True)
    finally:
        os.chdir(original)
    return records

