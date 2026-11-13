"""Generate stair families, native VST condition banks and shared point-query caches."""
import argparse
import copy
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from common.pointcloud_io import read_pcd_xyz, write_pcd
from common.geneator.stair_families import generate_family_scene, stair_membership
from Traversability.data.blocks import query_blocks
from Traversability.data.teacher_conditions import legacy_profiles, condition_banks, pairs_for
from utils.pointcloud import ROOT, CONDITION_NAMES, read_json, write_json, voxel_downsample

os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "datasets/cache/cupy"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))
sys.path.insert(0, str(ROOT / "common/vst_global_planner/src"))
from vst_global_planner.conditioned_teacher import ConditionedTeacher
from vst_global_planner.tomogram_cloud import tomogram_points

OUTPUT = ROOT / "datasets/wfc_stair_stand_v1"
SETTINGS = read_json(ROOT / "Traversability/configs/dataset_response.json")["blocks"]
OLD = ROOT / "datasets/wfc_response_paired_v4"


def label_scene(folder, teacher, profiles):
    """All label columns share native Tomogram XYZ, never nearest-input relabelling."""
    folder.mkdir(parents=True, exist_ok=True)
    costs = []
    for i, p in enumerate(profiles):
        path = folder / f"vst_{p['id']}.pickle"
        if path.exists():
            with path.open("rb") as stream:
                payload = pickle.load(stream)
        else:
            raw = teacher.cost(p["condition"])
            payload = teacher.payload(p["condition"], raw)
            with path.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        cloud = tomogram_points(payload)
        costs.append(cloud[:, 3])
        p["vst"] = str(path)
        if i == 0:
            xyz = cloud[:, :3]
            write_pcd(folder / "nominal.pcd", xyz, cloud[:, 3])
        if (i + 1) % 20 == 0:
            print(f"teacher {folder.parent.name}/{folder.name}: {i+1}/{len(profiles)}", flush=True)
    np.save(folder / "cost.npy", np.column_stack(costs))
    np.save(folder / "conditions.npy", np.asarray([p["condition"] for p in profiles], dtype=np.float32))
    write_json(folder / "profiles.json", profiles)
    return xyz


def geometry_cache(folder, xyz, scene, input_pcd=None, membership=None):
    cache = folder / "geometry"
    cache.mkdir(exist_ok=True)
    points = read_pcd_xyz(folder / "terrain_downsampled.pcd" if input_pcd is None else input_pcd)
    coarse = voxel_downsample(points, .5)
    _, gn = cKDTree(coarse).query(xyz, k=4)
    if membership is None:
        membership = stair_membership(xyz, scene)
    arrays = dict(input_coord=points, query_coord=xyz, global_coord=coarse, global_neighbors=gn,
                  stair_mask=membership)
    for name, array in arrays.items():
        np.save(cache / f"{name}.npy", array)
    rows = []
    for i, (ii, core, qi, lo, hi) in enumerate(query_blocks(points, xyz, SETTINGS["core_size_m"], SETTINGS["context_m"])):
        _, knn = cKDTree(points[ii]).query(xyz[qi], k=8)
        path = cache / f"block_{i:05d}.npz"
        np.savez(path, input_indices=ii, input_core=core, query_indices=qi, query_neighbors=knn)
        rows.append(dict(id=f"block_{i:05d}", scene=folder.name, seed=scene["seed"],
                         split=scene["split"], batch=scene["family_id"], file=str(path),
                         core_lower_m=lo.tolist(), core_upper_m=hi.tolist(),
                         input_points=len(ii), query_points=len(qi), stair_block=bool(membership[qi].any()),
                         replay=False))
    return {name:str(cache / f"{name}.npy") for name in arrays}, rows


def empty_metadata():
    return dict(version="stair_stand_v1", condition_names=CONDITION_NAMES,
                condition_scale=[1., .5, .3, 1., 1.], cost_scale=50., coordinate_frame="map",
                blocks_config=SETTINGS, scenes=[], blocks=[])


def legacy_stair_mask(xyz, metadata):
    """Only identifies stair blocks for replay sampling, never gates their targets."""
    angle = np.deg2rad(metadata["config"]["yaw_deg"])
    c, s = np.cos(angle), np.sin(angle)
    xy = xyz[:, :2] @ np.array([[c, -s], [s, c]])
    selected = np.zeros(len(xyz), bool)
    for r in metadata["stairs"]:
        local = xy - r["center_local_xy"]
        a = np.deg2rad(r.get("rotation_deg", 0))
        ca, sa = np.cos(a), np.sin(a)
        local = local @ np.array([[ca, -sa], [sa, ca]])
        w, h = r["unrotated_footprint_m"]
        z0, z1 = [metadata["config"]["floor_z_m"][r[k]] for k in ("from_floor", "to_floor")]
        selected |= ((abs(local[:,0]) < w/2) & (abs(local[:,1]) < h/2)
                     & (xyz[:,2] > z0+.03) & (xyz[:,2] < z1+.1))
    return selected


def save_indices(indices):
    for name, meta in indices.items():
        meta["splits"] = {split:dict(scenes=sum(s["split"] == split for s in meta["scenes"]),
                                    blocks=sum(b["split"] == split for b in meta["blocks"]))
                          for split in ["train", "val", "test"]}
        write_json(OUTPUT / name / "blocks.json", meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    robots = read_json(ROOT / "Traversability/configs/robots_response.json")
    indices = {name:empty_metadata() for name in ["A", "B", "C", "evaluation"]}
    old = read_json(OLD / "blocks.json")
    # The six accepted original training buildings remain unchanged.
    for previous in [s for s in old["scenes"] if s["split"] == "train"]:
        name = previous["id"]
        row = copy.deepcopy(previous)
        row["arrays"] = {k:str(OLD / v) for k,v in row["arrays"].items()}
        old_scene = read_json(OLD / name / "scene.json")
        xyz = np.load(row["arrays"]["query_coord"])
        membership = legacy_stair_mask(xyz, old_scene)
        block_rows = []
        for b in [b for b in old["blocks"] if b["scene"] == name]:
            b = dict(b, file=str(OLD / b["file"]), replay=True)
            with np.load(b["file"]) as data:
                b["stair_block"] = bool(membership[data["query_indices"]].any())
            block_rows.append(b)
        profiles = [p for p in row["profiles"] if p["condition"][3] == .5]
        for version in ["A", "B"]:
            indices[version]["scenes"].append(copy.deepcopy(row))
            indices[version]["blocks"].extend(copy.deepcopy(block_rows))
            indices[version]["condition_pairs"] = pairs_for([p["condition"] for p in profiles])
        if not args.pilot:
            teacher = ConditionedTeacher(read_pcd_xyz(OLD / previous["input_pcd"]))
            profiles, banks, pairs = condition_banks(teacher, previous["seed"])
            folder = OUTPUT / "replay" / name
            folder.mkdir(parents=True, exist_ok=True)
            label_scene(folder, teacher, profiles)
            row["arrays"].update(cost=str(folder / "cost.npy"), conditions=str(folder / "conditions.npy"))
            row.update(profiles=profiles, profile_banks=banks)
            indices["C"]["scenes"].append(row)
            indices["C"]["blocks"].extend(block_rows)
            indices["C"]["condition_pairs"] = pairs
            del teacher
            save_indices(indices)
            print(f"REPLAY READY {name}", flush=True)
    families = [("train", kind) for kind in ["straight"]*4+["switchback"]*2+["spiral"]*2]
    families += [(split, kind) for split in ["val", "test"] for kind in ["straight", "switchback", "spiral"]]
    geometry_rows = []
    for family, (split, kind) in enumerate(families):
        if args.pilot and family not in [0, 4, 6]:
            continue
        family_seed = 20262100 + family
        for variant in ([1] if args.pilot else range(3)):
            seed = family_seed * 10 + variant
            folder = OUTPUT / "scenes" / f"scene_{seed}"
            scene = (read_json(folder / "scene.json") if (folder / "scene.json").exists() else
                     generate_family_scene(folder, family_seed, variant, kind, split))
            teacher = ConditionedTeacher(read_pcd_xyz(folder / "terrain_downsampled.pcd"))
            profiles, banks, pairs = condition_banks(teacher, seed, banks=3 if split=="train" and not args.pilot else 1,
                                                    critical=split=="train", evaluation=split!="train")
            xyz = label_scene(folder / "conditioned", teacher, profiles)
            arrays, rows = geometry_cache(folder, xyz, scene)
            row = dict(id=folder.name, seed=seed, split=split, batch=scene["family_id"], family=scene["family_id"],
                       input_pcd=str(folder / "terrain_downsampled.pcd"), arrays=dict(arrays,
                       cost=str(folder / "conditioned/cost.npy"), conditions=str(folder / "conditioned/conditions.npy")),
                       query_points=len(xyz), input_points=scene["input_points"], profiles=profiles,
                       profile_banks=banks, blocks=len(rows), geometry_scene=str(folder / "scene.json"))
            version = "C" if split=="train" else "evaluation"
            indices[version]["scenes"].append(row)
            indices[version]["blocks"].extend(rows)
            indices[version]["condition_pairs"] = pairs
            if split=="train" and not args.pilot:
                legacy = legacy_profiles(robots, seed)
                label_scene(folder / "legacy", teacher, legacy)
                b = copy.deepcopy(row)
                b.pop("profile_banks")
                b.update(profiles=legacy, arrays=dict(arrays, cost=str(folder/"legacy/cost.npy"),
                                                    conditions=str(folder/"legacy/conditions.npy")))
                indices["B"]["scenes"].append(b)
                indices["B"]["blocks"].extend(rows)
            geometry_rows.append(dict(scene=folder.name, family=scene["family_id"], split=split, kind=kind,
                                      seed=seed, overview=str(folder/"overview.png"), stairs=scene["stairs"]))
            save_indices(indices)
            write_json(OUTPUT / "manifest.json", dict(scenes=geometry_rows, elapsed_s=time.perf_counter()-started))
            print(f"SCENE READY {folder.name} {split} {kind} blocks={len(rows)}", flush=True)
            del teacher
    write_json(OUTPUT / ("pilot_complete.json" if args.pilot else "data_ready.json"),
               dict(complete=True, seconds=time.perf_counter()-started, scenes=len(geometry_rows)))


if __name__ == "__main__":
    main()
