"""Partition full component point clouds into indexed core/context windows."""
import argparse
from pathlib import Path
import time

import numpy as np

from common.pointcloud_io import read_pcd_xyz, write_pcd
from utils.pointcloud import read_json, write_json, workspace_path, spatial_blocks


def export_boxes(folder, variant, rough, clean, clean_attributes, config, seed):
    """Keep every context point and its original row mapping in each box.

    Core cells partition the rough point set. Context crops overlap. Mesh
    references remain scene-wide so later D2 surface queries do not acquire
    artificial boundaries at crop edges.
    """
    scene = read_json(folder / "scene.json")
    size = np.asarray(config["core_size_m"])
    margin = config["context_margin_m"]
    shift = (np.asarray(config['grid_shift_m']) if 'grid_shift_m' in config else
             np.random.default_rng(seed).uniform(-size / 2, size / 2))
    coord = rough["coord"]
    groups, core_masks, records = [], [], []
    box_root = folder / "boxes" / f"variant_{variant:03d}"
    box_root.mkdir(parents=True, exist_ok=True)
    for index, (point_index, core_mask, lower) in enumerate(spatial_blocks(coord - shift, size, margin)):
        lower = lower + shift
        upper = lower + size
        context_lower, context_upper = lower - margin, upper + margin
        clean_index = np.flatnonzero(((clean >= context_lower) & (clean < context_upper)).all(1))
        clean_core = ((clean[clean_index] >= lower) & (clean[clean_index] < upper)).all(1)
        box_folder = box_root / f"box_{index:05d}"
        box_folder.mkdir(exist_ok=True)
        if config["export_individual_pcd_boxes"]:
            write_pcd(box_folder / "rough.pcd", coord[point_index])
            write_pcd(box_folder / "clean.pcd", clean[clean_index])
            np.savez(box_folder / "rough_attributes.npz",
                     **{name: value[point_index] for name, value in rough.items()},
                     point_index=point_index, core_mask=core_mask)
            np.savez(box_folder / "targets.npz", keep_target=rough["keep_target"][point_index],
                     surface_distance_m=rough["surface_distance_m"][point_index],
                     point_index=point_index, core_mask=core_mask)
            np.savez(box_folder / "clean_attributes.npz",
                     **{name: value[clean_index] for name, value in clean_attributes.items()},
                     point_index=clean_index, core_mask=clean_core)
        record = dict(id=index, box_path=str(box_folder.relative_to(folder)),
            core_lower=lower.tolist(), core_upper=upper.tolist(),
            context_lower=context_lower.tolist(), context_upper=context_upper.tolist(),
            context_points=len(point_index), core_points=int(core_mask.sum()),
            clean_context_points=len(clean_index), clean_core_points=int(clean_core.sum()),
            dynamic_points=int((rough["source_role"][point_index] == 7).sum()),
            registration_points=int(np.isin(rough["source_role"][point_index], [2, 6]).sum()),
            group_counts=np.bincount(rough["loss_group"][point_index[core_mask]], minlength=5).tolist())
        write_json(box_folder / "box.json", dict(**record,
            scene_id=folder.name, geometry_group_id=scene["geometry_group_id"], split=scene["split"],
            geometry_seed=scene["seed"], variant=variant, augmentation_seed=seed,
            coordinate_frame="map", units="m", source_rough="../../../" + f"rough_{variant:03d}.pcd",
            source_clean="../../../clean.pcd", target_mesh="../../../clean.obj",
            input="rough.pcd" if config['export_individual_pcd_boxes'] else f"../../../arrays_{variant:03d}/coord.npy",
            clean_reference="clean.pcd" if config['export_individual_pcd_boxes'] else "../../../clean.pcd",
            loss_points="rough core_mask only",
            targets="targets.npz" if config['export_individual_pcd_boxes'] else f"../../../arrays_{variant:03d}/keep_target.npy",
            window_index=f"../../../windows_{variant:03d}_index.npy",
            window_core=f"../../../windows_{variant:03d}_core.npy",
            window_offsets=f"../../../windows_{variant:03d}_offsets.npy",
            model_input_fields=["x", "y", "z"],
            continuous_target_status="D2 complete"))
        groups.append(point_index)
        core_masks.append(core_mask)
        records.append(record)

    counts = np.array([len(group) for group in groups], dtype=np.int64)
    np.savez(folder / f"windows_{variant:03d}.npz", point_index=np.concatenate(groups),
             core_mask=np.concatenate(core_masks), offsets=np.r_[0, np.cumsum(counts)])
    np.save(folder / f"windows_{variant:03d}_index.npy", np.concatenate(groups))
    np.save(folder / f"windows_{variant:03d}_core.npy", np.concatenate(core_masks))
    np.save(folder / f"windows_{variant:03d}_offsets.npy", np.r_[0, np.cumsum(counts)])
    selected = max(range(len(groups)), key=lambda i: int(((rough["semantic_id"][groups[i]] == 3)
        & (rough["source_role"][groups[i]] <= 1) & core_masks[i]).sum()))
    write_pcd(folder / f"window_{variant:03d}_context.pcd", coord[groups[selected]])
    write_pcd(folder / f"window_{variant:03d}_core.pcd", coord[groups[selected][core_masks[selected]]])
    description = dict(seed=seed, grid_shift_m=shift.tolist(), core_size_m=size.tolist(),
        context_margin_m=margin, unique_core_ownership=True,
        independent_pcd_boxes=bool(config['export_individual_pcd_boxes']),
        npz=f"windows_{variant:03d}.npz", selected_window=selected, windows=records)
    write_json(folder / f"windows_{variant:03d}.json", description)
    write_json(box_root / "index.json", description)
    return description


def collect_boxes(folder):
    """One flat sample list for future feature preparation and training."""
    scene = read_json(folder / "scene.json")
    samples = []
    for path in sorted((folder / "boxes").glob("variant_*/index.json")):
        variant = int(path.parent.name.rsplit("_", 1)[1])
        for box in read_json(path)["windows"]:
            samples.append(dict(id=f"{folder.name}_v{variant:03d}_b{box['id']:05d}",
                path=box["box_path"], variant=variant,
                rough_points=box["context_points"], core_points=box["core_points"],
                clean_points=box["clean_context_points"], split=scene["split"],
                dynamic_points=box["dynamic_points"], registration_points=box["registration_points"],
                group_counts=box["group_counts"],
                geometry_group_id=scene["geometry_group_id"]))
    result = dict(scene_id=folder.name, geometry_group_id=scene["geometry_group_id"],
        coordinate_frame="map", split=scene["split"], samples=samples, count=len(samples))
    write_json(folder / "boxes.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("Filter/configs/scoring_mesh_full.json"))
    parser.add_argument("--variants", type=int, nargs="+")
    args = parser.parse_args()
    folder = workspace_path(args.scene)
    config = read_json(workspace_path(args.config))
    scene = read_json(folder / "scene.json")
    manifest = read_json(folder.parent / "manifest.json")
    entry = next(row for row in manifest["scenes"] if row["scene_id"] == folder.name)
    clean = read_pcd_xyz(folder / "clean.pcd")
    clean_attributes = dict(np.load(folder / "clean_attributes.npz"))
    variants = [row["id"] for row in entry["variants"]] if args.variants is None else args.variants
    started = time.perf_counter()
    for variant in variants:
        rough = dict(np.load(folder / f"rough_attributes_{variant:03d}.npz"))
        windows = export_boxes(folder, variant, rough, clean, clean_attributes,
                               config["window"], scene["seed"] * 100 + variant)
        print(f"variant={variant:03d} | boxes={len(windows['windows'])}", flush=True)
    index = collect_boxes(folder)
    entry["boxes_path"] = f"{folder.name}/boxes.json"
    entry["box_count"] = index["count"]
    entry["training_unit"] = "individual_rough_pcd_box_with_context"
    write_json(folder.parent / "manifest.json", manifest)
    write_json(folder / "box_export.json", dict(boxes=index["count"], variants=variants,
        core_size_m=config["window"]["core_size_m"], context_margin_m=config["window"]["context_margin_m"],
        elapsed_s=time.perf_counter() - started))
    print(f"Exported {index['count']} individual PCD boxes | {folder / 'boxes.json'}", flush=True)


if __name__ == "__main__":
    main()
