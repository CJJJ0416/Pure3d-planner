"""WFC static reference -> fused XYZ variants -> continuous targets -> boxes."""
import argparse
from pathlib import Path
import time
import numpy as np
import trimesh

from common.geneator.filter_components import SEMANTICS
from common.pointcloud_io import write_pcd, read_pcd_xyz
from Filter.data.augment import augment_cloud, SOURCE_ROLES
from Filter.data.boxes import export_boxes, collect_boxes
from Filter.data.targets import StaticSurfaceTargets
from Filter.data.groups import loss_groups
from utils.mesh_sampling import sample_surface
from utils.pointcloud import read_json, write_json, workspace_path


def prepare_variants(folder, seed, variants, clean, clean_attributes, config, static_mesh, suite_seed=0):
    scene = read_json(folder / "scene.json")
    teacher = StaticSurfaceTargets(static_mesh, config["target"])
    variant_records = []
    for index in range(variants):
        started = time.perf_counter()
        augmentation_seed = seed * 100 + suite_seed * 10000 + index
        rough, record, sources = augment_cloud(clean, clean_attributes, config["augmentations"],
            augmentation_seed, config["variants"][index], scene)
        augment_s = time.perf_counter() - started
        targets, target_report = teacher.score(rough)
        rough.update(targets)
        rough['loss_group'] = loss_groups(rough)
        arrays = folder / f'arrays_{index:03d}'
        arrays.mkdir(exist_ok=True)
        for name in ['coord', 'keep_target', 'loss_group', 'semantic_id']:
            np.save(arrays / f'{name}.npy', rough[name])
        tick = time.perf_counter()
        write_pcd(folder / f"rough_{index:03d}.pcd", rough["coord"])
        np.savez(folder / f"rough_attributes_{index:03d}.npz", **rough)
        np.savez(folder / f"targets_{index:03d}.npz", **targets)
        np.savez(folder / f"augmentation_sources_{index:03d}.npz", **sources)
        write_pcd(folder / f"gt_{index:03d}.pcd", rough["coord"], rough["keep_target"], field_name="keep_target")
        write_pcd(folder / f"filtered_gt_{index:03d}.pcd", rough["coord"][rough["keep_target"] >= .5])
        for name, selected in {
            "dynamic_trails": rough["source_role"] == SOURCE_ROLES["dynamic_object_trail"],
            "registration_ghosts": np.isin(rough["source_role"], [SOURCE_ROLES["registration_fragment"], SOURCE_ROLES["registration_trail"]]),
            "static_controls": (rough["source_role"] == SOURCE_ROLES["static_object"]) | rough["is_disconnected_control"],
        }.items():
            write_pcd(folder / f"{name}_{index:03d}.pcd", rough["coord"][selected],
                      rough["keep_target"][selected], field_name="keep_target")
        record.update(labels_status="D2 complete", target_report=target_report)
        save_s = time.perf_counter() - tick
        tick = time.perf_counter()
        windows = export_boxes(folder, index, rough, clean, clean_attributes, config["window"], augmentation_seed)
        box_s = time.perf_counter() - tick
        record.update(window_index=windows["npz"], timings=dict(augmentation_s=augment_s,
            targets_s=target_report["distance_query_and_score_s"], save_s=save_s, boxes_s=box_s))
        write_json(folder / f"augmentation_{index:03d}.json", record)
        write_json(folder / f"targets_{index:03d}.json", target_report)
        variant_records.append(dict(id=index, name=config["variants"][index]["name"],
            seed=augmentation_seed, points=record["points"], rough_path=f"rough_{index:03d}.pcd",
            attributes_path=f"rough_attributes_{index:03d}.npz", targets_path=f"targets_{index:03d}.npz",
            augmentation_path=f"augmentation_{index:03d}.json", windows_path=f"windows_{index:03d}.json",
            boxes=len(windows["windows"]), timings=record["timings"], elapsed_s=time.perf_counter() - started))
        print(f"D1/D2 | {index:03d} {record['variant']['name']} | points={len(rough['coord']):,} | boxes={len(windows['windows'])} | {record['source_role_counts']}", flush=True)
    collect_boxes(folder)
    return variant_records, teacher.build_s


def update_manifest(folder, scene, config, clean_points, variants):
    root = folder.parent
    entry = dict(scene_id=folder.name, geometry_group_id=scene["geometry_group_id"],
        geometry_seed=scene["seed"], split=scene["split"], family=scene["family"],
        mesh_path=f"{folder.name}/scene.obj", target_mesh_path=f"{folder.name}/clean.obj",
        face_metadata_path=f"{folder.name}/face_metadata.npz", roles_path=f"{folder.name}/roles.json",
        overview_path=f"{folder.name}/overview.png", clean_path=f"{folder.name}/clean.pcd",
        clean_points=clean_points, variants=variants, boxes_path=f"{folder.name}/boxes.json",
        box_count=read_json(folder / "boxes.json")["count"], training_unit="individual_rough_xyz_box_with_context",
        model_input_fields=["x", "y", "z"], stage_status={"D0": "complete", "D1": "complete", "D2": "complete"})
    path = root / "manifest.json"
    manifest = read_json(path) if path.exists() else dict(version=config["version"],
        coordinate_frame="map", units="m", z_axis="up", scenes=[])
    manifest["scenes"] = sorted([s for s in manifest["scenes"] if s["scene_id"] != folder.name] + [entry],
                                key=lambda s: s["geometry_seed"])
    write_json(path, manifest)
    write_json(root / "splits.json", {name: [s["scene_id"] for s in manifest["scenes"] if s["split"] == name]
                                    for name in ("train", "val", "test")})


def prepare(config, seed, split, variants):
    started = time.perf_counter()
    folder = workspace_path(config["dataset_root"]) / f"scene_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    from common.geneator.filter_wfc import build_wfc_terrain
    mesh, building_mesh, static_objects_mesh, face_meta, roles, scene = build_wfc_terrain(seed, config["generation"], folder)
    for name, value in [("scene.obj", mesh), ("clean.obj", mesh),
                        ("terrain.obj", building_mesh), ("static_objects.obj", static_objects_mesh)]:
        value.export(folder / name, include_color=False, digits=9)
    np.savez(folder / "face_metadata.npz", **face_meta)
    np.savez(folder / "clean_face_metadata.npz", **face_meta)
    write_json(folder / "roles.json", dict(semantic_classes=SEMANTICS,
        target_roles={"dynamic_source": 0, "static_reference": 1}, mesh="scene.obj", target_mesh="clean.obj",
        identity_method="Source component IDs retained through surface union", instances=roles))
    scene.update(split=split, geometry_group_id=f"{scene['family']}_{seed}", status="D0_D1_D2_complete")
    write_json(folder / "scene.json", scene)
    d0_s = time.perf_counter() - started
    print(f"D0 | seed={seed} | building faces={scene['building_faces']} | static object faces={scene['static_object_faces']}", flush=True)
    tick = time.perf_counter()
    clean, clean_face, sampling_report = sample_surface(mesh, config["sampling"]["candidate_spacing_m"], return_face_ids=True)
    attrs = {key: value[clean_face] for key, value in face_meta.items()}
    attrs["surface_normal"] = mesh.face_normals[clean_face].astype(np.float32)
    write_pcd(folder / "terrain.pcd", clean)
    np.savez(folder / "terrain_attributes.npz", **attrs)
    surface_points = len(clean)
    grid = np.floor(clean.astype(np.float64) / config["sampling"]["voxel_m"]).astype(np.int64)
    _, representatives = np.unique(grid, axis=0, return_index=True)
    representatives.sort()
    clean = clean[representatives]
    attrs = {key: value[representatives] for key, value in attrs.items()}
    attrs["surface_sample_id"] = representatives
    write_pcd(folder / "terrain_downsampled.pcd", clean)
    write_pcd(folder / "clean.pcd", clean)
    np.savez(folder / "clean_attributes.npz", **attrs)
    sampling_s = time.perf_counter() - tick
    print(f"D1 | surface={surface_points:,} | clean={len(clean):,}", flush=True)
    records, build_s = prepare_variants(folder, seed, variants, clean, attrs, config, mesh)
    update_manifest(folder, scene, config, len(clean), records)
    write_json(folder / "config.resolved.json", config)
    write_json(folder / "preparation.json", dict(status="D0_D1_D2_complete", d0_s=d0_s,
        sampling_s=sampling_s, triangle_index_build_s=build_s, surface_points=surface_points,
        clean_points=len(clean), sampling=sampling_report, variants=records,
        data_preparation_s=time.perf_counter() - started))
    return folder


def prepare_and_render(config, seed, split, variants):
    started = time.perf_counter()
    folder = prepare(config, seed, split, variants)
    from Filter.visualization.visualize import render_scene
    tick = time.perf_counter()
    render_scene(folder)
    report = read_json(folder / "preparation.json")
    report.update(visualization_s=time.perf_counter() - tick, wall_s=time.perf_counter() - started)
    write_json(folder / "preparation.json", report)
    return folder


def augment_existing(config, folder, variants):
    """Regenerate D1/D2 of a v2 scene, reusing its static reference."""
    from Filter.visualization.visualize import render_scene
    started = time.perf_counter()
    scene = read_json(folder / "scene.json")
    clean = read_pcd_xyz(folder / "clean.pcd")
    attrs = dict(np.load(folder / "clean_attributes.npz"))
    mesh = trimesh.load(folder / "clean.obj", force="mesh", process=False)
    records, build_s = prepare_variants(folder, scene["seed"], variants, clean, attrs, config, mesh)
    update_manifest(folder, scene, config, len(clean), records)
    write_json(folder / "config.resolved.json", config)
    render_scene(folder)
    report = read_json(folder / "preparation.json")
    report.update(variants=records, last_d1_d2_regeneration_s=time.perf_counter() - started,
                  triangle_index_build_s=build_s)
    write_json(folder / "preparation.json", report)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("Filter/configs/dataset.json"))
    parser.add_argument("--seed", type=int, default=821000)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--variants", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--augment-only", type=Path, help="Reuse an existing filter_trails_v2 scene.")
    args = parser.parse_args()
    config = read_json(workspace_path(args.config))
    if args.output is not None:
        config["dataset_root"] = str(args.output)
    variants = config["splits"]["variants_per_geometry"] if args.variants is None else args.variants
    folder = (augment_existing(config, workspace_path(args.augment_only), variants) if args.augment_only
              else prepare_and_render(config, args.seed, args.split, variants))
    print(f"Done | {folder}", flush=True)


if __name__ == "__main__":
    main()
