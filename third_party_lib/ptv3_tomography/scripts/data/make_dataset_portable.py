#!/usr/bin/env python3
"""Copy v2 raw scans, labels and scene into one directory with relative runtime paths."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset
from ptv3_tomography.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = LidarStepDataset(args.dataset)
    output = args.output.resolve()
    output.mkdir(parents=True)
    status_file = output / "TRANSFER_STATUS.json"
    atomic_json(status_file, dict(status="copy_in_progress",frames=len(source)))
    runtime_hashes = {}

    def copy_asset(src, relative, expected):
        src, dst = Path(src), output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return expected

    manifest = copy.deepcopy(source.manifest)
    for i, frame in enumerate(manifest["frames"]):
        raw = f"raw/scan_{i:05d}.npz"
        label = f"labels/frame_{i:05d}.npz"
        copy_asset(source.asset(frame["scan_file"]), raw, frame["scan_sha256"])
        copy_asset(source.asset(frame["label_file"]), label, frame["label_sha256"])
        frame["scan_file"], frame["label_file"] = raw, label
    odom = manifest["lidar_odometry"]
    copy_asset(source.asset(odom["file"]), "lidar_odometry.npz", odom["sha256"])
    odom["file"] = "lidar_odometry.npz"
    source_hashes = {source.asset(path): digest for path, digest in source.spec["source_fingerprints"].items()}
    for name in ("scene.json", "terrain.obj", "terrain.pcd", "vst.pickle", "vst.meta.json"):
        src = source.asset(source.spec["scene_directory"]) / name
        relative = f"scene/{name}"
        runtime_hashes[relative] = copy_asset(src, relative, source_hashes[src])
    for name in ("profile.json", "report.json"):
        src = source.asset(source.spec["capture_directory"]) / name
        if src in source_hashes:
            relative = f"raw/{name}"
            runtime_hashes[relative] = copy_asset(src, relative, source_hashes[src])
    raw_report = source.asset(source.spec["capture_directory"]) / "report.json"
    if raw_report.is_file():
        for asset in json.loads(raw_report.read_text()).get("source_assets", []):
            if asset["base"] != "capture":
                continue
            relative_source = Path(asset["path"])
            src = source.asset(source.spec["capture_directory"]) / relative_source
            relative = "raw/" + relative_source.as_posix()
            runtime_hashes[relative] = copy_asset(src, relative, asset["sha256"])
    spec = copy.deepcopy(source.spec)
    spec["source_provenance"] = dict(original_source_fingerprints=spec["source_fingerprints"],
                                     original_scene_directory=spec["scene_directory"],
                                     original_capture_directory=spec["capture_directory"],
                                     note="Historical provenance only; original paths are not read at runtime")
    spec.update(scene_directory="scene", capture_directory="raw", source_fingerprints=runtime_hashes,
                portable_assets=True, portability_version="relative_runtime_paths_v1")
    atomic_json(output / "dataset_spec.json", spec)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(status_file, dict(status="complete", frames=len(manifest["frames"]), runtime_paths_relative=True,
                                 raw_and_label_bytes_unchanged=True, training_started=False))
    print(f'Portable dataset ready: {output} ({len(manifest["frames"])} frames, no training executed)')


if __name__ == "__main__":
    main()
