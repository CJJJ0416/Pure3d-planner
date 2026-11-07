#!/usr/bin/env python3
"""从已保存的 GPMP 路线计算课程参考，无需 GPU 或 ROS。"""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from stair_nav.training.curriculum import curriculum_references, dynamics_scales
from stair_nav.core.io import read_json, sha256, write_json


def main():
    """只读多个已有场景，为成功的原生 GPMP 路线导出离线课程参考报告。

    对每条 Nx5 CSV 使用 XYZ 计算弧长子目标和窗口，同时记录来源指纹并在计算后
    复核输入未变。输出必须是新文件，不启动 GPU、ROS、仿真器或策略训练。
    报告按点云指纹统计独立几何组，同一建筑的重叠窗口不能拆到不同训练/验证集。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/navigation_training.proposed.json")
    parser.add_argument("--output", type=Path, required=True, help="New report file; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report file")
    config = read_json(args.config)
    result = {"artifact_type": "offline_reference_curriculum", "config_status": config["status"],
              "scope": "Arc-length guidance and reference windows; no collision/support judgment, simulator rollout or policy training.",
              "reference_goals_require_runtime_ground_initialization": True,
              "reference_contains_attitude_targets": False,
              "command_interface": {"version": config.get("command_interface_version", "unspecified"),
                                    "action_order": config["network"]["action_order"]},
              "config_sha256": sha256(args.config), "dynamics_scales": dynamics_scales(config["rom"]), "scenes": []}
    for directory in args.scenes:
        directory = directory.resolve()
        scene_file = directory / "scene.json"
        report_file = directory / "paths.report.json"
        scene_digest, report_digest = sha256(scene_file), sha256(report_file)
        scene = read_json(scene_file)
        report = read_json(report_file)
        record = {"directory": str(directory), "seed": scene["seed"], "layout": scene.get("layout", "two_floor"),
                  "geometry_group": scene["pcd_sha256"], "queries": []}
        for query in report["queries"]:
            if not query.get("optimized", query.get("optimizer_succeeded", False)):
                continue
            file = directory / "paths" / f"{query['id']}.gpmp.csv"
            digest = sha256(file)
            csv = np.atleast_2d(np.loadtxt(file, delimiter=",", skiprows=1))
            if csv.shape[1] != 5:
                raise ValueError(f"Expected x,y,z,yaw,s CSV: {file}")
            planned = curriculum_references(csv[:, :3], scene, config["curriculum"])
            assert sha256(file) == digest
            record["queries"].append({"id": query["id"], "path_sha256": digest, **planned})
            print(f"{directory.name}/{query['id']}: {planned['features']['length_m']:.3f} m; "
                  f"subgoals={[v['subgoal_count'] for v in planned['guidance']]}; windows={planned['window_counts']}")
        if sha256(scene_file) != scene_digest or sha256(report_file) != report_digest:
            raise RuntimeError("Scene or report changed during curriculum calculation")
        result["scenes"].append(record)
    groups = [record["geometry_group"] for record in result["scenes"]]
    result["unique_geometry_groups"] = len(set(groups))
    result["split_rule"] = "Keep complete buildings and duplicate PCD fingerprints in the same split; windows are not independent scenes."
    write_json(args.output, result)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
