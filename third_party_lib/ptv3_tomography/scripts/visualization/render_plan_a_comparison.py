#!/usr/bin/env python3
"""Render comparison textures for the native Isaac viewer from saved Plan A predictions."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ptv3_tomography.io import atomic_json, file_sha256, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    manifest = read_json(args.maps / "manifest.json")
    cfg = manifest["map_config"]
    directory = args.maps / "plots"
    directory.mkdir(exist_ok=True)
    cost_cmap = LinearSegmentedColormap.from_list("planner_cost", ["#16a34a", "#fde047", "#ef4444"])
    cost_cmap.set_bad("#e5e7eb")
    height_cmap = plt.get_cmap("viridis").copy()
    height_cmap.set_bad("#e5e7eb")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#e5e7eb")
    nx, ny = cfg["grid_size_xy"]
    resolution = cfg["resolution_m"]
    extent = [(-nx // 2 - .5) * resolution, (nx - nx // 2 - .5) * resolution,
              (-ny // 2 - .5) * resolution, (ny - ny // 2 - .5) * resolution]
    result = {"frames": [], "cost_scale": [0, 60], "unknown_color": "grey",
              "comparison_mask": "independent mask_cost / mask_height; diagnostics only"}
    indices = args.frames if args.frames is not None else [row["frame_index"] for row in manifest["frames"]]
    for index in indices:
        row = next(row for row in manifest["frames"] if row["frame_index"] == index)
        file = args.maps / row["file"]
        with np.load(file, allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        best_band = int(np.argmax(arrays["mask_cost"].sum(axis=(1, 2))))
        output = {"frame_index": index, "best_band": best_band, "bands": []}
        for band in range(len(cfg["height_bin_edges_m"]) - 1):
            cost_mask, height_mask = arrays["mask_cost"][band], arrays["mask_height"][band]
            gt_cost, pred_cost = arrays["gt_cost_raw"][band], arrays["cost_raw"][band]
            gt_height, pred_height = arrays["gt_ground_height_gravity_m"][band], arrays["ground_height_gravity_m"][band]
            lo, hi = cfg["height_bin_edges_m"][band:band + 2]
            panels = [(gt_cost, cost_mask, "Tomography cost", cost_cmap, 0, 60, "cost"),
                      (pred_cost, cost_mask, "PTv3 predicted cost", cost_cmap, 0, 60, "cost"),
                      (np.abs(pred_cost - gt_cost), cost_mask, "Absolute cost error", error_cmap, 0, 60, "cost"),
                      (gt_height, height_mask, "Tomography ground height", height_cmap, lo, hi, "height (m)"),
                      (pred_height, height_mask, "PTv3 predicted ground height", height_cmap, lo, hi, "height (m)"),
                      (np.abs(pred_height - gt_height), height_mask, "Absolute height error", error_cmap, 0, .25, "error (m)")]
            with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                                 "figure.facecolor": "#f8fafc", "axes.spines.top": False, "axes.spines.right": False}):
                fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
                for ax, (values, mask, title, cmap, minimum, maximum, label) in zip(axes.flat, panels):
                    display = np.ma.masked_where(~mask, values)
                    im = ax.imshow(display.T, origin="lower", extent=extent, cmap=cmap,
                                   vmin=minimum, vmax=maximum, interpolation="nearest")
                    ax.set(title=title, xlabel="LiDAR X (m)", ylabel="LiDAR Y (m)")
                    ax.set_facecolor("#e5e7eb")
                    ax.set_xticks([-3, -1.5, 0, 1.5, 3])
                    ax.set_yticks([-3, -1.5, 0, 1.5, 3])
                    fig.colorbar(im, ax=ax, fraction=.044, pad=.03, label=label)
                    if not mask.any():
                        ax.text(.5, .5, "No supervised cells", transform=ax.transAxes, ha="center", va="center", color="#475569")
                cost_mae = float(np.abs(pred_cost[cost_mask] - gt_cost[cost_mask]).mean()) if cost_mask.any() else None
                height_mae = float(np.abs(pred_height[height_mask] - gt_height[height_mask]).mean()) if height_mask.any() else None
                text = (f"Plan A | trained full PTv3 | frame {index} | gravity height [{lo:g}, {hi:g}) m\n"
                        f"Native VST targets. Grey = no teacher label. Cost cells {cost_mask.sum()}, height cells {height_mask.sum()}.")
                fig.suptitle(text, fontsize=15, fontweight="bold", color="#0f172a")
                path = directory / f"frame_{index:05d}_band_{band}.png"
                fig.savefig(path, dpi=130)
                plt.close(fig)
            output["bands"].append({"band": band, "file": str(path.relative_to(args.maps)),
                                    "sha256": file_sha256(path), "cost_mae": cost_mae, "height_mae_m": height_mae})
        result["frames"].append(output)
        print(f"PLAN_A_COMPARISON frame={index} best_band={best_band}", flush=True)
    atomic_json(args.maps / "plot_index.json", result)


if __name__ == "__main__":
    main()
