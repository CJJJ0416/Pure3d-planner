"""在同一批查询位置上绘制原生 Tomogram GT 与训练后的代价预测。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.pointcloud import read_json


def main():
    """按俯视图对比 GT、预测和绝对误差；没有预测时展示各条件 GT。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="准备好的场景目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, help="evaluate --predictions 输出的同点 GT/预测 NPZ")
    parser.add_argument("--profile", type=int, default=0, help="要对比的机器人条件索引")
    parser.add_argument("--max-points", type=int, default=150000, help="仅限制绘图采样点数")
    args = parser.parse_args()
    if args.prediction:
        # 评估快照中的坐标、标签、预测共享同一行序，不混用部署 PCD 的点索引。
        with np.load(args.prediction) as archive:
            scene_rows = archive["scene"] == args.scene.name
            coord = archive["coord"][scene_rows]
            cost = archive["target"][scene_rows]
            prediction = archive["prediction"][scene_rows, args.profile]
            profile_name = str(archive["profiles"][args.profile])
        metadata = dict(id=args.scene.name, split="evaluation")
    else:
        # 直接读取原生 Tomogram 的共享数组，绘制全部教师条件。
        coord = np.load(args.scene / "blocks/query_coord.npy", mmap_mode="r")
        cost = np.load(args.scene / "blocks/cost.npy", mmap_mode="r")
        metadata = read_json(args.scene / "blocks.report.json")
    rng = np.random.default_rng(0)
    # 随机抽样仅用于绘图；按 Z 排序让高层点覆盖在低层点之上
    selected = rng.choice(len(coord), min(len(coord), args.max_points), replace=False)
    selected = selected[np.argsort(coord[selected, 2])]
    xyz = coord[selected]
    if args.prediction:
        prediction = prediction[selected]
        truth = cost[selected, args.profile]
        # 三联图：GT、预测、绝对误差使用同一空间范围
        panels = [
            ("VST ground truth", truth, "RdYlGn_r", 0, 1),
            ("PTv3 prediction | " + profile_name, prediction, "RdYlGn_r", 0, 1),
            ("Absolute error", np.abs(prediction - truth), "magma", 0, 1),
        ]
        rows, columns = 1, 3
    else:
        # 无预测时：第一幅为查询高度，其余展示全部条件的 GT 代价
        panels = [("Tomogram query height (metres)", xyz[:, 2], "Blues", float(coord[:, 2].min()), float(coord[:, 2].max()))]
        for i, profile in enumerate(metadata["profiles"]):
            condition = ", ".join(f"{value:.2f}" for value in profile["condition"])
            panels.append((f"{profile['id']}\n[{condition}]", cost[selected, i], "RdYlGn_r", 0, 1))
        columns = 3
        rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(17, 5.5 * rows), constrained_layout=True, squeeze=False)
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    for axis, (title, values, palette, low, high) in zip(axes.flat, panels):
        cmap = plt.get_cmap(palette).copy()
        # 灰色表示无效/缺失值
        cmap.set_bad("#cbd0d8")
        artist = axis.scatter(
            xyz[:, 0], xyz[:, 1], c=values, cmap=cmap, vmin=low, vmax=high,
            s=0.7, linewidths=0, rasterized=True, plotnonfinite=True,
        )
        axis.set(title=title, xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        fig.colorbar(artist, ax=axis, fraction=0.04, pad=0.02)
    fig.suptitle(
        f"{metadata['id']} | {metadata['split']} | {len(coord):,} native query points\n"
        f"Plot sample: {len(selected):,} points. Native cost in [0,1].",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
