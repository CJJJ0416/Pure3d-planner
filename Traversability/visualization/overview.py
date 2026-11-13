"""为已有场景重新导出 overview 截图，不重建几何或标签。"""
import argparse
from pathlib import Path
import sys

import trimesh

from utils.pointcloud import ROOT, read_json, workspace_path, write_json


def main():
    """按 manifest 逐个场景重绘总览、布局与连接图，并刷新汇总图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/dataset.json")
    parser.add_argument("--scene", help="只处理指定场景 ID，例如 scene_20261109；默认处理全部场景。")
    args = parser.parse_args()
    cfg = read_json(args.config)
    source = workspace_path(cfg["output"])
    manifest = read_json(source / "manifest.json")
    # 复用场景生成器中的可视化函数，避免维护两套绘图实现
    sys.path.insert(0, str(ROOT / "common/geneator"))
    from visualize_wfc import render, render_connections, render_layout
    from Traversability.data.generate import render_contact_sheet
    for row in manifest["scenes"]:
        if args.scene and row["id"] != args.scene:
            continue
        folder = source / row["folder"]
        scene = read_json(folder / "scene.json")
        # 直接加载已保存的楼层与楼梯网格，不重新生成场景
        floors = [trimesh.load_mesh(folder / f"floor_{i}.obj", process=False) for i in range(len(scene["floors"]))]
        stairs = [trimesh.load_mesh(folder / f"{s['id']}.obj", process=False) for s in scene["stairs"]]
        render(folder, scene["config"], floors, stairs)
        render_layout(folder, scene["config"])
        render_connections(folder, scene["config"], stairs, scene["stairs"])
        # 同步更新 manifest 与 scene.json 中的图片路径
        row["overview"] = f"{row['folder']}/overview.png"
        scene["overview"] = "overview.png"
        write_json(folder / "scene.json", scene)
        print(folder / "overview.png", flush=True)
    write_json(source / "manifest.json", manifest)
    if not args.scene:
        # 全量运行时额外生成所有场景的拼图
        render_contact_sheet(source, manifest["scenes"])
        print(source / "overview_all.png", flush=True)


if __name__ == "__main__":
    main()
