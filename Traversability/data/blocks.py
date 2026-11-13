"""建筑输入点云与原生 Tomogram 查询点云相互独立的空间分块工具。

设计要点：
- 建筑几何（输入）与 Tomogram 查询点（监督位置）来自不同点集，行索引互不复用；
- 每个查询点只被分配到一个核心块，核心块尺寸默认 6×6×4 m；
- 每个核心块额外携带 1.5 m 上下文几何，保证网络能看到块边界附近的墙体/台阶；
- 另存一份 0.5 m 体素降采样的全楼粗几何，用于捕获远距离结构；
- 块索引文件只保存索引，点坐标以 mmap 方式共享读取，避免重复拷贝。
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np

from common.pointcloud_io import fields_xyz, read_pcd_fields
from utils.pointcloud import (
    ROOT, read_json, voxel_downsample, workspace_path, write_json,
)


def query_blocks(input_coord, query_coord, size_m, context_m):
    """把每个查询点分配到唯一核心块，再从该块的上下文盒中取建筑几何。

    两类点集都保留原始行号，便于后续按索引回查；整个场景只构建一次 KD 树，
    避免逐块扫描整份输入 PCD。

    生成器依次产出：
        input_indices: 上下文盒内建筑点的原始索引（按索引排序）
        input_core:    其中位于核心块内部的布尔掩码
        query_indices: 属于当前核心块的查询点原始索引
        lower/upper:   核心块在世界坐标系下的最小/最大角点
    """
    from scipy.spatial import cKDTree

    size, context = np.asarray(size_m), np.asarray(context_m)
    # 以两类点云的最小角点为原点并按块尺寸对齐，保证块边界落在整数网格上
    origin = np.floor(np.minimum(input_coord.min(0), query_coord.min(0)) / size) * size
    # 计算每个查询点所在的三维块坐标，并压平成一维 key
    cells = np.floor((query_coord.astype(np.float64) - origin) / size).astype(np.int64)
    keys = np.ravel_multi_index(cells.T, tuple(cells.max(0) + 1))
    order = np.argsort(keys, kind="stable")
    _, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    # 整个场景只建一次 KD 树，后续按上下文盒半径查询候选点
    tree = cKDTree(input_coord)
    half = size / 2 + context
    for start, count in zip(starts, counts):
        query_indices = order[start:start + count]
        cell = cells[query_indices[0]]
        lower = origin + cell * size
        upper = lower + size
        # 先用无穷范数球查询粗筛，再用轴对齐盒精确裁剪上下文几何
        candidates = np.asarray(tree.query_ball_point((lower + upper) / 2, half.max(), p=np.inf), dtype=np.int64)
        local = input_coord[candidates]
        inside = ((local >= lower - context) & (local < upper + context)).all(1)
        input_indices = np.sort(candidates[inside])
        # 标记哪些输入点位于核心块内部（可视化时区分核心/上下文）
        input_core = ((input_coord[input_indices] >= lower) & (input_coord[input_indices] < upper)).all(1)
        yield input_indices, input_core, query_indices, lower, upper


def render_blocks(root, scene, rows, inputs, queries, costs, settings):
    """在世界坐标系下检查真实分块归属与未经过滤的监督目标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    folder = root / scene["folder"]
    rng = np.random.default_rng(scene["seed"])
    # memberships[q] 记录查询点 q 所属的块编号，用于给第一幅图着色
    memberships = np.empty(len(queries), dtype=np.int64)
    for index, row in enumerate(rows):
        with np.load(root / row["file"]) as archive:
            memberships[archive["query_indices"]] = index
    # 全场概览最多显示 15 万个查询点，抽样不参与训练
    chosen = rng.choice(len(queries), min(len(queries), 150000), replace=False)
    # 选查询点最多的块作为细节子图；选择过程完全不使用代价标签
    row = max(rows, key=lambda r:r["query_points"])
    with np.load(root / row["file"]) as archive:
        ii, core, qi = archive["input_indices"], archive["input_core"], archive["query_indices"]
    # 细节子图最多显示 9 万个建筑点，仅影响绘图
    view_input = rng.choice(len(ii), min(len(ii), 90000), replace=False)
    ii, core = ii[view_input], core[view_input]
    fig = plt.figure(figsize=(20, 7), facecolor="#f7f9fc")
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    # 给每个块分配一种颜色，便于观察空间划分是否均匀
    palette = rng.permutation(len(rows))
    axes[0].scatter(*queries[chosen].T, c=palette[memberships[chosen]], cmap="turbo", s=.25,
                    linewidths=0, depthshade=False, rasterized=True)
    axes[0].set_title(f"All query cores | {len(rows)} blocks\n{len(queries):,} queries, each assigned once")
    for mask, color, label in [(~core, "#bdc7d3", "Context geometry"), (core, "#2575d5", "Core geometry")]:
        axes[1].scatter(*inputs[ii[mask]].T, color=color, s=.4, linewidths=0,
                        depthshade=False, label=label, rasterized=True)
    context_text = "/".join(f"{v:g}" for v in settings["context_m"])
    axes[1].set_title(f"{row['id']} | building geometry\n{row['input_points']:,} points incl. {context_text} m context")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[2].scatter(*inputs[ii].T, color="#cbd2dc", s=.15, alpha=.3,
                    linewidths=0, depthshade=False, rasterized=True)
    cmap = LinearSegmentedColormap.from_list("native_cost", ["#16a34a", "#fde047", "#ef4444"])
    image = axes[2].scatter(*queries[qi].T, c=costs[qi, 0], cmap=cmap, vmin=0, vmax=1,
                            s=1.4, linewidths=0, depthshade=False, rasterized=True)
    axes[2].set_title(f"Same core | nominal Tomogram GT\n{len(qi):,} independent query positions")
    lower, upper = np.asarray(row["core_lower_m"]), np.asarray(row["core_upper_m"])
    corners = np.array([[x, y, z] for x in [lower[0], upper[0]]
                        for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]])
    edges = [[corners[a], corners[b]] for a in range(8) for b in range(a + 1, 8)
             if bin(a ^ b).count("1") == 1]
    for axis in axes[1:]:
        axis.add_collection3d(Line3DCollection(edges, colors="#23384d", linewidths=.8))
    all_bounds = np.array([queries.min(0), queries.max(0)])
    local_bounds = np.array([lower - settings["context_m"], upper + settings["context_m"]])
    for index, axis in enumerate(axes):
        lo, hi = all_bounds if index == 0 else local_bounds
        axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                 xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
        axis.set_box_aspect(hi - lo)
        axis.view_init(elev=30, azim=-55)
        axis.set_proj_type("ortho")
    fig.colorbar(image, ax=axes[2], shrink=.6, pad=.05, label="Native cost / 50")
    size_text = " x ".join(f"{v:g}" for v in settings["core_size_m"])
    fig.suptitle(f"Building geometry and Tomogram queries | seed {scene['seed']} | {scene['split']}\n"
                 f"{size_text} m disjoint cores | {context_text} m context | no target filtering", fontsize=18)
    fig.text(.5, .03, "Saved arrays retain every input point and every native query. Display thinning applies only to geometry and the whole-scene panel.",
             ha="center", fontsize=10, color="#526277")
    fig.subplots_adjust(top=.79, bottom=.12, left=.02, right=.99, wspace=.1)
    fig.savefig(folder / "blocks_overview.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def prepare_scene(root, scene, settings):
    """为单个场景生成共享数组、核心块索引和可视化报告。"""
    started = time.perf_counter()
    folder = root / scene["folder"]
    cache = folder / "blocks"
    cache.mkdir(exist_ok=True)
    # 重复运行时先删除旧块索引，避免残留文件与本次结果混淆
    for old in cache.glob("block_*.npz"):
        old.unlink()
    inputs = fields_xyz(read_pcd_fields(root / scene["pcd"]))
    profiles = scene["profiles"]
    first = read_pcd_fields(folder / profiles[0]["tomogram"])
    queries = fields_xyz(first)
    # 原生导出对所有机器人条件使用相同的 XYZ 与行序；代价仅作为监督目标。
    # 输入几何中不存储任何教师特征，避免信息泄漏。
    costs = np.column_stack([read_pcd_fields(folder / p["tomogram"])["trav_cost"] for p in profiles])
    conditions = np.asarray([p["condition"] for p in profiles], dtype=np.float32)
    # 全楼粗几何：0.5 m 体素下采样，仅提供远距离上下文
    global_coord = voxel_downsample(inputs, settings["global_voxel_m"])
    arrays = dict(input_coord=inputs, query_coord=queries, cost=costs,
                  conditions=conditions, global_coord=global_coord)
    # 每个场景的共享数组各存一份 .npy，后续用 mmap 读取
    for name, array in arrays.items():
        np.save(cache / f"{name}.npy", array)
    rows = []
    for index, (ii, core, qi, lower, upper) in enumerate(query_blocks(
        inputs, queries, settings["core_size_m"], settings["context_m"],
    )):
        name = f"block_{index:05d}"
        path = cache / f"{name}.npz"
        # 未压缩的纯索引文件，避免对大点云数组反复压缩而浪费时间
        np.savez(path, input_indices=ii, input_core=core, query_indices=qi)
        # 每条块记录保存几何/查询索引文件路径、核心块范围与点数统计
        rows.append(dict(id=name, scene=scene["id"], seed=scene["seed"], split=scene["split"],
                         batch=scene["batch"], file=str(path.relative_to(root)),
                         core_lower_m=lower.tolist(), core_upper_m=upper.tolist(),
                         input_points=len(ii), input_core_points=int(core.sum()), query_points=len(qi)))
    # 场景级记录汇总输入点数、查询点数、粗几何点数与条件数量
    record = dict(
        id=scene["id"], seed=scene["seed"], split=scene["split"], batch=scene["batch"],
        input_pcd=scene["pcd"], input_pcd_sha256=scene["pcd_sha256"],
        query_sources=[dict(id=p["id"], condition=p["condition"], tomogram=f"{scene['folder']}/{p['tomogram']}") for p in profiles],
        profiles=profiles, arrays={name:str((cache / f"{name}.npy").relative_to(root)) for name in arrays},
        input_points=len(inputs), query_points=len(queries), global_points=len(global_coord),
        conditions=len(conditions), blocks=len(rows), generation_s=time.perf_counter() - started,
        overview=f"{scene['folder']}/blocks_overview.png",
    )
    # 生成分块总览图，并写回场景报告与 scene.json 的阶段信息
    render_blocks(root, scene, rows, inputs, queries, costs, settings)
    write_json(folder / "blocks.report.json", record)
    scene_metadata = read_json(folder / "scene.json")
    scene_metadata["blocks"] = dict(directory="blocks", report="blocks.report.json", overview="blocks_overview.png",
                                    count=len(rows), query_points=len(queries), config=settings)
    scene_metadata["stage"] = "building geometry and native Tomogram spatial blocks ready, query model pending"
    write_json(folder / "scene.json", scene_metadata)
    print(f"seed={scene['seed']} split={scene['split']} blocks={len(rows)} "
          f"input={len(inputs):,} queries={len(queries):,} global={len(global_coord):,} "
          f"prepare_s={record['generation_s']:.2f}", flush=True)
    return record, rows


def prepare_blocks(root, settings, batches=None):
    """汇总所有场景的分块结果，写入 blocks.json 与 manifest.json，并补近邻表。"""
    root = Path(root)
    source = read_json(root / "manifest.json")
    # 顶层元数据记录条件定义、代价尺度与分块参数，供训练侧直接读取
    metadata = dict(version="tomogram_query_blocks_v1", source_manifest="manifest.json",
                    condition_names=source["condition_names"], condition_scale=source["condition_scale"],
                    cost_scale=source["cost_scale"], coordinate_frame="map", units="m", z_axis="up",
                    blocks_config=settings, scenes=[], blocks=[])
    if batches and (root / "blocks.json").exists():
        # 增量重建：先移除指定批次，再重新加入本次生成的场景与块
        previous = read_json(root / "blocks.json")
        for key in ("scenes", "blocks"):
            metadata[key] = [row for row in previous[key] if row["batch"] not in batches]
    for scene in source["scenes"]:
        if batches and scene["batch"] not in batches:
            continue
        record, rows = prepare_scene(root, scene, settings)
        metadata["scenes"].append(record)
        metadata["blocks"].extend(rows)
        # 每完成一个场景就落盘，长任务中断后可保留已生成结果
        write_json(root / "blocks.json", metadata)
    # 按 train/val/test 统计场景数、块数、查询点数与监督目标数
    counts = {}
    for split in sorted({r["split"] for r in metadata["scenes"]}):
        scenes = [r for r in metadata["scenes"] if r["split"] == split]
        counts[split] = dict(scenes=len(scenes), blocks=sum(r["blocks"] for r in scenes),
                             queries=sum(r["query_points"] for r in scenes),
                             cost_targets=sum(r["query_points"] * r["conditions"] for r in scenes))
    metadata["splits"] = counts
    write_json(root / "blocks.json", metadata)
    # 把分块索引同步进数据集 manifest，便于总览与后续工具定位
    source["prepared_blocks"] = dict(index="blocks.json", config=settings, splits=counts)
    write_json(root / "manifest.json", source)
    print(f"Prepared: {root / 'blocks.json'}\n{counts}", flush=True)
    # 分块完成后立即计算查询近邻表（仅依赖几何，不使用代价标签）
    return prepare_neighbor_tables(root, settings["query_neighbors"], batches)


def prepare_neighbor_tables(root, settings, batches=None):
    """在模型执行前一次性缓存仅依赖几何的三维查询近邻。

    对每个查询点：
    - 在局部精细几何中找 local_k 个最近邻（每个核心块单独建立 KD 树）；
    - 在全楼粗几何中找 global_k 个最近邻（整个场景共用一棵 KD 树）。
    近邻只依据坐标计算，不使用代价标签，因此不会造成监督泄漏。
    """
    from scipy.spatial import cKDTree

    root = Path(root)
    metadata = read_json(root / "blocks.json")
    metadata["blocks_config"]["query_neighbors"] = settings
    for scene in metadata["scenes"]:
        if batches and scene["batch"] not in batches:
            continue
        started = time.perf_counter()
        # 共享数组按 mmap 方式打开，避免把整份点云读入内存
        inputs = np.load(root / scene["arrays"]["input_coord"], mmap_mode="r")
        queries = np.load(root / scene["arrays"]["query_coord"], mmap_mode="r")
        coarse = np.load(root / scene["arrays"]["global_coord"], mmap_mode="r")
        # 全楼粗几何近邻：整个场景只查询一次
        distances, global_neighbors = cKDTree(coarse).query(queries, k=settings["global_k"])
        path = Path(scene["arrays"]["query_coord"]).with_name("global_neighbors.npy")
        np.save(root / path, global_neighbors)
        scene["arrays"]["global_neighbors"] = str(path)
        local_max = 0.
        for row in metadata["blocks"]:
            if row["scene"] != scene["id"]:
                continue
            with np.load(root / row["file"]) as archive:
                ii, core, qi = archive["input_indices"], archive["input_core"], archive["query_indices"]
            # 局部近邻：仅用当前块的上下文几何，索引相对块内点云
            local_distances, local_neighbors = cKDTree(inputs[ii]).query(queries[qi], k=settings["local_k"])
            # 把近邻表追加到块索引文件，原索引字段保持不变
            np.savez(root / row["file"], input_indices=ii, input_core=core,
                     query_indices=qi, query_neighbors=local_neighbors)
            local_max = max(local_max, float(local_distances.max()))
        # 记录近邻统计，便于发现 K 过小或查询点远离几何等异常
        scene["query_neighbors"] = dict(**settings, local_max_distance_m=local_max,
                                        global_max_distance_m=float(distances.max()),
                                        preparation_s=time.perf_counter() - started)
        report_path = (root / scene["arrays"]["query_coord"]).parents[1] / "blocks.report.json"
        write_json(report_path, scene)
        print(f"{scene['id']}: local K={settings['local_k']}, global K={settings['global_k']}, "
              f"neighbour tables {scene['query_neighbors']['preparation_s']:.2f}s", flush=True)
    # 更新全局索引与 manifest，使训练侧知道近邻表已经就绪
    write_json(root / "blocks.json", metadata)
    source = read_json(root / "manifest.json")
    source["prepared_blocks"]["config"]["query_neighbors"] = settings
    write_json(root / "manifest.json", source)
    return metadata


class TomogramBlocks:
    """Map 风格数据集：几何与监督目标使用各自独立的源索引。

    坐标只按共享的块中心平移，不做缩放；将 origin 加回
    input_coord、query_coord 或 global_coord 即可恢复地图 XYZ。
    场景级数组以 mmap 方式按需打开并缓存，避免多个进程重复拷贝。
    """
    def __init__(self, root, split):
        """加载 blocks.json，并筛选指定 split 的块。"""
        self.root = Path(root)
        self.metadata = read_json(self.root / "blocks.json")
        # 只保留指定 split（train/val/test）的块
        self.rows = [r for r in self.metadata["blocks"] if r["split"] == split]
        self.scene_rows = {r["id"]:r for r in self.metadata["scenes"]}
        self.scenes = {}

    def __len__(self):
        """返回当前 split 的块数量。"""
        return len(self.rows)

    def __getitem__(self, index):
        """读取一个块的输入几何、查询点、代价标签与近邻索引。"""
        row = self.rows[index]
        name = row["scene"]
        # 每个场景的共享数组只打开一次，之后复用 mmap 句柄
        if name not in self.scenes:
            self.scenes[name] = {key:np.load(self.root / path, mmap_mode="r")
                                 for key,path in self.scene_rows[name]["arrays"].items()}
        scene = self.scenes[name]
        with np.load(self.root / row["file"]) as archive:
            ii, core, qi = archive["input_indices"], archive["input_core"], archive["query_indices"]
            neighbours = archive["query_neighbors"]
        # 以核心块中心作为局部坐标原点，保证不同块坐标尺度一致
        origin = (np.asarray(row["core_lower_m"], dtype=np.float32)
                  + np.asarray(row["core_upper_m"], dtype=np.float32)) / 2
        # 输入几何只取上下文索引；查询点只取核心块索引；粗几何保留全楼
        return dict(input_coord=scene["input_coord"][ii] - origin,
                    query_coord=scene["query_coord"][qi] - origin,
                    global_coord=scene["global_coord"][:] - origin,
                    target=np.array(scene["cost"][qi]), conditions=np.array(scene["conditions"]),
                    query_neighbors=neighbours, global_neighbors=np.array(scene["global_neighbors"][qi]),
                    query_neighbor_mask=np.ones_like(neighbours, dtype=np.bool_),
                    input_core=core, input_index=ii, query_index=qi, origin=origin,
                    scene=name, block=row["id"], split=row["split"])


def collate_tomogram_blocks(samples):
    """拼接变长的输入/查询/粗几何点云，并用偏移与 batch 索引隔离各块。"""
    # 点级字段直接拼接；块级字段在下方单独处理
    batch = {key:np.concatenate([s[key] for s in samples]) for key in (
        "input_coord", "query_coord", "global_coord", "target", "input_core", "input_index", "query_index")}
    for prefix in ("input", "query", "global"):
        # offset 标记每个块在拼接数组中的结束位置，batch 标记每个点所属块
        sizes = [len(s[prefix + "_coord"]) for s in samples]
        batch[prefix + "_offset"] = np.cumsum(sizes, dtype=np.int64)
        batch[prefix + "_batch"] = np.repeat(np.arange(len(samples)), sizes)
    for key,prefix in [("query_neighbors", "input"), ("global_neighbors", "global")]:
        # 近邻索引是块内相对索引，拼接后需要加上对应块的点数偏移
        starts = np.cumsum([0, *[len(s[prefix + "_coord"]) for s in samples[:-1]]])
        batch[key] = np.concatenate([s[key] + start for s,start in zip(samples,starts)])
    for key in ("conditions", "origin"):
        batch[key] = np.stack([s[key] for s in samples])
    batch["scene"] = [s["scene"] for s in samples]
    batch["block"] = [s["block"] for s in samples]
    return batch


def main():
    """命令行入口：默认重建全部分块，也可只补查询近邻表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Traversability/configs/dataset.json")
    parser.add_argument("--batch", nargs="+", help="只重建这些批次的数据")
    parser.add_argument("--neighbors-only", action="store_true", help="仅给已有分块补充查询近邻表。")
    args = parser.parse_args()
    # 固定 Matplotlib 缓存目录，避免多进程争用用户主目录
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "datasets/cache/matplotlib"))
    cfg = read_json(args.config)
    if args.neighbors_only:
        # 不重算分块，只根据现有几何补 local/global 近邻
        prepare_neighbor_tables(workspace_path(cfg["output"]), cfg["blocks"]["query_neighbors"], args.batch)
    else:
        # 完整流程：共享数组、核心块、统计报告与近邻表
        prepare_blocks(workspace_path(cfg["output"]), cfg["blocks"], args.batch)


if __name__ == "__main__":
    main()
