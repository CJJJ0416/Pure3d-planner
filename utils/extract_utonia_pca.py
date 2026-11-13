"""Extract full-scene Utonia features and export PCA colors in original PCD order.

Run from the workspace root with ``python -m utils.extract_utonia_pca --pcd ...``.
Feature rows are stored once per model input voxel; ``inverse.npy`` restores input order.
The encoder sees the entire scene in one pass. Only feature export and PCA are streamed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import point_transformer_v3 as ptv3

from common.pointcloud_io.pcd import fields_xyz, read_pcd_fields


def hierarchy(output):
    levels = [output]
    while "pooling_parent" in levels[-1]:
        levels.append(levels[-1]["pooling_parent"])
    return levels[::-1]


def feature_rows(levels, start, end):
    """Equivalent to upcast(levels=None), gathering just these level-0 rows."""
    ids = torch.arange(start, end, device=levels[0].feat.device)
    pieces = [levels[0].feat[ids]]
    for level in levels[1:]:
        ids = level.pooling_inverse[ids]
        pieces.append(level.feat[ids])
    return torch.cat(pieces, dim=1)


def write_colored_clouds(folder, source, xyz, scores, rgb):
    """Write binary PCD with original fields plus RGB/PC1-3; standard colored PLY."""
    dtype = np.dtype(source.dtype.descr + [("rgb", "<f4"), ("pc1", "<f4"),
                                         ("pc2", "<f4"), ("pc3", "<f4")])
    rows = np.empty(len(source), dtype=dtype)
    for name in source.dtype.names:
        rows[name] = source[name]
    packed = ((rgb[:, 0].astype(np.uint32) << 16)
              | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32))
    rows["rgb"] = packed.view(np.float32)
    for i in range(3):
        rows[f"pc{i + 1}"] = scores[:, i]
    names = dtype.names
    sizes = [str(dtype.fields[k][0].itemsize) for k in names]
    types = [{"f": "F", "i": "I", "u": "U"}[dtype.fields[k][0].kind] for k in names]
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS " + " ".join(names)
              + "\nSIZE " + " ".join(sizes) + "\nTYPE " + " ".join(types)
              + "\nCOUNT " + " ".join(["1"] * len(names))
              + f"\nWIDTH {len(rows)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              + f"POINTS {len(rows)}\nDATA binary\n")
    with (folder / "pca_colored.pcd").open("wb") as stream:
        stream.write(header.encode("ascii"))
        rows.tofile(stream)
    del rows
    ply = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(("x", "y", "z")):
        ply[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        ply[name] = rgb[:, i]
    with (folder / "pca_colored.ply").open("wb") as stream:
        stream.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "end_header\n").encode("ascii"))
        ply.tofile(stream)


def previews(folder, xyz, rgb, explained):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rng = np.random.default_rng(0)
    ids = np.sort(rng.choice(len(xyz), min(160_000, len(xyz)), replace=False))
    p, c = xyz[ids], rgb[ids].astype(np.float32) / 255
    height = plt.colormaps["viridis"]((p[:, 2] - p[:, 2].min()) / np.ptp(p[:, 2]))[:, :3]
    fig = plt.figure(figsize=(16, 10), facecolor="#f7f8fa")
    views = [(height, "Geometry | height", 27, -55),
             (c, "Utonia tiny | feature PCA", 27, -55),
             (c, "Feature PCA | top view", 90, -90),
             (c, "Feature PCA | opposite view", 20, 125)]
    span = float(np.ptp(p, axis=0).max())
    mid = (p.min(0) + p.max(0)) / 2
    for i, (color, title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d", facecolor="#f7f8fa")
        ax.scatter(*p.T, c=color, s=.35, depthshade=False, linewidths=0, rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(p[:, 2].min(), p[:, 2].max())
        ax.set_box_aspect((span, span, np.ptp(p[:, 2])))
        ax.set_title(title, fontsize=14, pad=0)
        ax.set_axis_off()
    fig.suptitle("building2_9.pcd  /  Utonia tiny", fontsize=23, weight="bold", y=.97)
    fig.text(.5, .035, f"Full scene: {len(xyz):,} points | 486-D features -> PC1/PC2/PC3 -> RGB"
             f" | top-3 variance: {100 * sum(explained):.1f}%\n"
             "Colors show feature variation; they are not semantic labels. "
             "Preview displays a fixed sample of 160,000 points.",
             ha="center", fontsize=11, color="#424b59")
    fig.subplots_adjust(left=.015, right=.985, bottom=.08, top=.9, wspace=0, hspace=.02)
    fig.savefig(folder / "preview.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Embedded Plotly runtime makes the viewer usable offline.
    ids = ids[::2]
    p, c = xyz[ids], rgb[ids]
    colors = [f"rgb({r},{g},{b})" for r, g, b in c]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Geometry / height", "Utonia tiny / PCA features"))
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                             marker=dict(size=1.6, color=p[:, 2], colorscale="Viridis"),
                             hoverinfo="x+y+z", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                             marker=dict(size=1.6, color=colors),
                             hoverinfo="x+y+z", showlegend=False), row=1, col=2)
    scene = dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)")
    fig.update_layout(title="building2_9 | Utonia tiny PCA (80,000-point display sample)",
                      scene=scene, scene2=scene, margin=dict(l=0, r=0, b=0, t=70), height=850)
    fig.write_html(folder / "viewer.html", include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    started = time.perf_counter()
    source = read_pcd_fields(args.pcd)
    xyz = fields_xyz(source)
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("Need at least three finite input points")
    normal_names = ("normal_x", "normal_y", "normal_z")
    has_normals = all(k in source.dtype.names for k in normal_names)
    normals = np.column_stack([source[k] for k in normal_names]) if has_normals else np.zeros_like(xyz)
    normal_valid = np.isfinite(normals).all(1)
    normals[~normal_valid] = 0
    colors = np.zeros_like(xyz)
    print(f"Read {len(xyz):,} points; original normals={has_normals}; RGB=zeros", flush=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = ptv3.load("utonia_tiny", weights=args.weights, device=device,
                      precision="auto", head=False, warmup=False).requires_grad_(False)
    with torch.inference_mode():
        point, inverse_t = ptv3.prepare(xyz, colors, normals, preset="utonia_tiny", device=device)
        inverse = inverse_t.cpu().numpy().astype(np.int32)
        np.save(args.output / "inverse.npy", inverse)
        # Save a world-coordinate representative for every feature row.
        first = np.full(len(point.feat), len(xyz), dtype=np.int64)
        np.minimum.at(first, inverse, np.arange(len(xyz)))
        np.save(args.output / "feature_xyz.npy", xyz[first])
        print(f"Model input: {len(point.feat):,} voxels (0.02 m in original coordinates)", flush=True)
        tick = time.perf_counter()
        output = model(point)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_s = time.perf_counter() - tick
        levels = hierarchy(output)
        count = len(levels[0].feat)
        channels = sum(level.feat.shape[1] for level in levels)
        print(f"One full-scene forward: {forward_s:.2f}s; {channels} channels", flush=True)
        features = np.lib.format.open_memmap(args.output / "features.npy", mode="w+",
                                             dtype=np.float16, shape=(count, channels))
        total = torch.zeros(channels, dtype=torch.float64, device=device)
        gram = torch.zeros((channels, channels), dtype=torch.float64, device=device)
        for a in range(0, count, args.chunk_size):
            b = min(a + args.chunk_size, count)
            # PCA is fitted to precisely the exported float16 features, promoted to fp32.
            f = feature_rows(levels, a, b).half()
            features[a:b] = f.cpu().numpy()
            f = f.float()
            if not torch.isfinite(f).all():
                raise ValueError(f"Non-finite features in rows {a}:{b}")
            total += f.sum(0, dtype=torch.float64)
            gram += (f.T @ f).double()
            if a == 0 or b == count or (a // args.chunk_size) % 10 == 0:
                print(f"Export/PCA moments: {b:,}/{count:,}", flush=True)
        features.flush()
        mean = total / count
        covariance = (gram - count * mean[:, None] * mean[None, :]) / (count - 1)
        covariance = covariance.cpu().numpy()
        mean = mean.cpu().numpy()
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    del output, levels, point, inverse_t, f, model, gram, total
    if device.type == "cuda":
        torch.cuda.empty_cache()
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    components = vectors[:, :3].T.copy()
    # Fix the arbitrary sign of every eigenvector for repeatable color orientation.
    for component in components:
        if component[np.abs(component).argmax()] < 0:
            component *= -1
    explained = values[:3] / np.maximum(values, 0).sum()
    grid_scores = np.lib.format.open_memmap(args.output / "pca_grid.npy", mode="w+",
                                           dtype=np.float32, shape=(count, 3))
    mean32, components32 = mean.astype(np.float32), components.astype(np.float32)
    for a in range(0, count, args.chunk_size):
        b = min(a + args.chunk_size, count)
        grid_scores[a:b] = (features[a:b].astype(np.float32) - mean32) @ components32.T
    grid_scores.flush()
    low, high = np.percentile(grid_scores, [2, 98], axis=0).astype(np.float32)
    scores = np.lib.format.open_memmap(args.output / "pca.npy", mode="w+",
                                       dtype=np.float32, shape=(len(xyz), 3))
    rgb = np.empty((len(xyz), 3), np.uint8)
    for a in range(0, len(xyz), args.chunk_size):
        b = min(a + args.chunk_size, len(xyz))
        scores[a:b] = grid_scores[inverse[a:b]]
        rgb[a:b] = (255 * np.clip((scores[a:b] - low) / np.maximum(high - low, 1e-8), 0, 1)).round().astype(np.uint8)
    scores.flush()
    np.save(args.output / "rgb.npy", rgb)
    np.savez(args.output / "pca_model.npz", mean=mean32, components=components32,
             explained_variance=values[:3], explained_variance_ratio=explained,
             rgb_percentile_low=low, rgb_percentile_high=high)
    print(f"PCA explained variance ratios: {explained.tolist()}", flush=True)
    write_colored_clouds(args.output, source, xyz, scores, rgb)
    print("Colored PCD/PLY saved; rendering previews", flush=True)
    previews(args.output, xyz, rgb, explained)
    metadata = dict(input=str(args.pcd.resolve()), model="utonia_tiny", device=str(device),
                    precision="fp16 autocast" if device.type == "cuda" else "fp32",
                    input_points=len(xyz), feature_rows=count, feature_channels=channels,
                    feature_dtype="float16", input_normals=has_normals,
                    invalid_normals_replaced_with_zero=int((~normal_valid).sum()), input_rgb="zeros",
                    scale=.5, model_grid_size_m=.01, physical_grid_size_m=.02,
                    scene_passes=1, spatial_chunking=False, random_subsampling=False,
                    pca_fit="all model voxels, equal voxel weight, centered, no channel standardization",
                    pca_components=3, rgb_mapping="PC1/PC2/PC3, per-channel 2nd-98th percentiles",
                    explained_variance_ratio=explained.tolist(), forward_s=forward_s,
                    peak_cuda_allocated_mib=peak_mib, elapsed_s=time.perf_counter() - started,
                    torch_version=torch.__version__, complete=True)
    if args.weights:
        metadata["weights"] = str(args.weights.resolve())
        metadata["weights_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# building2_9 Utonia tiny PCA\n\n"
        f"Input: `{args.pcd.resolve()}`. {len(xyz):,} original points -> {count:,} model voxels.\n\n"
        "The entire scene is encoded once at Utonia's default indoor scale 0.5 and 1 cm grid "
        "(2 cm in the original coordinates). Original normals are used, RGB is zero. "
        "All five encoder levels are concatenated into 486 channels. PCA is fitted globally "
        "over all voxel features. No spatial tiles or random input sampling are used.\n\n"
        "- `pca_colored.pcd`: all original points/fields/order plus packed RGB and pc1/pc2/pc3.\n"
        "- `pca_colored.ply`: all original XYZ with PCA RGB, ready for CloudCompare/Open3D.\n"
        "- `pca.npy`: float32 [original points, 3], centered PC scores in original input order.\n"
        "- `features.npy`: float16 [model voxels, 486], unprojected features. Load with mmap_mode='r'.\n"
        "- `inverse.npy`: int32 [original points], original point -> feature row.\n"
        "- `feature_xyz.npy`: original XYZ representative for each feature row.\n"
        "- `pca_grid.npy`: float32 [model voxels, 3]; `pca_model.npz`: mean, components, variance, RGB limits.\n"
        "- `rgb.npy`: uint8 [original points, 3]; `preview.png`: four views.\n"
        "- `viewer.html`: offline interactive geometry/PCA comparison, 80,000 display points.\n\n"
        "PC1, PC2 and PC3 map to R, G and B with per-channel 2nd-98th percentile clipping. "
        "Colors show pretrained feature differences and are not semantic or traversability labels. "
        "Points in the same model voxel share features.\n\n"
        "```python\nimport numpy as np\nf = np.load('features.npy', mmap_mode='r')\n"
        "inverse = np.load('inverse.npy', mmap_mode='r')\n"
        "pca = np.load('pca.npy', mmap_mode='r')\n"
        "first_100_original_features = f[inverse[:100]]\n```\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
