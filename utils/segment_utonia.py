"""Run Utonia's released ScanNet-20 linear probe on an entire local PCD scene.

The encoder runs once; streamed feature gathering avoids a full N x 1386 upcast.
Outputs retain all original points and their order. Labels are zero-based ScanNet-20.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import point_transformer_v3 as ptv3

from common.pointcloud_io.pcd import fields_xyz, read_pcd_fields
from utils.extract_utonia_pca import feature_rows, hierarchy


# ScanNet benchmark colors, in the same order as PRESETS['utonia'].heads['scannet20'].
PALETTE = np.array([
    (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120),
    (188, 189, 34), (140, 86, 75), (255, 152, 150), (214, 39, 40),
    (197, 176, 213), (148, 103, 189), (196, 156, 148), (23, 190, 207),
    (247, 182, 210), (219, 219, 141), (255, 127, 14), (158, 218, 229),
    (44, 160, 44), (112, 128, 144), (227, 119, 194), (82, 84, 163),
], dtype=np.uint8)
NYU40_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39], np.uint32)


class RopeByRows(torch.nn.Module):
    """Bound temporary memory of row-independent RoPE, keeping attention unchanged."""
    def __init__(self, inner, chunk_size):
        super().__init__()
        self.inner = inner
        self.chunk_size = chunk_size

    def forward(self, qkv, coord):
        if len(qkv) <= self.chunk_size:
            return self.inner(qkv, coord)
        result = torch.empty_like(qkv)
        for a in range(0, len(qkv), self.chunk_size):
            b = min(a + self.chunk_size, len(qkv))
            result[a:b] = self.inner(qkv[a:b], coord[a:b])
        return result


def bound_rope_memory(model, chunk_size):
    from point_transformer_v3.modules import SerializedAttention
    for module in model.modules():
        if isinstance(module, SerializedAttention) and module.rope is not None:
            module.rope = RopeByRows(module.rope, chunk_size)


def write_clouds(folder, source, xyz, labels, confidence):
    rgb = PALETTE[labels]
    fields = source.dtype.descr + [("rgb", "<f4"), ("label", "<u4"),
                                  ("nyu40_id", "<u4"), ("confidence", "<f4")]
    rows = np.empty(len(xyz), dtype=fields)
    for name in source.dtype.names:
        rows[name] = source[name]
    packed = ((rgb[:, 0].astype(np.uint32) << 16)
              | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32))
    rows["rgb"] = packed.view(np.float32)
    rows["label"], rows["nyu40_id"], rows["confidence"] = labels, NYU40_IDS[labels], confidence
    names = rows.dtype.names
    sizes = [str(rows.dtype.fields[k][0].itemsize) for k in names]
    types = [{"f": "F", "u": "U", "i": "I"}[rows.dtype.fields[k][0].kind] for k in names]
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS " + " ".join(names)
              + "\nSIZE " + " ".join(sizes) + "\nTYPE " + " ".join(types)
              + "\nCOUNT " + " ".join(["1"] * len(names))
              + f"\nWIDTH {len(rows)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              + f"POINTS {len(rows)}\nDATA binary\n")
    with (folder / "seg_colored.pcd").open("wb") as stream:
        stream.write(header.encode("ascii"))
        rows.tofile(stream)
    del rows
    ply = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                    ("label", "<u4"), ("confidence", "<f4")])
    for i, name in enumerate(("x", "y", "z")):
        ply[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        ply[name] = rgb[:, i]
    ply["label"], ply["confidence"] = labels, confidence
    with (folder / "seg_colored.ply").open("wb") as stream:
        stream.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "property uint label\nproperty float confidence\nend_header\n").encode("ascii"))
        ply.tofile(stream)


def previews(folder, xyz, labels, confidence, names, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import plotly.graph_objects as go

    ids = np.sort(np.random.default_rng(0).choice(len(xyz), min(160_000, len(xyz)), replace=False))
    p, lab, conf = xyz[ids], labels[ids], confidence[ids]
    rgb = PALETTE[lab].astype(np.float32) / 255
    cut = float((xyz[:, 2].min() + xyz[:, 2].max()) / 2)
    fig = plt.figure(figsize=(16, 11), facecolor="#f7f8fa")
    views = [(np.ones(len(p), bool), 27, -55, rgb, "Full scene | semantic classes"),
             (np.ones(len(p), bool), 90, -90, rgb, "Top view | semantic classes"),
             (p[:, 2] <= cut, 27, -55, rgb, f"Lower half | Z <= {cut:.2f} m"),
             (p[:, 2] <= cut, 27, -55, plt.colormaps["magma"](conf)[:, :3],
              "Lower half | max softmax probability")]
    span = float(np.ptp(p, axis=0).max())
    mid = (p.min(0) + p.max(0)) / 2
    for i, (keep, elev, azim, color, title) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d", facecolor="#f7f8fa")
        ax.scatter(*p[keep].T, c=color[keep], s=.5, depthshade=False, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(p[:, 2].min(), p[:, 2].max())
        ax.set_box_aspect((span, span, np.ptp(p[:, 2])))
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
    cax = fig.add_axes([.935, .255, .008, .15])
    fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="magma"),
                 cax=cax, ticks=[0, .5, 1])
    cax.tick_params(labelsize=9)
    active = np.flatnonzero(counts)
    fig.legend(handles=[Patch(color=PALETTE[i] / 255,
                             label=f"{names[i]} ({counts[i] / len(xyz):.1%})") for i in active],
               loc="lower center", bbox_to_anchor=(.5, .048), ncol=min(7, len(active)),
               frameon=False, fontsize=10)
    fig.suptitle("building2_9.pcd  /  Utonia + ScanNet-20 segmentation", fontsize=21, weight="bold", y=.975)
    fig.text(.5, .018, "All 2,000,000 points exported | Display sample: 160,000 points | "
             "Pretrained predictions; no ground-truth evaluation | No stair/railing classes",
             ha="center", fontsize=10, color="#424b59")
    fig.subplots_adjust(left=.01, right=.99, top=.91, bottom=.16, wspace=0, hspace=.02)
    fig.savefig(folder / "preview.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # A class legend and height clipping make the indoor scene inspectable through its roof.
    ids = ids[::2]
    p, lab, conf = xyz[ids], labels[ids], confidence[ids]
    fig = go.Figure()
    for i in active:
        keep = lab == i
        c = PALETTE[i]
        fig.add_trace(go.Scatter3d(x=p[keep, 0], y=p[keep, 1], z=p[keep, 2], mode="markers",
                                  name=f"{names[i]} ({counts[i] / len(xyz):.1%})",
                                  customdata=conf[keep], marker=dict(size=1.7, color=f"rgb({c[0]},{c[1]},{c[2]})"),
                                  hovertemplate=f"{names[i]}<br>X=%{{x:.2f}} Y=%{{y:.2f}} Z=%{{z:.2f}}"
                                                "<br>max softmax=%{customdata:.3f}<extra></extra>"))
    fig.update_layout(title="building2_9 | Utonia / ScanNet-20 (80,000 display points)",
                      scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
                      margin=dict(l=0, r=0, b=0, t=60), height=850,
                      legend=dict(title="Click to hide a class; double-click to isolate"))
    post = """
    const graph = document.getElementById('{plot_id}');
    const originals = graph.data.map(t => ({x:Array.from(t.x),y:Array.from(t.y),z:Array.from(t.z),c:Array.from(t.customdata)}));
    const bar = document.createElement('div');
    bar.style.cssText='padding:12px 18px;font:16px sans-serif;background:#f3f5f8;';
    bar.innerHTML='Height clipping: Z ≤ <span id="zvalue"></span> m <input id="zcut" type="range" step="0.05" style="width:40%;vertical-align:middle">';
    graph.parentNode.insertBefore(bar,graph);
    const slider=bar.querySelector('#zcut'),value=bar.querySelector('#zvalue');
    const allz=originals.flatMap(t=>t.z);
    slider.min=allz.reduce((a,b)=>Math.min(a,b),Infinity);slider.max=allz.reduce((a,b)=>Math.max(a,b),-Infinity);slider.value=slider.max;value.textContent=Number(slider.value).toFixed(2);
    slider.addEventListener('input',()=>{const zmax=Number(slider.value);value.textContent=zmax.toFixed(2);
      const xx=[],yy=[],zz=[],cc=[];for(const t of originals){const x=[],y=[],z=[],c=[];
        for(let j=0;j<t.z.length;j++){if(t.z[j]<=zmax){x.push(t.x[j]);y.push(t.y[j]);z.push(t.z[j]);c.push(t.c[j]);}}
        xx.push(x);yy.push(y);zz.push(z);cc.push(c);}
      Plotly.restyle(graph,{x:xx,y:yy,z:zz,customdata:cc});});
    """
    # Plain arrays keep post_script filtering compatible with Plotly's HTML serialization.
    for trace in fig.data:
        for key in ("x", "y", "z", "customdata"):
            trace[key] = np.asarray(trace[key]).tolist()
    fig.write_html(folder / "viewer.html", include_plotlyjs=True, post_script=post, full_html=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--rope-chunk-size", type=int, default=65536)
    args = parser.parse_args()
    if min(args.chunk_size, args.rope_chunk_size) <= 0:
        parser.error("Chunk sizes must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    source = read_pcd_fields(args.pcd)
    xyz = fields_xyz(source)
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("Input coordinates must be finite and nonempty")
    keys = ("normal_x", "normal_y", "normal_z")
    has_normals = all(k in source.dtype.names for k in keys)
    normals = np.column_stack([source[k] for k in keys]) if has_normals else np.zeros_like(xyz)
    normals[~np.isfinite(normals).all(1)] = 0
    colors = np.zeros_like(xyz)
    device = torch.device(args.device)
    print(f"Read {len(xyz):,} points; original normals={has_normals}; RGB=zeros", flush=True)
    model = ptv3.load("utonia", device=device, precision="auto", head="scannet20", warmup=False)
    model.requires_grad_(False)
    bound_rope_memory(model, args.rope_chunk_size)
    names = ptv3.PRESETS["utonia"].heads["scannet20"].class_names
    print(f"Loaded Utonia + trained {model.seg_head.in_features}->20 ScanNet linear probe", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        point, inverse_t = ptv3.prepare(xyz, colors, normals, preset="utonia", device=device)
        inverse = inverse_t.cpu().numpy().astype(np.int32)
        count = len(point.feat)
        np.save(args.output / "inverse.npy", inverse)
        print(f"Full-scene forward: {count:,} model voxels", flush=True)
        tick = time.perf_counter()
        output = model(point)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_s = time.perf_counter() - tick
        print(f"Encoder completed in {forward_s:.2f}s; streaming segmentation head", flush=True)
        levels = hierarchy(output)
        logits = np.lib.format.open_memmap(args.output / "logits_grid.npy", mode="w+",
                                          dtype=np.float32, shape=(count, 20))
        for a in range(0, count, args.chunk_size):
            b = min(a + args.chunk_size, count)
            features = feature_rows(levels, a, b).to(model.seg_head.weight.dtype)
            values = model.seg_head(features)
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite logits in {a}:{b}")
            logits[a:b] = values.cpu().numpy()
        logits.flush()
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    weights_info = model.weights_info
    del model, output, levels, point, inverse_t, features, values
    if device.type == "cuda":
        torch.cuda.empty_cache()
    grid_labels = np.empty(count, np.uint8)
    grid_confidence = np.empty(count, np.float32)
    for a in range(0, count, args.chunk_size):
        b = min(a + args.chunk_size, count)
        v = np.array(logits[a:b])
        grid_labels[a:b] = v.argmax(1)
        exp = np.exp(v - v.max(1, keepdims=True))
        grid_confidence[a:b] = 1 / exp.sum(1)
    labels, confidence = grid_labels[inverse], grid_confidence[inverse]
    np.save(args.output / "labels.npy", labels)
    np.save(args.output / "confidence.npy", confidence)
    counts = np.bincount(labels, minlength=20)
    classes = [dict(label=i, nyu40_id=int(NYU40_IDS[i]), name=name,
                    rgb=PALETTE[i].tolist(), points=int(counts[i]), fraction=float(counts[i] / len(xyz)))
               for i, name in enumerate(names)]
    (args.output / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    with (args.output / "class_counts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "nyu40_id", "class", "points", "fraction"])
        for row in classes:
            writer.writerow([row[k] for k in ("label", "nyu40_id", "name", "points", "fraction")])
    for i in counts.argsort()[::-1]:
        if counts[i]:
            print(f"{names[i]}: {counts[i]:,} ({counts[i] / len(xyz):.2%})", flush=True)
    write_clouds(args.output, source, xyz, labels, confidence)
    print("Colored PCD/PLY saved; rendering preview/viewer", flush=True)
    previews(args.output, xyz, labels, confidence, names, counts)
    metadata = dict(input=str(args.pcd.resolve()), model="utonia", head="scannet20",
                    feature_channels=1386, input_points=len(xyz), model_voxels=count,
                    scene_passes=1, spatial_chunking=False, pointwise_rope_chunk_size=args.rope_chunk_size,
                    input_rgb="zeros", original_normals=has_normals,
                    physical_grid_size_m=.02, device=str(device), precision="auto",
                    forward_s=forward_s, peak_cuda_allocated_mib=peak,
                    mean_max_softmax=float(confidence.mean()), median_max_softmax=float(np.median(confidence)),
                    fraction_max_softmax_below_0_5=float((confidence < .5).mean()),
                    ground_truth_available=False, accuracy_metrics=None,
                    class_ids="zero-based contiguous ScanNet-20; separate nyu40_id field",
                    missing_target_classes=["stair", "railing", "person"],
                    weights_info=weights_info, torch_version=torch.__version__,
                    elapsed_s=time.perf_counter() - started, complete=True)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# Utonia ScanNet-20 segmentation of building2_9\n\n"
        "Official full Utonia encoder plus its pretrained ScanNet-20 linear probe. "
        "Tiny has no released compatible ScanNet-20 head. Original XYZ/normals are used, RGB is zero. "
        "One full-scene encoder pass; 2 cm input-coordinate voxel size. Row-independent rotary position "
        "encoding is computed in 65,536-row batches to bound temporary memory; attention is unchanged. "
        "All original points/fields/order "
        "are preserved. No input random subsampling or spatial tiles.\n\n"
        "- `seg_colored.pcd`: original fields plus packed RGB, label (0-19), nyu40_id, confidence.\n"
        "- `seg_colored.ply`: original XYZ with RGB, label and confidence.\n"
        "- `labels.npy`: uint8 [2,000,000], in original input order.\n"
        "- `confidence.npy`: float32 max softmax probability in original input order.\n"
        "- `logits_grid.npy`: float32 [model voxels, 20]; `inverse.npy`: original point -> voxel row.\n"
        "- `classes.json` / `class_counts.csv`: class IDs, names, colors and counts.\n"
        "- `preview.png`: semantic views plus lower-half confidence view.\n"
        "- `viewer.html`: offline 3D viewer with class legend toggles and height clipping.\n\n"
        "The ScanNet-20 head has no stair, railing or person categories. Every point is assigned "
        "one of its 20 classes, including unfamiliar structures. No scene labels are available "
        "for accuracy evaluation. Max softmax is a model score, not validated correctness.\n\n"
        "```bash\n/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m utils.segment_utonia "
        f"--pcd {args.pcd.resolve()} --output /path/to/new/output --device cuda\n```\n",
        encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
