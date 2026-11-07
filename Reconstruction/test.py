#!/usr/bin/env python3
"""Utonia-tiny 入门：默认使用模拟点云，可选读取一部分 PCD。

运行示例：
    conda activate pure3d-ptv3
    python /home/cjh/Pure3D_ws/Reconstruction/test.py

如需对自己的 PCD 试运行，显式指定文件：
    python /home/cjh/Pure3D_ws/Reconstruction/test.py \
        --pcd /home/cjh/Pure3D_ws/datasets/rot_stairs/raw.pcd

这是特征提取示例。预训练 Utonia-tiny 默认没有语义分割或 trav_cost 预测头。
"""

import argparse
from pathlib import Path

import numpy as np
import point_transformer_v3 as ptv3


def read_binary_pcd_sample(path: Path, max_points: int):
    """从当前项目的 binary PCD 中均匀选取一部分点，不读取全部点到内存。"""
    header = {}
    with path.open("rb") as file:
        while True:
            raw_line = file.readline()
            if not raw_line:
                raise ValueError("PCD 文件缺少 DATA 标记")
            line = raw_line.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                offset = file.tell()
                break

    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    if header["DATA"] != ["binary"] or any(s != 4 for s in sizes) or any(t != "F" for t in types) or any(c != 1 for c in counts):
        raise ValueError("此教学示例只支持每个字段为单个 float32 的 binary PCD")

    num_points = int(header["POINTS"][0])
    data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(num_points, len(fields)))
    indices = np.linspace(0, num_points - 1, min(max_points, num_points), dtype=np.int64)
    sample = np.asarray(data[indices], dtype=np.float32)
    coord = sample[:, [fields.index(name) for name in ("x", "y", "z")]]

    normal_names = ("normal_x", "normal_y", "normal_z")
    normal = sample[:, [fields.index(name) for name in normal_names]] if all(name in fields for name in normal_names) else np.zeros_like(coord)
    valid = np.isfinite(coord).all(axis=1)
    coord, normal = coord[valid], normal[valid]
    normal[~np.isfinite(normal)] = 0
    lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, lengths, out=np.zeros_like(normal), where=lengths > 1e-8)
    return coord, normal, num_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, help="可选：读取 binary PCD，不指定时使用模拟点云")
    parser.add_argument("--points", type=int, default=128, help="用于本次示例的点数，默认 128")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--weights", type=Path, help="可选：手动指定 utonia_tiny.pth")
    args = parser.parse_args()
    if args.points < 1:
        parser.error("--points 必须大于 0")

    # 1. 准备点云：coord 是 N×3，单位为米。默认数据仅用于学习接口。
    if args.pcd:
        coord, normal, total = read_binary_pcd_sample(args.pcd, args.points)
        print(f"输入文件: {args.pcd}，总点数: {total}，本次选取: {len(coord)}")
    else:
        coord = np.random.default_rng(7).random((args.points, 3), dtype=np.float32)
        normal = np.zeros_like(coord)
        print(f"输入: 模拟点云，本次点数: {len(coord)}")
    if len(coord) == 0:
        raise ValueError("没有可用的有限坐标点")

    # 2. Utonia 支持缺失颜色时输入零值，不把 LiDAR 强度当作 RGB。
    color = np.zeros_like(coord)

    # 3. 加载官方预训练权重。weights=None 时自动使用或下载 HF 缓存。
    model = ptv3.load("utonia_tiny", device=args.device, warmup=False, weights=args.weights)
    weight_info = model.weights_info["backbone"]
    print("模型设备:", next(model.parameters()).device)
    print("权重位置:", weight_info.get("path", weight_info.get("file")))

    # 4. embed 是推理接口：每个输入点得到一个 486 维特征。
    features = model.embed(coord, color, normal)
    print("坐标 shape:", coord.shape)
    print("特征 shape:", tuple(features.shape))
    print("第一个点的前 8 维特征:", features[0, :8].cpu().numpy())
    print("注意：特征不是语义类别，也不是 trav_cost。")


if __name__ == "__main__":
    main()
