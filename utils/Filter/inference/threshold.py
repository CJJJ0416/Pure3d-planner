"""Explicitly select points from an existing keep_score PCD, without running a model."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from common.pointcloud_io import fields_xyz, read_pcd_fields, write_pcd
from utils.pointcloud import workspace_path, write_json


def select_points(prediction, threshold, output):
    started = perf_counter()
    fields = read_pcd_fields(prediction)
    xyz, score = fields_xyz(fields), fields['keep_score']
    keep = score >= threshold
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_pcd(output/'filtered.pcd', xyz[keep])
    write_pcd(output/'removed.pcd', xyz[~keep], score[~keep], field_name='keep_score')
    np.savez(output/'filter_mapping.npz', point_index=np.flatnonzero(keep), keep_score=score[keep])
    report = dict(source=str(prediction), threshold=threshold, keep_rule='keep_score >= threshold',
                  input_points=len(xyz), retained_points=int(keep.sum()), removed_points=int((~keep).sum()),
                  retained_pct=100*float(keep.mean()), coordinate_frame='map',
                  original_order_preserved=True, input_downsampled=False,
                  selection_s=perf_counter()-started)
    write_json(output/'selection.json', report)
    print(report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True, help='XYZ + keep_score PCD')
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    select_points(workspace_path(args.prediction), args.threshold, workspace_path(args.output))


if __name__ == '__main__':
    main()
