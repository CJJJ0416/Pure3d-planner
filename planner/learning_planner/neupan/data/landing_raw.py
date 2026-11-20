# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.data.landing_raw --help
"""Export pre-inflation VST costs from the native kernel on a saved grid."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from common.pointcloud_io import read_pcd_xyz
from planner.learning_planner.neupan.data.expert import ROOT, fingerprint
from planner.learning_planner.neupan.geometry.tomogram import Tomogram


def export_raw(pcd, grid, output, config=None):
    vst = ROOT / 'third_party_lib/vst_global_planner'
    sys.path.insert(0, str(vst / 'src'))
    from vst_global_planner.conditioned_teacher import ConditionedTeacher, travKernel
    import cupy as cp

    with grid.open('rb') as stream:
        payload = pickle.load(stream)
    points = read_pcd_xyz(pcd)
    reference = Tomogram.read(grid)
    source_parameters = (json.loads(config.read_text()) if config else payload['teacher_parameters'])
    parameters = source_parameters.get('parameters', source_parameters)
    trav = parameters['trav']
    shape = list(payload['data'].shape[1:])
    ground_h = parameters['map']['ground_h']
    if config:
        # Native snapshots may have dropped redundant layers. Rebuild the full
        # layer stack first, then recover each saved layer by its ground grid.
        shape[0] = max(shape[0], int(np.ceil((points[:, 2].max() - ground_h) / payload['slice_dh'])))
    teacher = ConditionedTeacher(
        points, resolution=payload['resolution'], slice_dh=payload['slice_dh'], ground_h=ground_h,
        grid=dict(center=payload['center'], shape=shape, slice_h0=payload['slice_h0']))
    half = int(trav['kernel_size'] / 2)
    standable = int(trav['standable_ratio'] * (2 * half + 1) ** 2) - 1
    nx, ny = map(int, teacher.shape_xy)
    raw = cp.zeros_like(teacher.grad_sq)
    travKernel(nx, ny, half, trav['interval_min'], trav['interval_free'], trav['step_max'],
               1.2 * teacher.resolution * np.tan(trav['slope_max']), standable, trav['cost_barrier'])(
                   teacher.interval, teacher.grad_sq, teacher.grad_max, raw, size=raw.size)
    raw[~teacher.valid_ground] = trav['cost_barrier']

    ground, ceiling, cost = teacher.ground, teacher.ceiling, raw.get()
    retained = list(range(len(ground)))
    if config:
        native_ground = ground.astype(np.float16)
        retained = []
        for saved in reference.ground.astype(np.float16):
            start = retained[-1] + 1 if retained else 0
            equivalent = next((i for i in range(start, len(ground))
                               if np.array_equal(native_ground[i], saved, equal_nan=True)), None)
            if equivalent is None:
                raise ValueError('Could not align a retained layer in the observed snapshot')
            retained.append(equivalent)
        ground, ceiling, cost = ground[retained], ceiling[retained], cost[retained]
    match = np.isfinite(ground) & np.isfinite(reference.ground)
    difference = np.abs(ground[match] - reference.ground[match])
    missing_delta = int(np.count_nonzero(np.isfinite(ground) != np.isfinite(reference.ground)))
    max_delta = float(difference.max()) if len(difference) else None
    report = dict(source_pcd=str(pcd.resolve()), source_sha256=fingerprint(pcd), grid=str(grid.resolve()),
                  grid_sha256=fingerprint(grid), grid_matches=missing_delta == 0 and max_delta is not None
                  and max_delta <= .005, ground_mask_differences=missing_delta, max_z_difference_m=max_delta,
                  source_scope='observed_snapshot' if config else 'clean_gt', parameters=trav, retained_layers=retained)
    if not config and not report['grid_matches']:
        raise ValueError(f'Clean map does not match the archived grid: {report}')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ground=ground, cost=cost, ceiling=ceiling,
                        resolution=reference.resolution, origin=reference.origin)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcd', type=Path, required=True)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Override parameters for an observed snapshot audit')
    args = parser.parse_args()
    print(json.dumps(export_raw(args.pcd, args.grid, args.output, args.config)))


if __name__ == '__main__':
    main()
