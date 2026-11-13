"""Evaluation-only VST comparison on one GT-defined, fixed native grid."""
import numpy as np
from common.vst_global_planner.src.vst_global_planner.conditioned_teacher import ConditionedTeacher


def compare_vst(complete, partial, repaired, core_bounds=None):
    gt = ConditionedTeacher(complete)
    grid = gt.metadata()
    condition = [gt.trav[k] for k in ['safe_margin','slope_max','step_max','interval_free','standable_ratio']]
    target = gt.cost(condition) / gt.cost_scale
    gt_known = np.isfinite(gt.ground)
    if core_bounds is not None:
        lo,hi=map(np.asarray,core_bounds)
        ix,iy=np.indices(gt.shape_xy)
        wx=(ix-gt.shape_xy[0]//2)*gt.resolution+gt.center[0]
        wy=(iy-gt.shape_xy[1]//2)*gt.resolution+gt.center[1]
        gt_known &= ((wx>=lo[0])&(wx<hi[0])&(wy>=lo[1])&(wy<hi[1]))[None]
        gt_known &= (gt.ground>=lo[2])&(gt.ground<hi[2])
    dangerous = gt_known & (target >= .8)
    report = dict(core_bounds=core_bounds,grid=grid, condition=condition, dangerous_threshold=.8, low_cost_threshold=.5,
                  gt_valid_ground_cells=int(gt_known.sum()), unknown_policy='excluded from cost errors, separately counted')
    outputs={}
    for name, xyz in [('input', partial), ('repaired', repaired)]:
        teacher = ConditionedTeacher(xyz, grid=grid)
        cost = teacher.cost(condition) / teacher.cost_scale
        known = np.isfinite(teacher.ground)
        outputs[name]=(known,cost)
        overlap = gt_known & known
        danger_known = dangerous & known
        report[name] = dict(
            valid_ground_coverage=float(overlap.sum()/gt_known.sum()),
            aligned_ground_coverage_10cm=float((overlap & (np.abs(teacher.ground-gt.ground)<=.1)).sum()/gt_known.sum()),
            normalized_cost_mae=float(np.abs(cost[overlap]-target[overlap]).mean()) if overlap.any() else None,
            dangerous_low_cost_fraction=float((danger_known & (cost<.5)).sum()/dangerous.sum()) if dangerous.any() else None,
            dangerous_known_low_cost_fraction=float((cost[danger_known]<.5).mean()) if danger_known.any() else None,
            unknown_gt_ground_cells=int((gt_known & ~known).sum()),
            unknown_dangerous_cells=int((dangerous & ~known).sum()),
            extra_ground_in_gt_unknown_cells=int((known & ~np.isfinite(gt.ground)).sum()),
            cost_compared_cells=int(overlap.sum()))
    common=gt_known & outputs['input'][0] & outputs['repaired'][0]
    report['common_compared_cells']=int(common.sum())
    for name,(_,cost) in outputs.items():
        report[name]['common_support_cost_mae']=float(np.abs(cost[common]-target[common]).mean()) if common.any() else None
    return report
