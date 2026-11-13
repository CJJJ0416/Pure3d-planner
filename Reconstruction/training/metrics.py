"""Measure missing surface recovery and false surface generation against Mesh."""
import numpy as np
from scipy.spatial import cKDTree


def nearest(reference,points):
    if not len(points):return np.full(len(reference),np.inf)
    return cKDTree(points).query(reference,workers=4)[0]


def geometry_metrics(reference,generated,kept,input_xyz,query,scene,leaf=.015,semantic=None):
    observed=nearest(reference,input_xyz)<=.03
    distance_to_kept=nearest(reference,kept)
    retained=distance_to_kept<=.03
    generated_distance=query.distances(generated)
    output=np.concatenate([kept,generated])
    distance=nearest(reference,output)
    def rate(mask):return float((distance[mask]<=.02).mean()) if mask.any() else None
    precision=float((generated_distance<=.02).mean()) if len(generated) else 0.
    missing=~retained
    recall=rate(missing)
    r=recall if recall is not None else rate(np.ones(len(reference),bool))
    f1=2*precision*r/(precision+r) if precision+r else 0.
    openings={}
    frame=np.asarray(scene['frame']);rotation=frame[:3,:3];translation=frame[:3,3]
    local=(generated-translation)@rotation
    from Reconstruction.data.geometry import grid_between
    for name in ['door','stairwell','railing_gap']:
        filled=0;total=0
        for opening in [v for v in scene['openings'] if v['name']==name]:
            opening_frame=np.asarray(opening.get('frame',frame))
            rotation=opening_frame[:3,:3];translation=opening_frame[:3,3]
            local=(generated-translation)@rotation
            lo=np.asarray(opening['lower']);hi=np.asarray(opening['upper'])
            if np.any(hi<=lo):continue
            grid=grid_between(lo,hi,leaf);centers=(grid+.5)*leaf
            world=centers@rotation.T+translation
            valid=(query.distances(world)>.02)&((world>=scene['core_lower'])&(world<scene['core_upper'])).all(1)
            ids={tuple(c) for c in grid[valid]};total+=len(ids)
            mask=((local>=lo)&(local<hi)).all(1)
            occupied={tuple(c) for c in np.floor(local[mask]/leaf).astype(np.int64)}
            filled+=len(ids&occupied)
        openings[name]=dict(filled_cells=filled,empty_cells=total,fill_rate=filled/total if total else None)
    per_structure={}
    if semantic is not None:
        from common.geneator.filter_components import SEMANTICS
        for name,value in SEMANTICS.items():
            mask=semantic==value
            if mask.any():per_structure[name]=dict(complete_recall_2cm=rate(mask),missing_recall_2cm=rate(mask&missing),gt_points=int(mask.sum()))
    return dict(f1_2cm=f1,precision_2cm=precision,recall_2cm=r,per_structure=per_structure,
        precision_3cm=float((generated_distance<=.03).mean()) if len(generated) else 0.,
        complete_recall_3cm=float((distance<=.03).mean()),
        missing_recall_2cm=recall,
        missing_input_recall_2cm=float((distance_to_kept[missing]<=.02).mean()) if missing.any() else None,
        input_complete_recall_2cm=float((distance_to_kept<=.02).mean()),
        complete_recall_gain=float((distance<=.02).mean()-(distance_to_kept<=.02).mean()),
        subtraction_missing_recall_2cm=rate(~observed),filter_deleted_recall_2cm=rate(observed&~retained),
        complete_recall_2cm=rate(np.ones(len(reference),bool)),
        subtraction_missing_gt_points=int((~observed).sum()),filter_deleted_gt_points=int((observed&~retained).sum()),
        generated_points=len(generated),retained_points=len(kept),
        generated_distance_mean_m=float(generated_distance.mean()) if len(generated) else None,
        generated_distance_p95_m=float(np.percentile(generated_distance,95)) if len(generated) else None,openings=openings)


def summarize(rows):
    names=['precision_3cm','complete_recall_3cm','missing_recall_2cm','missing_input_recall_2cm','f1_2cm','precision_2cm','recall_2cm','complete_recall_2cm','input_complete_recall_2cm','complete_recall_gain','subtraction_missing_recall_2cm','filter_deleted_recall_2cm',
           'generated_distance_mean_m','generated_distance_p95_m']
    result={}
    for name in names:
        values=[r[name] for r in rows if r[name] is not None]
        result[name]=float(np.mean(values)) if values else None
    result['samples']=len(rows)
    result['openings']={}
    for name in ['door','stairwell','railing_gap']:
        filled=sum(r['openings'][name]['filled_cells'] for r in rows)
        empty=sum(r['openings'][name]['empty_cells'] for r in rows)
        result['openings'][name]=dict(filled_cells=filled,empty_cells=empty,fill_rate=filled/empty if empty else None)
    if rows and 'vst' in rows[0]:
        result['vst']={}
        for mode in ['input','repaired']:
            result['vst'][mode]={}
            for key in rows[0]['vst'][mode]:
                values=[r['vst'][mode][key] for r in rows if r['vst'][mode][key] is not None]
                result['vst'][mode][key]=float(np.mean(values)) if values else None
        a,b=result['vst']['input'],result['vst']['repaired']
        result['success_checks']=dict(
            missing_coverage_improves=result['missing_recall_2cm']>result['missing_input_recall_2cm'] if result['missing_recall_2cm'] is not None else None,
            ground_coverage_improves=b['valid_ground_coverage']>a['valid_ground_coverage'],
            cost_mae_decreases=b['common_support_cost_mae']<a['common_support_cost_mae'] if a['common_support_cost_mae'] is not None else None,
            dangerous_low_cost_not_increased=b['dangerous_low_cost_fraction']<=a['dangerous_low_cost_fraction'] if a['dangerous_low_cost_fraction'] is not None else None)

    return result
