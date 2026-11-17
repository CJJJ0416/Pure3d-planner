"""Point metrics and support-part groups used by controlled training losses."""
import numpy as np

TREAD_PARTS=[1,2,3,4,6,7,8,9]


LANDING_PARTS=[0,5,10]


def point_metrics(gt,pred):
    error=pred-gt;result=dict(points=len(gt),MAE=float(abs(error).mean()),MAE_raw50=float(abs(error).mean()*50),
                           bias=float(error.mean()),bias_raw50=float(error.mean()*50),thresholds={})
    for threshold in [.3,.5,.6]:
        low=gt<=threshold;high=~low;blocked=low&(pred>threshold);passed=high&(pred<=threshold)
        result['thresholds'][str(threshold)]=dict(false_block_count=int(blocked.sum()),GT_low_count=int(low.sum()),
            false_block_pct=float(100*blocked.sum()/low.sum()) if low.any() else None,
            false_pass_count=int(passed.sum()),GT_high_count=int(high.sum()),
            false_pass_pct=float(100*passed.sum()/high.sum()) if high.any() else None)
    return result
