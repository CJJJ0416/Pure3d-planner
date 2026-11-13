"""Select threshold on validation only, then evaluate fixed held-out suites."""
import argparse
from pathlib import Path
import torch
from Filter.models.filter import FilterHead,geometry_mode
from Filter.data.dataset import CachedCores
from Filter.training.train import evaluate
from Filter.training.metrics import Metrics
from utils.pointcloud import write_json,workspace_path


def load_head(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    c=ckpt['config'];torch.set_num_threads(c['cpu_threads'])
    model=FilterHead(ckpt['geometry_mean'],ckpt['geometry_std'],c['dropout'],geometry_mode=geometry_mode(c)).to(device)
    model.load_state_dict(ckpt['model']);model.eval()
    return model,ckpt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--score-only',action='store_true',help='Continuous metrics and full point scores, no threshold selection')
    a=p.parse_args();device=torch.device('cuda:0');model,ckpt=load_head(workspace_path(a.checkpoint),device)
    c=ckpt['config'];out=workspace_path(a.output or Path(c['output'])/'evaluations');out.mkdir(parents=True,exist_ok=True)
    val=CachedCores(workspace_path(c['cache']),'val',geometry_mode=geometry_mode(c))
    report=evaluate(model,val,c,device,() if a.score_only else c['threshold_candidates'],out/'validation_predictions')
    if a.score_only:
        tau=None;write_json(out/'validation.json',report)
    else:
        selected=min(report['decisions'],key=lambda d:(-d['macro_f1'],d['static_false_deletion_rate'],abs(d['threshold']-.5)))
        tau=selected['threshold'];write_json(out/'threshold.json',dict(threshold=tau,gt_threshold=.5,selection_split='val',validation=report))
    reports={}
    test_suites=sorted({r['suite'] for r in val.metadata['records'] if r['split']=='test'})
    for suite in test_suites:
        ds=CachedCores(workspace_path(c['cache']),'test',suite,geometry_mode=geometry_mode(c))
        metric=evaluate(model,ds,c,device,() if a.score_only else [tau],out/f'{suite}_predictions')
        baseline=Metrics(() if a.score_only else [tau],device)
        # The reference uses the identical fixed labels, without neural inputs.
        from Filter.training.train import loader,move
        for batch in loader(ds,c):
            b=move(batch,device);baseline.update(torch.ones_like(b['target']),b['target'],b['group'],b['semantic'])
        metric['keep_all_baseline']=baseline.report();reports[suite]=metric
        write_json(out/f'test_{suite}.json',metric)
        print(f"test {suite}: macro_mae={metric['macro_mae']:.5f} points={metric['points']}",flush=True)
    write_json(out/'summary.json',dict(checkpoint=str(workspace_path(a.checkpoint)),threshold=tau,gt_threshold=.5,
        score_only=a.score_only,tests=reports))

if __name__=='__main__': main()
