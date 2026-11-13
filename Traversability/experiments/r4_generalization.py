"""Read-only model evaluation outside the fitted R4 labels, plus four held-out seeds."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time

from utils.pointcloud import ROOT,read_json,workspace_path
from Traversability.experiments.r4_controlled_data import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/traversability/r4_controlled_fit_v5/generalization')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    neural='/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python';geometry='/home/cjh/glocal2-nav/.venv/bin/python'
    before=ROOT/'runs/traversability/cost_parameter_decoupling_v4/best.pt'
    after=ROOT/'runs/traversability/r4_controlled_fit_v5/best.pt'
    meta=read_json(ROOT/'datasets/wfc_parameter_decoupling_v4/evaluation/blocks.json')
    chosen=[next(s for s in meta['scenes'] if s['source']=='new' and s['split']=='test' and s['kind']==kind)
            for kind in ['straight','switchback','spiral','ramp']]
    selected={s['id'] for s in chosen}
    subset=copy.deepcopy(meta);subset['scenes']=chosen;subset['blocks']=[b for b in meta['blocks'] if b['scene'] in selected]
    write_json(out/'independent_data/blocks.json',subset)
    config=read_json(ROOT/'runs/traversability/cost_parameter_decoupling_v4/evaluation_config.json')
    config.update(data=str(out/'independent_data'),output=str(out),num_workers=2)
    write_json(out/'evaluation_config.json',config)
    protocol=dict(initial_checkpoint=str(before),after_checkpoint=str(after),before_sha256=sha(before),after_sha256=sha(after),
        condition=[.5,.6,.4,.05],independent_scenes=[dict(id=s['id'],kind=s['kind'],pcd=s['input_pcd'],geometry=s['geometry_scene']) for s in chosen],
        same_map='Other stair labels were not fitted, but full building geometry was input context during R4 fitting.',
        independent_scope='Four held-out seeds from the existing v4 test split, never used to optimize R4. Same generator distribution, not arbitrary-building generalization.',
        training=False,astar=False,gpmp=False)
    write_json(out/'protocol.json',protocol)
    os.environ.update(PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',MPLCONFIGDIR=str(out/'matplotlib'))
    began=time.perf_counter()

    def execute(name,python,module,arguments):
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.Popen([python,'-u','-m',module,*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                write_json(out/'progress.json',dict(stage=name,child_pid=process.pid,complete=False,elapsed_s=time.perf_counter()-began))
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,module)
        print('COMPLETE',name,flush=True)

    for label,checkpoint in [('initial',before),('best',after)]:
        if not (out/(label+'_regression.npz')).exists():
            execute(label+'_regression',neural,'Traversability.training.evaluate',
                ['--checkpoint',checkpoint,'--data',ROOT/'datasets/wfc_regression_20261127','--split','test',
                 '--output',out/(label+'_regression.json'),'--predictions',out/(label+'_regression.npz')])
        if not (out/(label+'_test.npz')).exists():
            execute(label+'_independent',neural,'Traversability.experiments.decoupling_eval',
                ['--checkpoint',checkpoint,'--config',out/'evaluation_config.json','--split','test',
                 '--output',out/(label+'_test.json'),'--predictions',out/(label+'_test.npz')])
    if not (out/'best.pt').exists():(out/'best.pt').symlink_to(after)
    execute('all_stairs_audit',geometry,'Traversability.experiments.all_stair_audit',['--run',out])
    for s in chosen:
        target=out/'predictions'/s['kind']
        if not (target/'prediction.json').exists():
            execute('PCD_'+s['kind'],neural,'Traversability.inference.predict',
                ['--checkpoint',after,'--pcd',s['input_pcd'],'--geometry-python',geometry,
                 '--condition',.5,.6,.4,.05,'--output',target])
    execute('report',geometry,'Traversability.experiments.r4_generalization_report',['--run',out])
    summary=read_json(out/'summary.json')
    write_json(out/'verification.json',dict(before_weights_unchanged=sha(before)==protocol['before_sha256'],
        after_weights_unchanged=sha(after)==protocol['after_sha256'],same_XYZ=summary['same_XYZ'],same_GT=summary['same_GT']))
    write_json(out/'progress.json',dict(complete=True,stage='complete',elapsed_s=time.perf_counter()-began,report=str(out/'REPORT.md')))
    print('ALL_COMPLETE',out,flush=True)


if __name__=='__main__':main()
