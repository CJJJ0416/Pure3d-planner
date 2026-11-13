"""Staged data preparation and explicit two-stage RECON training."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from utils.pointcloud import read_json,workspace_path
from Reconstruction.data.storage import json_save


def run_training(config_file,device=None):
    config=read_json(workspace_path(config_file))
    if device is not None:config['device']=device
    root=workspace_path(config['output']);root.mkdir(parents=True,exist_ok=True)
    resolved=root/'training.resolved.json';json_save(resolved,config)
    status_file=root/'status.json'
    def status(phase,**kw):json_save(status_file,dict(pid=os.getpid(),phase=phase,**kw))
    def child(args,log):
        with (root/log).open('a') as stream:
            subprocess.run([sys.executable,'-m',*args],cwd=workspace_path('.'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        if not (workspace_path(config['data'])/'manifest.json').exists():
            status('preparing_stage_1')
            child(['Reconstruction.data.stages','initialize','--config',config['dataset_config'],'--device',config['device']],'prepare.log')
        for stage in ['ae','flow']:
            if stage=='flow':
                import torch
                ckpt=torch.load(root/'ae'/'best.pt',map_location='cpu',weights_only=False)
                validation=next(r['validation'] for r in reversed(ckpt['history']) if 'validation' in r)
                if not all(validation.get(k,0)>=v for k,v in config['ae_gate'].items()):
                    status('ae_validation_gate_not_met',validation=validation,training_stage='ae')
                    print(f'Stage A complete. Stage B withheld: validation gate {validation}',flush=True)
                    return
            last=root/stage/'last.pt'
            for until in [5,config['epochs']]:
                args=['Reconstruction.training.train','--config',str(resolved),'--stage',stage,'--epochs',str(until)]
                if last.exists():
                    import torch
                    previous=torch.load(last,map_location='cpu',weights_only=False)
                    if previous['epoch']>=until and previous['next_index']==0:continue
                    args+=['--resume',str(last)]
                    del previous
                elif stage=='flow':args+=['--ae-checkpoint',str(root/'ae'/'best.pt')]
                status('training',training_stage=stage,until_epoch=until)
                child(args,f'{stage}.log')
            status('evaluating_test',training_stage=stage)
            child(['Reconstruction.training.evaluate','--checkpoint',str(root/stage/'best.pt'),
                   '--split','test','--output',str(root/stage/'test'),
                   '--steps',*(['4','8','16'] if stage=='flow' else ['16'])],f'{stage}_test.log')
        status('complete')
    except Exception as exc:
        status('failed',error=str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['initialize','prepare-stage','train']:
        s=sub.add_parser(name);s.add_argument('--config',default='Reconstruction/configs/training.json' if name=='train' else 'Reconstruction/configs/dataset.json')
        s.add_argument('--device',default='cuda:0')
        if name=='prepare-stage':s.add_argument('--stage-id',type=int,required=True,choices=[1,2,3,4])
    a=p.parse_args()
    if a.command=='train':run_training(a.config,a.device)
    else:
        from Reconstruction.data.stages import initialize,prepare_stage
        cfg=read_json(workspace_path(a.config))
        if a.command=='initialize':initialize(cfg,a.device)
        else:prepare_stage(cfg,a.stage_id,a.device)

if __name__=='__main__':main()
