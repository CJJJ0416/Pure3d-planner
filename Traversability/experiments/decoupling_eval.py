"""Two-bank frozen evaluation for the parameter-decoupling experiment."""
import argparse
from pathlib import Path
import torch

from Traversability.models.model import ConditionedCost
from Traversability.training.decoupling import evaluate_decoupling
from utils.pointcloud import read_json,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--split',default='val')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--predictions',type=Path)
    args=p.parse_args();config=read_json(args.config)
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=ConditionedCost(config['model'],cp['condition_scale'],pretrained=False).to(config['device'])
    model.load_state_dict(cp['model'])
    metrics=evaluate_decoupling(model,config,args.split,args.predictions)
    write_json(args.output,dict(checkpoint=str(args.checkpoint),split=args.split,**metrics))
    print('EVALUATION_COMPLETE',args.split,metrics['selection_score'],flush=True)


if __name__=='__main__':main()
