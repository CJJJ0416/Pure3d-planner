"""All active corrupted blocks, plus 20 percent fixed complete controls."""
import numpy as np
from torch.utils.data import Sampler


def band_for_epoch(config,epoch):
    return next(b for b in config['stages'] if b['epochs'][0]<=epoch<=b['epochs'][1])


def difficulty(config,ratio):
    if ratio==0:return 'complete'
    for name,(lower,upper) in config['validation_bins'].items():
        if lower<=ratio<=upper:return name
    return 'outside_validation'


def pools(rows,config,epoch):
    return ([i for i,r in enumerate(rows) if r['mask_id']!='complete'],
            [i for i,r in enumerate(rows) if r['mask_id']=='complete'])


def coverage_report(rows,config):
    main,easier=pools(rows,config,1)
    return dict(main_samples=len(main),complete_controls=len(easier),all_blocks_participate=True)


class CurriculumSampler(Sampler):
    def __init__(self,rows,config,seed=42):
        self.rows=rows;self.config=config;self.seed=seed;self.epoch=1
        self.main,self.easier=pools(rows,config,1)
        self.replay=round(len(self.main)*config['easier_fraction']/(1-config['easier_fraction']))
        self.count=len(self.main)+self.replay

    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.count

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        ids=np.r_[self.main,rng.choice(self.easier,self.replay,replace=self.replay>len(self.easier))]
        rng.shuffle(ids)
        return iter(ids.tolist())
