"""PCD comparisons in a shared map frame, without simulator dependencies."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def comparison(path,clouds,title,legend=None):
    fig=plt.figure(figsize=(5*len(clouds),5))
    all_xyz=np.concatenate([xyz for _,xyz,_ in clouds if len(xyz)])
    lo,hi=all_xyz.min(0),all_xyz.max(0)
    for i,(name,xyz,color) in enumerate(clouds):
        ax=fig.add_subplot(1,len(clouds),i+1,projection='3d')
        ids=np.random.default_rng(0).choice(len(xyz),min(len(xyz),45000),replace=False)
        if isinstance(color,dict):
            palette=np.array([[.27,.40,.87],[.12,.65,.43]])
            color=palette[(ids>=color['observed_count']).astype(int)]
        elif isinstance(color,np.ndarray):
            color=color[ids]
        if len(ids):ax.scatter(*xyz[ids].T,s=.35,c=color,alpha=.65,rasterized=True)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='Map X (m)',ylabel='Map Y (m)',zlabel='Z (m)')
        ax.set_box_aspect(np.maximum(hi-lo,.1));ax.view_init(28,-65)
        ax.set_title(f'{name}\n{len(xyz):,} points')
    fig.suptitle(title)
    if legend:
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color=color,label=name) for name,color in legend],
                   loc='lower center',ncol=len(legend))
    fig.tight_layout(rect=(0,.07 if legend else 0,1,.96));Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150);plt.close(fig)


def curves(history,path):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epoch=[h['epoch'] for h in history]
    axes[0].plot(epoch,[h['loss'] for h in history]);axes[0].set(xlabel='Epoch',ylabel='Train loss')
    rows=[h for h in history if 'validation' in h]
    axes[1].plot([h['epoch'] for h in rows],[h['validation']['f1_2cm'] for h in rows],label='F1 @ 2 cm')
    axes[1].plot([h['epoch'] for h in rows],[h['validation']['recall_2cm'] for h in rows],label='Recall @ 2 cm')
    axes[1].set(xlabel='Epoch',ylim=(0,1));axes[1].legend()
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
