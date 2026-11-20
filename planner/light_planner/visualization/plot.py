"""从正式运行目录绘制DC与唯一消融的路径，不重新规划。"""
import json
import os
from pathlib import Path

import numpy as np

from common.pointcloud_io import fields_xyz,read_pcd_fields


def plot_run(output):
    output=Path(output);summary=json.loads((output/'summary.json').read_text())
    os.environ.setdefault('MPLCONFIGDIR',str(output/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cloud=read_pcd_fields(output/'map/voxel_cost.pcd');xyz=fields_xyz(cloud)
    sample=np.arange(0,len(xyz),max(1,len(xyz)//50000))
    fig=plt.figure(figsize=(15,7),layout='constrained')
    for i,stage in enumerate(['astar','gpmp'],1):
        ax=fig.add_subplot(1,2,i,projection='3d')
        ax.scatter(*xyz[sample].T,c=cloud['trav_cost'][sample],s=.3,cmap='RdYlGn_r',vmin=0,vmax=1,alpha=.15)
        for method,record in summary['variants'].items():
            if not record['success']:continue
            a=np.genfromtxt(output/method/('astar_path.csv' if stage=='astar' else 'gpmp_trajectory.csv'),delimiter=',',names=True)
            points=np.column_stack([a[k] for k in ['x','y','z']]);color='#c323e8' if method=='with_step' else '#4d32a8'
            ax.plot(*points.T,color=color,lw=2,ls='-' if method=='with_step' else '--',
                label=f'{method} {record[stage]["length_m"]:.3f}m')
        ax.set(title=stage.upper(),xlabel='Map X(m)',ylabel='Map Y(m)',zlabel='Map Z(m)')
        ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),1.));ax.view_init(25,-60);ax.legend(fontsize=9)
    fig.suptitle('PCD → DC → A* → gravity GPMP | same map, cost and endpoints')
    fig.savefig(output/'overview.png',dpi=150);plt.close(fig)
