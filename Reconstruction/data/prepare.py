"""Index shared global masks and prepare fixed-grid Mesh-supervised blocks."""
import argparse
from pathlib import Path
import numpy as np
from utils.pointcloud import read_json,write_json,workspace_path


def input_indices(row):
    root=workspace_path(row['terrain_folder'])
    complete=np.load(workspace_path(row['complete_indices']),mmap_mode='r')
    if row['mask_id']=='complete':return np.array(complete)
    mask=workspace_path(row['mask_file']) if 'mask_file' in row else root/row['mask_id']/'indices.npz'
    with np.load(mask) as data:removed=data['removed_indices']
    return complete[~np.isin(complete,removed)]


def openings_for_scene(scene):
    """Explicit doorway empty volumes. Mesh proximity removes their borders."""
    result=[];transform=np.asarray(scene['transform'])
    for corridor in scene.get('corridors',[]):
        frame=transform@np.asarray(corridor['local_to_scene'])
        x=(corridor['width_m']+corridor['wall_thickness_m'])/2
        half=corridor['wall_thickness_m']/2
        y=corridor['doorway_offset_m'];w=corridor['door_width_m']/2
        for side in [-1,1]:
            result.append(dict(name='door',lower=[side*x-half,y-w,0.],upper=[side*x+half,y+w,corridor['door_height_m']],frame=frame.tolist()))
    return result


def main():
    from Reconstruction.data.stages import initialize,prepare_stage
    p=argparse.ArgumentParser(description='Prepare the current stage, never an all-stage mask bank.')
    p.add_argument('--config',default='Reconstruction/configs/dataset.json')
    p.add_argument('--stage-id',type=int,choices=[1,2,3,4])
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();cfg=read_json(workspace_path(a.config))
    if a.stage_id is None:initialize(cfg,a.device)
    else:prepare_stage(cfg,a.stage_id,a.device)

if __name__=='__main__':main()
