"""Fixed stress suites of the held-out building, generated before training."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import trimesh

from Filter.data.prepare import prepare_variants
from Filter.visualization.visualize import render_scene
from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import read_json,write_json,workspace_path


def generate(config):
    data=workspace_path(config['dataset_root'])
    for entry in read_json(data/'manifest.json')['scenes']:
        if entry['split']!='test': continue
        base=data/entry['scene_id']
        clean=read_pcd_xyz(base/'clean.pcd');attrs=dict(np.load(base/'clean_attributes.npz'))
        mesh=trimesh.load(base/'clean.obj',force='mesh',process=False)
        for spec in config['test_suites'][1:]:
            folder=base/'suites'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
            for name in ['scene.obj','clean.obj','terrain.obj','static_objects.obj','face_metadata.npz',
                         'clean.pcd','clean_attributes.npz','roles.json','wfc_config.resolved.json','scene.json']:
                target=folder/name
                if not target.exists(): target.symlink_to(Path('../..')/name)
            cfg=deepcopy(config);aug=cfg['augmentations'];aug['suite_seed']=spec['seed']
            for key in ['retention_probability_range','hole_size_m_range']:
                if key in spec: aug[key]=spec[key]
            if 'positions_range' in spec: aug['dynamic_trails']['positions_range']=spec['positions_range']
            records,build=prepare_variants(folder,entry['geometry_seed'],4,clean,attrs,cfg,mesh,suite_seed=spec['seed'])
            write_json(folder/'config.resolved.json',cfg)
            write_json(folder/'preparation.json',dict(suite=spec,variants=records,triangle_index_build_s=build))
            render_scene(folder)
            print(f"Fixed test suite complete: {spec['name']}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('Filter/configs/dataset.json'))
    a=p.parse_args();generate(read_json(workspace_path(a.config)))
if __name__=='__main__': main()
