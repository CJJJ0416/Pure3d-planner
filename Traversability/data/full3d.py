"""Full geometry windows with unchanged native Tomogram supervision."""
import argparse
from copy import deepcopy
from pathlib import Path

from Traversability.data.rotating import prepare_generation
from utils.pointcloud import read_json, workspace_path, write_json


def prepare_fixed(config):
    root = workspace_path(config['data']) / 'fixed'
    root.mkdir(parents=True, exist_ok=True)
    source = workspace_path(config['fixed_data'])
    metadata = read_json(source / 'manifest.json')
    metadata['scenes'] = [s for s in metadata['scenes'] if s['split'] in ['val', 'test']]
    metadata['blocks'] = [s for s in metadata['blocks'] if s['split'] in ['val', 'test']]
    metadata.update(architecture=config['architecture'], config=config)
    for scene in metadata['scenes']:
        folder = root / scene['id']
        folder.mkdir(exist_ok=True)
        # Labels/XYZ and query-core partitions are immutable. Feature folders are new.
        for p in (source / scene['id']).iterdir():
            if p.is_file() and p.name != 'ready.json':
                link = folder / p.name
                if not link.exists():
                    link.symlink_to(p.resolve())
        write_json(folder / 'ready.json', dict(scene=scene,
            blocks=[b for b in metadata['blocks'] if b['scene'] == scene['id']]))
    write_json(root / 'manifest.json', metadata)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    local = deepcopy(config)
    local['fixed_data'] = str(prepare_fixed(config))
    prepare_generation(local, args.generation)


if __name__ == '__main__':
    main()
