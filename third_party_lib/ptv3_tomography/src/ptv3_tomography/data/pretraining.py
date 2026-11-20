"""Registered current-frame captures and train/dev/test frame references."""
import json

from ptv3_tomography.data.lidar_reference_pairs import LidarStepDataset
from ptv3_tomography.data.perception_samples import cropped_config, load_sample
from ptv3_tomography.io import read_json
from ptv3_tomography.paths import relocated_path


class FrameCollection:
    def __init__(self, registry_path, config):
        registry_path = relocated_path(registry_path).resolve()
        registry = read_json(registry_path)
        self.config, self.datasets, self.identity = config, [], []
        self.references = {key: [] for key in ('train', 'dev', 'test')}
        for entry in registry['datasets']:
            directory = relocated_path(entry['path'])
            if not directory.is_absolute():
                directory = registry_path.parent / directory
            dataset = LidarStepDataset(directory)
            self.map_config = cropped_config(dataset, config['grid_size_xy'])
            package = len(self.datasets)
            self.datasets.append(dataset)
            indices = entry.get('frame_indices', list(range(len(dataset))))
            self.references[entry['split']].extend((package, i) for i in indices)
            self.identity.append({k: entry[k] for k in ('id', 'building_id', 'split')} | {'frame_indices': indices})

    def sample(self, split, index):
        package, frame = self.references[split][index]
        sample = load_sample(self.datasets[package], frame, self.config['max_points'], self.config['grid_size_xy'])
        sample['capture_id'] = self.identity[package]['id']
        return sample

    def summary(self):
        return dict(frames={name: len(refs) for name, refs in self.references.items()},
                    captures=self.identity, map_config=self.map_config)


def canonical_digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
