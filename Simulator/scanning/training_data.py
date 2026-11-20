"""Filter XYZ windows with complete static references for data diagnostics."""
from pathlib import Path
import numpy as np

from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import read_json, write_json


def export_reference_windows(folder, attributes, clean, boxes):
    """Store complete static reference indices on exactly the Filter windows."""
    indices, cores, offsets, samples = [], [], [0], []
    for row in boxes['windows']:
        low, high = np.array(row['context_lower']), np.array(row['context_upper'])
        selected = np.flatnonzero(((clean >= low) & (clean < high)).all(1))
        low, high = np.array(row['core_lower']), np.array(row['core_upper'])
        core = ((clean[selected] >= low) & (clean[selected] < high)).all(1)
        indices.append(selected)
        cores.append(core)
        offsets.append(offsets[-1] + len(selected))
        samples.append(dict(id=row['id'], box=row['box_path'], input_points=row['context_points'],
            input_core_points=row['core_points'], target_points=len(selected),
            target_core_points=int(core.sum())))
    np.save(folder / 'reference_index.npy', np.concatenate(indices) if indices else np.empty(0, np.int64))
    np.save(folder / 'reference_core.npy', np.concatenate(cores) if cores else np.empty(0, bool))
    np.save(folder / 'reference_offsets.npy', np.asarray(offsets, np.int64))
    write_json(folder / 'training_pair.json', dict(
        model_input_fields=['x', 'y', 'z'], coordinate_frame='map', units='m',
        rough='arrays_000/coord.npy', filter_target='arrays_000/keep_target.npy',
        reference='../clean.pcd', reference_mesh='../static_mesh.ply',
        filter_windows=dict(index='windows_000_index.npy', core='windows_000_core.npy', offsets='windows_000_offsets.npy'),
        reference_windows=dict(index='reference_index.npy', core='reference_core.npy', offsets='reference_offsets.npy'),
        samples=samples, geometry_group_id=read_json(folder / 'scene.json')['geometry_group_id'],
        reference_scope='complete static Mesh, including unobserved surfaces inside each context',
        metadata_is_teacher_only=True, no_training_started=True))
    # Check the real arrays and shared maps, rather than building a test framework.
    np.testing.assert_array_equal(read_pcd_xyz(folder / 'rough.pcd'), attributes['coord'])
    np.testing.assert_array_equal(np.load(folder / 'arrays_000/keep_target.npy'), attributes['keep_target'])
    return dict(boxes=len(samples), reference_points=len(clean),
        observations=len(attributes['coord']), filter_pcd_array_alignment=True,
        input='XYZ only', static_reference='complete static Mesh samples')


class SharedScanBoxes:
    """A paired-data loader, without encoding or changing either network.

    `xyz` is the only model input. Targets, core masks and indices are training
    supervision. Clean Mesh points are never concatenated into the input.
    """
    def __init__(self, folder):
        self.folder = Path(folder)
        self.metadata = read_json(self.folder / 'training_pair.json')
        self.xyz = np.load(self.folder / self.metadata['rough'], mmap_mode='r')
        self.target = np.load(self.folder / self.metadata['filter_target'], mmap_mode='r')
        self.clean = read_pcd_xyz(self.folder / self.metadata['reference'])
        self.input_index = np.load(self.folder / 'windows_000_index.npy', mmap_mode='r')
        self.input_core = np.load(self.folder / 'windows_000_core.npy', mmap_mode='r')
        self.input_offsets = np.load(self.folder / 'windows_000_offsets.npy', mmap_mode='r')
        # Previously generated datasets retain their recorded array filenames.
        reference = self.metadata.get('reference_windows', self.metadata.get('reconstruction_windows'))
        if reference is None:
            raise ValueError('Dataset metadata is missing static reference windows')
        self.clean_index = np.load(self.folder / reference['index'], mmap_mode='r')
        self.clean_core = np.load(self.folder / reference['core'], mmap_mode='r')
        self.clean_offsets = np.load(self.folder / reference['offsets'], mmap_mode='r')

    def __len__(self):
        return len(self.metadata['samples'])

    def __getitem__(self, index):
        a, b = self.input_offsets[index:index+2]
        c, d = self.clean_offsets[index:index+2]
        point_index = np.array(self.input_index[a:b])
        target_index = np.array(self.clean_index[c:d])
        return dict(xyz=np.array(self.xyz[point_index]), point_index=point_index,
            core_mask=np.array(self.input_core[a:b]), keep_target=np.array(self.target[point_index]),
            clean_xyz=self.clean[target_index], clean_point_index=target_index,
            clean_core_mask=np.array(self.clean_core[c:d]))
