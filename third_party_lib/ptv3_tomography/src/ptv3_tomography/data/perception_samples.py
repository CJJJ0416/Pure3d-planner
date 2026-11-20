"""Current-frame point selection and local-map cropping."""
import numpy as np

from ptv3_tomography.data.lidar_reference_pairs import add_teacher_masks


def cropped_config(dataset, grid_size_xy):
    return dict(dataset.spec['config'], grid_size_xy=list(grid_size_xy),
                supervision_policy='native_vst_available_only')


def load_sample(dataset, index, max_points, grid_size_xy):
    frame = dataset.frames[index]
    with np.load(dataset.asset(frame['scan_file']), allow_pickle=False) as archive:
        ids = np.flatnonzero(archive['hit_mask'])
        if len(ids) > max_points:
            ids = ids[np.linspace(0, len(ids)-1, max_points).astype(int)]
        points = archive['points_sensor_m'][ids].astype(np.float32)
    with np.load(dataset.directory / frame['label_file'], allow_pickle=False) as archive:
        source = {k: archive[k].copy() for k in archive.files}
    original = np.asarray(dataset.spec['config']['grid_size_xy'])
    size = np.asarray(grid_size_xy)
    offset = (original-size)//2
    crop = (slice(None), slice(offset[0], offset[0]+size[0]), slice(offset[1], offset[1]+size[1]))
    labels = add_teacher_masks({k: v[crop].copy()
              if v.ndim >= 3 and v.shape[:3] == source['target_available'].shape else v
              for k, v in source.items()})
    targets = dict(cost=labels['cost_normalized'].astype(np.float32),
                   height=labels['ground_height_gravity_m'].astype(np.float32),
                   mask_cost=labels['mask_cost'], mask_height=labels['mask_height'])
    return dict(points=points, up=labels['gravity_up_sensor'].astype(np.float32), targets=targets,
                frame_index=index, selected_valid_points=len(ids), episode_id=frame['episode_id'])
