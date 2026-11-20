"""ROS/CUDA-independent map, interval state and persistence."""
from common.io import atomic_json, atomic_npy
from dataclasses import dataclass
from pathlib import Path
import threading
import numpy as np


def block_name(key):
    return '_'.join(map(str, key))


def partition(coord, size):
    """Fixed signed grid; return lexicographically ordered keys and input rows."""
    if len(coord)>=65536:
        from common.native.arrays import native_partition
        return native_partition(coord,size)
    return numpy_partition(coord,size)


def numpy_partition(coord,size):
    if not len(coord):
        return []
    cells = np.floor(np.asarray(coord, np.float64) / np.asarray(size)).astype(np.int64)
    order = np.lexsort(cells.T[::-1])
    sorted_cells = cells[order]
    starts = np.r_[0, np.flatnonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(order)]
    return [(tuple(map(int, sorted_cells[a])), np.sort(order[a:b])) for a, b in zip(starts, ends)]


@dataclass(frozen=True)
class Snapshot:
    revision: int
    stamp: float
    coord: np.ndarray
    blocks: tuple


class BlockMap:
    def __init__(self, size=(4., 4., 3.)):
        self.size = np.asarray(size, np.float64)
        if self.size.shape != (3,) or not np.isfinite(self.size).all() or np.any(self.size <= 0):
            raise ValueError('core_size_m must contain three positive finite values')
        self.lock = threading.RLock()
        self.chunks = []
        self.blocks = {}
        self.revision = self.count = self.invalid_points = self.duplicate_frames = 0
        self.stamp = None

    def add(self, coord, stamp):
        coord = np.asarray(coord, np.float32).reshape(-1, 3)
        if not np.isfinite(stamp):
            raise ValueError('Nonfinite scan timestamp')
        with self.lock:
            if self.stamp is not None and stamp < self.stamp:
                raise ValueError('Scan time moved backwards; restart Entry for the new LIO session')
            if self.stamp == stamp:
                self.duplicate_frames += 1
                return False
            valid = np.isfinite(coord).all(1)
            self.invalid_points += int((~valid).sum())
            coord = np.ascontiguousarray(coord[valid])
            if not len(coord):
                return False
            groups = partition(coord, self.size)
            self.revision += 1
            for key, indices in groups:
                if key not in self.blocks:
                    self.blocks[key] = dict(key=list(key), marker_id=len(self.blocks), points=0, revision=0)
                row = self.blocks[key]
                row['points'] += len(indices)
                row['revision'] += 1
                row['last_map_revision'] = self.revision
            coord.flags.writeable = False
            self.chunks.append(coord)
            self.count += len(coord)
            self.stamp = float(stamp)
            return True

    def metadata(self):
        with self.lock:
            return dict(revision=self.revision, stamp=self.stamp, points=self.count,
                        blocks=[dict(v) for v in self.blocks.values()])

    def snapshot(self):
        with self.lock:
            chunks = tuple(self.chunks)
            revision, stamp = self.revision, self.stamp
            blocks = tuple(dict(v) for v in self.blocks.values())
        coord = np.concatenate(chunks) if chunks else np.empty((0, 3), np.float32)
        coord.flags.writeable = False
        return Snapshot(revision, stamp or 0., coord, blocks)


class NodeStore:
    def __init__(self, tolerance=.3, dwell=.5):
        self.tolerance, self.dwell = tolerance, dwell
        self.nodes = []
        self.goal = None
        self.goal_sequence = 0
        self.goal_reached = False
        self.inside_since = None
        self.baseline = {}
        self.goals = []
        self.last_pose_stamp = None

    def set_goal(self, position, stamp):
        position = np.asarray(position, np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid goal')
        if self.goal is not None and np.linalg.norm(position - self.goal['position']) < .01:
            return False
        self.goal_sequence += 1
        self.goal = dict(id=self.goal_sequence, position=position.tolist(), stamp=float(stamp))
        self.goals.append(dict(self.goal))
        self.goal_reached = False
        self.inside_since = None
        return True

    def observe(self, position, quaternion, stamp, metadata):
        if not metadata['points']:
            return None
        if self.last_pose_stamp is not None and stamp < self.last_pose_stamp:
            raise ValueError('Odometry time moved backwards; restart Entry')
        self.last_pose_stamp = stamp
        if not self.nodes:
            return self._close(position, quaternion, stamp, metadata, root=True)
        if self.goal is None or self.goal_reached:
            return None
        if np.linalg.norm(np.asarray(position) - self.goal['position']) > self.tolerance:
            self.inside_since = None
            return None
        if self.inside_since is None:
            self.inside_since = stamp
        if stamp - self.inside_since + 1e-9 < self.dwell:
            return None
        self.goal_reached = True
        return self._close(position, quaternion, stamp, metadata)

    def _close(self, position, quaternion, stamp, metadata, root=False):
        versions = {block_name(b['key']): b['revision'] for b in metadata['blocks']}
        node = dict(id=f'n{len(self.nodes)}', parent=self.nodes[-1]['id'] if self.nodes else None,
                    frame_id='map', start_stamp=self.nodes[-1]['end_stamp'] if self.nodes else stamp,
                    end_stamp=float(stamp), position=list(map(float, position)),
                    quaternion_xyzw=list(map(float, quaternion)),
                    goal=None if root else dict(self.goal), goals_in_interval=list(self.goals),
                    map_revision=metadata['revision'], point_count=metadata['points'],
                    block_versions=versions,
                    new_blocks=[k for k in versions if k not in self.baseline],
                    updated_blocks=[k for k, v in versions.items() if k in self.baseline and v != self.baseline[k]])
        self.nodes.append(node)
        self.baseline = versions
        self.goals.clear()
        return node


def persist_snapshot(directory, snapshot, size, node=None):
    """Persist immutable block versions; nodes share files for unchanged versions."""
    directory = Path(directory)
    records = {tuple(b['key']): b for b in snapshot.blocks}
    for key, rows in partition(snapshot.coord, size):
        record = records[key]
        path = directory / 'blocks' / block_name(key) / f'r{record["revision"]}.npy'
        if not path.exists():
            atomic_npy(path, snapshot.coord[rows])
        atomic_json(path.parent / 'latest.json', record)
    if node is not None:
        atomic_json(directory / 'nodes' / (node['id'] + '.json'), node)
    atomic_npy(directory / 'latest_geometry.npy', snapshot.coord)
    atomic_json(directory / 'geometry.json', dict(frame_id='map', revision=snapshot.revision,
                stamp=snapshot.stamp, points=len(snapshot.coord), xyz='latest_geometry.npy', blocks=list(snapshot.blocks)))


def load_config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    if value['frame'] != 'map' or value['source_frame'] != 'camera_init':
        raise ValueError('This pipeline requires map <- camera_init with identity TF')
    size = np.asarray(value['core_size_m'], float)
    BlockMap(size)
    if value['context_m'] <= 0 or value['visualization_hz'] <= 0 or value['snapshot_hz'] <= 0:
        raise ValueError('Context and rates must be positive')
    if len(value['condition']) != 4 or not np.isfinite(value['condition']).all():
        raise ValueError('Expected four finite robot conditions')
    if value['arrival_tolerance_m'] <= 0 or value['arrival_dwell_s'] < 0:
        raise ValueError('Invalid arrival tolerance/dwell')
    if value.get('cost_trigger','continuous') not in ('continuous','goal'):raise ValueError('Invalid cost_trigger')
    if value.get('inference_frame','map') not in ('map','world'):raise ValueError('Invalid inference_frame')
    if value.get('inference_recompute','full') not in ('full','changed_blocks'):raise ValueError('Invalid inference_recompute')
    if value.get('scan_path_source','gpmp') not in ('astar','gpmp'):raise ValueError('Invalid scan_path_source')
    if value.get('planner_backend','dc') not in ('dc','surface'):raise ValueError('Invalid planner_backend')
    if not np.isfinite(value.get('surface_slice_dh',.5)) or value.get('surface_slice_dh',.5)<=0:
        raise ValueError('Invalid surface_slice_dh')
    if (not np.isfinite([value.get('display_voxel_m',.1),value.get('display_hz',1.)]).all() or
            value.get('display_voxel_m',.1)<=0 or value.get('display_hz',1.)<=0 or value.get('display_max_points',200000)<=0):
        raise ValueError('Invalid display downsampling settings')
    return value
