"""ROS-independent bounded LIO voxel map and latest-only timer scheduling."""
from Entry.utils.paths import ROOT, NATIVE_DIR
import ctypes
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import numpy as np
from Entry.utils.core import Snapshot


def load_local_config(path, model=None):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault('context_mode', 'stable_core')
    if cfg['context_mode'] not in ('stable_core', 'roi_clipped'):
        raise ValueError('context_mode must be stable_core or roi_clipped')
    cfg['model'] = model or cfg['model']
    cfg.update(cfg['models'][cfg['model']])
    if cfg['frame'] != 'world' or cfg['source_frame'] != 'camera_init':
        raise ValueError('Local cost expects gravity-aligned world <- camera_init')
    cfg.setdefault('odom_topic', '/faster_lio/Odometry')
    cfg.setdefault('body_frame', 'body')
    cfg.setdefault('box_size_m', [6., 6., 5.])
    cfg.setdefault('box_shift_m', .2)
    cfg.setdefault('max_pose_age_s', .5)
    cfg.setdefault('cache_publish_hz', 2.)
    from Entry.utils.box import LioBox
    LioBox(cfg['box_size_m'], cfg['voxel_m'], cfg['box_shift_m'])
    for key in ('voxel_m', 'period_s', 'max_input_age_s', 'max_result_age_s', 'max_pose_age_s', 'cache_publish_hz'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if cfg['max_voxels'] <= 0 or len(cfg['condition']) != 4 or not np.isfinite(cfg['condition']).all():
        raise ValueError('Invalid local point budget/condition')
    if cfg['expected_encoder'] not in ('utonia_small', 'utonia_tiny'):
        raise ValueError('Unsupported local model')
    for key in ('core_size_m',):
        a = np.asarray(cfg[key])
        if a.shape != (3,) or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(cfg['context_m']) or cfg['context_m'] <= 0:
        raise ValueError('Invalid context width')
    for key in ('checkpoint', 'output_root'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    from Entry.utils.knn import options_from_config
    cfg.setdefault('knn_filter',{'enabled':False})
    options_from_config(cfg['knn_filter'])
    maximum=cfg['knn_filter'].get('removed_display_max_points',2000)
    if not isinstance(maximum,int) or isinstance(maximum,bool) or maximum<1:
        raise ValueError('knn_filter.removed_display_max_points must be a positive integer')
    return cfg


class LocalVoxelMap:
    """First actual XYZ per world voxel; retain only points inside [lower,upper).

    The original full map is independent. No centroid, occupancy inflation,
    geometric terrain filtering, or free-space filling is applied here.
    """
    def __init__(self, voxel_m=.05, max_voxels=500000):
        from planner.native import compile_cached
        path, _, _ = compile_cached((NATIVE_DIR / 'local_voxels.cpp'),
            ROOT / '.cache/native/local_requests', 'libentry_local_voxels')
        self.lib = ctypes.CDLL(str(path)); lib = self.lib; ptr = ctypes.c_void_p
        lib.local_voxels_new.argtypes = [ctypes.c_double, ctypes.c_uint64]
        lib.local_voxels_new.restype = ptr
        lib.local_voxels_free.argtypes = [ptr]
        lib.local_voxels_bounds.argtypes = [ptr, ptr, ptr]
        lib.local_voxels_add.argtypes = [ptr, ptr, ctypes.c_uint64]
        lib.local_voxels_stats.argtypes = [ptr, ptr]
        lib.local_voxels_copy.argtypes = [ptr, ptr]
        lib.local_voxels_error.restype = ctypes.c_char_p
        self.handle = lib.local_voxels_new(voxel_m, max_voxels)
        self.lock = threading.RLock(); self.lower = self.upper = None
        self.stamp = None; self.duplicates = 0
        self.last_in_box_observations = 0
        self.total_in_box_observations = 0
        if not self.handle:
            raise ValueError(lib.local_voxels_error().decode())

    def _check(self, value):
        if value < 0:
            raise ValueError(self.lib.local_voxels_error().decode())
        return bool(value)

    def set_bounds(self, lower, upper):
        lower, upper = (np.ascontiguousarray(x, dtype=np.float64) for x in (lower, upper))
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError('Expected XYZ box bounds')
        with self.lock:
            changed = self._check(self.lib.local_voxels_bounds(self.handle, lower.ctypes.data, upper.ctypes.data))
            self.lower, self.upper = lower.copy(), upper.copy()
            return changed

    def add(self, xyz, stamp):
        if not np.isfinite(stamp):
            raise ValueError('Invalid scan stamp')
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError('Expected XYZ')
        with self.lock:
            self.last_in_box_observations = 0
            if self.stamp is not None and stamp < self.stamp:
                raise ValueError('LIO time moved backwards; restart the local cost session')
            if self.stamp == stamp:
                self.duplicates += 1
                return False
            changed = self._check(self.lib.local_voxels_add(self.handle, xyz.ctypes.data, len(xyz)))
            observed = self.stats()['in_box_observations']
            self.last_in_box_observations = observed-self.total_in_box_observations
            self.total_in_box_observations = observed
            self.stamp = float(stamp)
            return changed

    def stats(self):
        with self.lock:
            a = np.empty(5, np.uint64)
            self.lib.local_voxels_stats(self.handle, a.ctypes.data)
            return dict(zip(('points', 'revision', 'capacity_rejections', 'invalid_points', 'in_box_observations'), map(int, a)),
                        stamp=self.stamp, duplicate_scans=self.duplicates)

    def snapshot(self):
        with self.lock:
            info = self.stats(); coord = np.empty((info['points'], 3), np.float32)
            self._check(self.lib.local_voxels_copy(self.handle, coord.ctypes.data))
            return Snapshot(info['revision'], self.stamp, coord, ())

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.local_voxels_free(self.handle); self.handle = None


@dataclass(frozen=True)
class ContextSnapshot(Snapshot):
    context_coord: np.ndarray
    query_indices: np.ndarray
    context_revision: int
    query_revision: int
    metadata: dict


class ContextVoxelMap:
    """Bounded observed context with an independent, half-open output ROI."""
    def __init__(self, voxel_m=.05, max_voxels=500000, core_size=(4.,4.,3.), halo=1., mode='stable_core'):
        self.size = np.asarray(core_size, dtype=float)
        if self.size.shape != (3,) or not np.isfinite(self.size).all() or np.any(self.size <= 0):
            raise ValueError('Invalid context core size')
        if not np.isfinite(halo) or halo <= 0 or mode not in ('stable_core', 'roi_clipped'):
            raise ValueError('Invalid context mode/halo')
        self.halo, self.mode = float(halo), mode
        self.native = LocalVoxelMap(voxel_m, max_voxels)
        self.lock = self.native.lock
        self.native.lib.local_voxels_query_bounds.argtypes = [ctypes.c_void_p]*3
        self.native.lib.local_voxels_query_stats.argtypes = [ctypes.c_void_p]*2
        self.lower = self.upper = None
        self.core_lower = self.core_upper = None
        self.revision = self.query_revision = 0
        self.stamp = None
        self.last_in_box_observations = self.total_in_box_observations = 0
        self.last_maintenance_s = 0.

    def set_bounds(self, lower, upper):
        lower, upper = (np.ascontiguousarray(a, dtype=np.float64) for a in (lower, upper))
        if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower,upper]).all()
                or np.any(lower >= upper)):
            raise ValueError('Invalid ROI bounds')
        with self.lock:
            if self.lower is not None and np.array_equal(lower,self.lower) and np.array_equal(upper,self.upper):
                return False
            started = time.perf_counter()
            self.core_lower = np.floor(lower/self.size).astype(np.int64)
            self.core_upper = np.ceil(upper/self.size).astype(np.int64)
            context_lower = self.core_lower*self.size-self.halo if self.mode == 'stable_core' else lower
            context_upper = self.core_upper*self.size+self.halo if self.mode == 'stable_core' else upper
            self.native.set_bounds(context_lower, context_upper)
            self.native._check(self.native.lib.local_voxels_query_bounds(
                self.native.handle, lower.ctypes.data, upper.ctypes.data))
            self.lower, self.upper = lower.copy(), upper.copy()
            self.revision += 1; self.query_revision += 1
            self.last_maintenance_s = time.perf_counter()-started
            return True

    def _query_stats(self):
        a = np.empty(2, np.uint64)
        self.native.lib.local_voxels_query_stats(self.native.handle, a.ctypes.data)
        return tuple(map(int, a))

    def add(self, xyz, stamp):
        started = time.perf_counter()
        with self.lock:
            before, _ = self._query_stats()
            changed = self.native.add(xyz, stamp)
            count, observed = self._query_stats()
            self.last_in_box_observations = observed-self.total_in_box_observations
            self.total_in_box_observations = observed
            if self.last_in_box_observations:
                self.stamp = float(stamp)
            if changed:
                self.revision += 1
            if count != before:
                self.query_revision += 1
            self.last_maintenance_s = time.perf_counter()-started
            if self.native.stats()['capacity_rejections']:
                raise ValueError('Context capacity exceeded; increase max_voxels and restart cost. Input was truncated.')
            return changed

    def stats(self):
        with self.lock:
            info = self.native.stats(); count, observed = self._query_stats()
            return dict(info, points=count, context_points=info['points'],
                context_revision=info['revision'], query_revision=self.query_revision, revision=self.revision,
                stamp=self.stamp, context_stamp=info['stamp'], context_mode=self.mode,
                in_box_observations=observed, context_observations=info['in_box_observations'],
                context_lower=None if self.native.lower is None else self.native.lower.tolist(),
                context_upper=None if self.native.upper is None else self.native.upper.tolist(),
                context_maintenance_s=self.last_maintenance_s)

    def snapshot(self):
        with self.lock:
            started = time.perf_counter()
            full = self.native.snapshot(); copied = time.perf_counter()
            rows = (np.empty(0, np.int64) if self.lower is None else
                    np.flatnonzero(((full.coord>=self.lower)&(full.coord<self.upper)).all(1)))
            coord = np.ascontiguousarray(full.coord[rows]); extracted = time.perf_counter()
            info = self.stats()
            metadata = dict(context_mode=self.mode, context_revision=full.revision,
                query_revision=self.query_revision, context_stamp=full.stamp,
                roi_lower=None if self.lower is None else self.lower.tolist(),
                roi_upper=None if self.upper is None else self.upper.tolist(),
                context_lower=info['context_lower'], context_upper=info['context_upper'],
                retained_core_lower=None if self.core_lower is None else self.core_lower.tolist(),
                retained_core_upper=None if self.core_upper is None else self.core_upper.tolist(),
                context_points=len(full.coord), query_points=len(coord),
                context_array_bytes=int(full.coord.nbytes), query_array_bytes=int(coord.nbytes),
                context_copy_s=copied-started, query_extract_s=extracted-copied)
            return ContextSnapshot(self.revision, self.stamp, coord, (), full.coord, rows,
                                   full.revision, self.query_revision, metadata)

    def close(self):
        self.native.close()


class LatestOnlyTimer:
    """At most one in-flight job. Ticks never queue intermediate snapshots."""
    def __init__(self, period_s):
        if not np.isfinite(period_s) or period_s <= 0:
            raise ValueError('Invalid timer period')
        self.period = float(period_s); self.next_due = 0.
        self.active = None; self.completed = None

    def start(self, now, revision):
        if self.active is not None or now < self.next_due or revision == self.completed:
            return False
        self.active = revision; self.next_due = now + self.period
        return True

    def finish(self, success=True):
        if success:
            self.completed = self.active
        self.active = None


class LocalLatency:
    """Monotonic receiver-to-publication timing for coalesced scan batches.

    Each received scan is associated with the first subsequent snapshot. A
    snapshot has many scans: latest-frame age and per-frame latency are
    intentionally separate. A cache-only refresh without new input contributes
    no new per-frame samples.
    """
    def __init__(self):
        self.pending = []
        self.latest = None

    def received(self, stamp, started, ingested, changed):
        if ingested < started:
            raise ValueError('Ingestion ended before reception')
        row = dict(stamp=float(stamp), received_wall=float(started),
                   ingested_wall=float(ingested), geometry_changed=bool(changed))
        self.pending.append(row)
        self.latest = row

    def snapshot(self, started, ready):
        if ready < started:
            raise ValueError('Snapshot ended before it started')
        batch = dict(frames=self.pending, latest=self.latest,
                     snapshot_started_wall=float(started), snapshot_ready_wall=float(ready))
        self.pending = []
        return batch

    @staticmethod
    def finish(batch, submitted, result_ready, published=None):
        if not batch['snapshot_ready_wall'] <= submitted <= result_ready:
            raise ValueError('Invalid local timing order')
        frames, latest = batch['frames'], batch['latest']
        out = dict(clock='host monotonic wall time', start='Entry cloud callback entry',
                   end='cost publisher publish() returned',
                   input_frames=len(frames), changed_frames=sum(r['geometry_changed'] for r in frames),
                   snapshot_s=batch['snapshot_ready_wall']-batch['snapshot_started_wall'],
                   snapshot_to_submit_s=submitted-batch['snapshot_ready_wall'],
                   submit_to_result_s=result_ready-submitted,
                   latest_scan_wait_to_snapshot_s=None if latest is None else batch['snapshot_started_wall']-latest['ingested_wall'],
                   latest_scan_ingest_s=None if latest is None else latest['ingested_wall']-latest['received_wall'],
                   published=published is not None)
        if published is None:
            return out
        if published < result_ready:
            raise ValueError('Publication preceded result')
        out.update(result_to_publish_s=published-result_ready,
                   latest_scan_to_publish_s=None if latest is None else published-latest['received_wall'],
                   frame_to_publish_s=[published-r['received_wall'] for r in frames],
                   changed_frame_to_publish_s=[published-r['received_wall'] for r in frames if r['geometry_changed']],
                   frame_ingest_s=[r['ingested_wall']-r['received_wall'] for r in frames],
                   frame_wait_to_snapshot_s=[batch['snapshot_started_wall']-r['ingested_wall'] for r in frames],
                   frame_stamps=[r['stamp'] for r in frames])
        return out
