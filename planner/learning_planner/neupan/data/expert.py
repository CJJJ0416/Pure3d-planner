"""Generate supervision using the existing native VST A* implementation."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

from planner.learning_planner.neupan.geometry.tomogram import Tomogram, VST_LIMIT

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAP = ROOT / 'datasets/wfc_capability_dr_v2/scene_20261109/vst_nominal.pickle'
NATIVE = ROOT / 'third_party_lib/vst_global_planner/native'
KINDS = ('flat', 'detour', 'up', 'down', 'cross_floor')


def fingerprint(path):
    with Path(path).open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_native():
    sources = [NATIVE / 'src/a_star' / file for file in ('a_star_search.cc', 'python_interface.cc')]
    headers = [NATIVE / 'src/a_star/a_star_search.h', NATIVE / 'src/common/data_types.h']
    digest = sha256(''.join(fingerprint(p) for p in sources + headers).encode()).hexdigest()[:16]
    build = Path(__file__).parent / '__pycache__' / 'vst' / digest
    build.mkdir(parents=True, exist_ok=True)
    name = 'a_star_learning'
    library = build / (name + sysconfig.get_config_var('EXT_SUFFIX'))
    if not library.exists():
        with NamedTemporaryFile(dir=build, suffix='.so') as tmp:
            subprocess.run(['g++', '-O3', '-DNDEBUG', '-std=c++17', '-shared', '-fPIC',
                            '-DVST_ASTAR_MODULE=' + name, '-I/usr/include/eigen3',
                            '-I' + sysconfig.get_path('include'), '-I' + str(NATIVE / 'src'),
                            '-I' + str(NATIVE / '3rdparty/pybind11/include'),
                            *map(str, sources), '-o', tmp.name], check=True)
            # Publish atomically. Concurrent readers only see a complete library.
            os.link(tmp.name, build / (Path(tmp.name).name + '.ready'))
            os.replace(build / (Path(tmp.name).name + '.ready'), library)
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def native_log(path):
    # The native API writes using printf; capture its per-query timings separately.
    sys.stdout.flush()
    with path.open('w') as log:
        original = os.dup(1)
        try:
            os.dup2(log.fileno(), 1)
            yield
        finally:
            import ctypes
            ctypes.CDLL(None).fflush(None)
            os.dup2(original, 1)
            os.close(original)


class NativeExpert:
    def __init__(self, tomo):
        self.tomo = tomo
        self.astar = load_native().Astar()
        cost = tomo.cost.copy()
        cost[~np.isfinite(tomo.ground)] = 50.
        gateway = np.zeros_like(cost)
        shared = np.abs(np.diff(tomo.ground, axis=0)) < .1
        change = np.diff(cost, axis=0)
        gateway[:-1][shared & (change < -8)] = 2.
        gateway[1:][shared & (change > 8)] = -2.
        self.astar.init(VST_LIMIT, len(cost), tomo.resolution, 1.,
                        *(np.ascontiguousarray(grid.reshape(-1, grid.shape[-1]), dtype=np.float64)
                          for grid in (cost, tomo.ground, gateway)))

    def route(self, start, goal):
        # Search accepts (layer, y, x); the result matrix emits (layer, x, y).
        if not self.astar.search(start[[0, 2, 1]], goal[[0, 2, 1]]):
            return None
        suffix = np.asarray(self.astar.get_result_matrix(), dtype=np.int32).reshape(-1, 3)
        if len(suffix) == 0:
            return None
        cells = np.vstack((start, suffix))
        xyz = self.tomo.xyz(cells)
        # Native closure can identify equal discretized heights across slices.
        if np.linalg.norm(xyz[-1] - self.tomo.xyz([goal])[0]) > .025:
            return None
        if np.any(np.max(np.abs(np.diff(cells[:, 1:], axis=0)), axis=1) != 1):
            return None
        return cells


def category(xyz):
    dz = float(xyz[-1, 2] - xyz[0, 2])
    if abs(dz) >= 1.2:
        return 'cross_floor'
    if abs(dz) >= .18:
        return 'up' if dz > 0 else 'down'
    if np.ptp(xyz[:, 2]) > .30:
        return None
    distance = np.linalg.norm(xyz[-1, :2] - xyz[0, :2])
    length = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum()
    return 'detour' if length > 1.25 * distance else 'flat'


def split_regions(xyz, tile_size, scene_split, region_file, seed):
    if scene_split:
        yield scene_split, 'scene', np.ones(len(xyz), dtype=bool), lambda path: True
        return
    if region_file:
        regions = json.loads(Path(region_file).read_text())['regions']
    else:
        if not np.isfinite(tile_size) or tile_size <= 6.1:
            raise ValueError('Spatial split tiles must fit the two 3.05 m margins')
        tiles = np.floor(xyz[:, :2] / tile_size).astype(int)
        within = xyz[:, :2] - tiles * tile_size
        eligible = np.all((within >= 3.05) & (within < tile_size - 3.05), axis=1)
        tiles, counts = np.unique(tiles[eligible], axis=0, return_counts=True)
        tiles = tiles[counts > 1]
        rng = np.random.default_rng(seed)
        regions = []
        for index, tile in enumerate(rng.permutation(tiles)):
            split = ('validation' if index == 1 else 'test' if index == 2 else
                     'train' if index == 0 or index % 10 < 7 else
                     'validation' if index % 10 < 9 else 'test')
            regions.append(dict(id=f'{tile[0]}_{tile[1]}', split=split,
                                lower=(tile * tile_size).tolist(),
                                upper=((tile + 1) * tile_size).tolist()))
    for index, region in enumerate(regions):
        for previous in regions[:index]:
            if region['split'] != previous['split'] and np.all(
                    np.maximum(region['lower'], previous['lower'])
                    < np.minimum(region['upper'], previous['upper'])):
                raise ValueError(f'Spatial splits overlap: {previous["id"]}, {region["id"]}')
        lower, upper = np.asarray(region['lower']), np.asarray(region['upper'])
        if (lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper)
                or region['split'] not in ('train', 'validation', 'test')):
            raise ValueError(f'Invalid spatial split: {region}')
        # A full 6 m student crop stays in its split even when started near an endpoint.
        def inside(points, lower=lower + 3.05, upper=upper - 3.05):
            return np.all((points[:, :2] >= lower) & (points[:, :2] < upper), axis=1)
        yield region['split'], str(region['id']), inside(xyz), lambda points, check=inside: check(points).all()


def generate(path, output, count, seed, attempts, scene_split=None, tile_size=12., region_file=None):
    tomo = Tomogram.read(path)
    output.mkdir(parents=True, exist_ok=False)
    cells = np.argwhere(tomo.admitted).astype(np.int32)
    xyz = tomo.xyz(cells)
    _, unique = np.unique(np.round(xyz, 3), axis=0, return_index=True)
    cells, xyz = cells[unique], xyz[unique]
    rng = np.random.default_rng(seed)
    pools = []
    for split, region, mask, contains in split_regions(xyz, tile_size, scene_split, region_file, seed):
        indices = np.flatnonzero(mask)
        if len(indices) > 1:
            pools.append((split, region, indices, cKDTree(xyz[indices]), contains))
    if not pools:
        raise ValueError('No region has two eligible endpoints. Check region bounds or use --scene-split for pilot data.')
    if not scene_split and {pool[0] for pool in pools} != {'train', 'validation', 'test'}:
        raise ValueError('Spatial split needs usable train, validation and test regions. Provide --regions or use pilot --scene-split.')
    target = {kind: count // len(KINDS) + (index < count % len(KINDS))
              for index, kind in enumerate(KINDS)}
    failures, totals, tried, paths = Counter(), Counter(), set(), []
    route_records = []
    grouped = defaultdict(list)
    for pool in pools:
        grouped[pool[0]].append(pool)
    splits = list(grouped)
    requested_splits = ({scene_split: count} if scene_split else
                        {'train': count - 2 * (count * 15 // 100),
                         'validation': count * 15 // 100, 'test': count * 15 // 100})
    totals_split = Counter()
    split_schedule = (['train'] * 7 + ['validation', 'test', 'validation', 'test']
                      if not scene_split else [scene_split])
    with native_log(output / 'native.log'):
        native_by_region = {}
        for attempt in range(attempts):
            if len(paths) >= count:
                break
            wanted = KINDS[attempt % len(KINDS)]
            # Reserve the first 80% of attempts for the class targets; use any
            # remaining valid routes afterwards and report the resulting imbalance.
            balanced = attempt < attempts * .8
            if balanced and totals[wanted] >= target[wanted]:
                continue
            split = split_schedule[attempt % len(split_schedule)]
            if totals_split[split] >= requested_splits[split]:
                remaining = [label for label in splits if totals_split[label] < requested_splits[label]]
                split = remaining[attempt % len(remaining)]
            regions = grouped[split]
            _, region, indices, tree, contains = regions[int(rng.integers(len(regions)))]
            key = (split, region)
            if key not in native_by_region:
                lower = cells[indices].min(axis=0)
                upper = cells[indices].max(axis=0) + 1
                lower[0], upper[0] = 0, len(tomo.cost)
                block = tuple(slice(a, b) for a, b in zip(lower, upper))
                regional = Tomogram(tomo.cost[block], tomo.ground[block],
                                    tomo.origin + lower[1:] * tomo.resolution, tomo.resolution)
                native_by_region[key] = NativeExpert(regional), lower
            native, lower = native_by_region[key]
            first = int(rng.choice(indices))
            last = indices[tree.query_ball_point(xyz[first], 8.)]
            delta = xyz[last] - xyz[first]
            distance = np.linalg.norm(delta, axis=1)
            eligible = distance >= .8
            if balanced:
                dz = delta[:, 2]
                eligible &= {
                    'flat': abs(dz) < .18, 'detour': abs(dz) < .18,
                    'up': (dz >= .18) & (dz < 1.2), 'down': (dz <= -.18) & (dz > -1.2),
                    'cross_floor': abs(dz) >= 1.2,
                }[wanted]
            last = last[eligible]
            if not len(last):
                failures['no_endpoint_pair'] += 1
                continue
            second = int(rng.choice(last))
            pair = (first, second)
            if pair in tried:
                failures['duplicate_pair'] += 1
                continue
            tried.add(pair)
            route = native.route(cells[first] - lower, cells[second] - lower)
            if route is None:
                failures['native_no_valid_path'] += 1
                continue
            route += lower
            locations = tomo.xyz(route)
            if not contains(locations):
                failures['crosses_split_boundary'] += 1
                continue
            kind = category(locations)
            if kind is None or (balanced and kind != wanted):
                # A pair sampled for flat travel can turn out to require a detour.
                tried.discard(pair)
                failures['category_mismatch'] += 1
                continue
            totals[kind] += 1
            totals_split[split] += 1
            route_records.append(dict(id=len(paths), split=split, region=region, kind=kind,
                                      goal=tomo.xyz([cells[second]])[0].tolist()))
            paths.append(route)
    counts = np.r_[0, np.cumsum([len(route) for route in paths])]
    np.savez_compressed(output / 'routes.npz', offsets=counts,
                        cells=np.concatenate(paths) if paths else np.empty((0, 3), dtype=np.int32),
                        support_xyz=tomo.xyz(np.concatenate(paths)) if paths else np.empty((0, 3)))
    report = dict(tomogram=str(Path(path).resolve()), sha256=fingerprint(path),
                  seed=seed, requested=count, generated=len(paths), target_per_kind=target,
                  generated_per_kind=dict(totals), generated_per_split=dict(Counter(r['split'] for r in route_records)),
                  rejected=dict(failures), teacher='native_vst_astar', native_scope='regional_xy', routes=route_records)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return {key: value for key, value in report.items() if key != 'routes'}


def read_routes(directory):
    directory = Path(directory)
    report = json.loads((directory / 'manifest.json').read_text())
    with np.load(directory / 'routes.npz', allow_pickle=False) as store:
        cells, offsets = store['cells'], store['offsets']
    return report, [cells[a:b] for a, b in zip(offsets, offsets[1:])]
