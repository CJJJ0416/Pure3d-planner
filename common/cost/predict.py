"""Persistent PTv3 inference with context-level scalar prediction caches."""
from copy import deepcopy
import hashlib
from pathlib import Path
import time
import numpy as np
from common.mapping.core import block_name, partition
from common.mapping.core import Snapshot
from Simulator.unitree_gazebo.scripts.frame_config import rotation_matrix


class BlockCostPredictor:
    def __init__(self, checkpoint, condition, size, context, device='cuda:0', expected_epoch=56,
                 expected_encoder='utonia_small'):
        import torch
        import point_transformer_v3 as ptv3
        from utils.Traversability.models.sliced import SlicedCostHead

        self.torch = torch
        self.device = device
        checkpoint = Path(checkpoint).resolve()
        # Read once: a concurrently updated training file cannot change this session.
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            stream.seek(0)
            state = torch.load(stream, map_location='cpu', weights_only=False)
        self.config = deepcopy(state['config'])
        if state['epoch'] != expected_epoch:
            raise ValueError(f'Expected epoch {expected_epoch}; checkpoint contains {state["epoch"]}')
        if self.config['encoder']['name'] != expected_encoder:
            raise ValueError(f'Entry expects the selected {expected_encoder} checkpoint')
        if not np.allclose(size, self.config['window']['core_size_m']) or not np.isclose(context, self.config['window']['context_m']):
            raise ValueError('Entry window must match checkpoint core/context sizes')
        self.config['device'] = device
        torch.set_num_threads(self.config['cpu_threads'])
        torch.manual_seed(self.config['seed'])
        self.encoder = ptv3.PointTransformerV3.from_preset(self.config['encoder']['name']).to(device).eval()
        self.encoder.precision = 'fp32'
        self.encoder.load_state_dict(state['encoder'])
        self.encoder.requires_grad_(False)
        self.head = SlicedCostHead(self.config).to(device).eval()
        self.head.load_state_dict(state['model'])
        self.head.requires_grad_(False)
        self.condition = torch.tensor(condition, device=device, dtype=torch.float32)[None, None]
        self.size = np.asarray(size, np.float64)
        self.context = float(context)
        self.cache = {}
        self.encoder_calls = 0
        self.metadata = dict(checkpoint=str(checkpoint), sha256=digest, epoch=state['epoch'],
                             architecture=state['architecture'], encoder=self.config['encoder'],
                             condition=list(condition), core_size_m=list(size), context_m=context,
                             inference=dict(encoder_precision=self.encoder.precision, device=str(device)),
                             preprocessing='existing world-voxel first representative + PTv3 prepare + nearest representative',
                             config=self.config)
        del state

    def _clock(self):
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def predict(self, coord, force_full=False, query_range=None, query_indices=None, return_features=False):
        import torch
        from utils.Traversability.data.voxel_hash import representatives, VoxelHash
        from utils.Traversability.training.full3d_cache import encode_context

        coord = np.asarray(coord, np.float32)
        if not len(coord) or not np.isfinite(coord).all():
            raise ValueError('Expected nonempty finite XYZ')
        if query_indices is not None:
            if query_range is not None:
                raise ValueError('Specify query_indices or query_range')
            query_indices = np.asarray(query_indices)
            if (query_indices.ndim != 1 or query_indices.dtype.kind not in 'iu' or not len(query_indices)
                    or query_indices.min() < 0 or query_indices.max() >= len(coord)
                    or len(np.unique(query_indices)) != len(query_indices)):
                raise ValueError('Invalid query indices')
            query = coord[query_indices]
            first_row = last_row = None
        else:
            first_row, last_row = (0, len(coord)) if query_range is None else map(int, query_range)
            if not 0 <= first_row < last_row <= len(coord):
                raise ValueError('Invalid query range')
            query = coord[first_row:last_row]
        started = self._clock()
        with torch.inference_mode():
            raw = torch.from_numpy(np.array(coord, copy=True)).to(self.device)
            points, original, keys, low, shape = representatives(raw, self.config['geometry_voxel_size_m'])
            voxel_s = self._clock() - started
            tick = self._clock()
            table = VoxelHash(points, original, keys, low, shape, self.config['geometry_voxel_size_m'])
            source = torch.empty(len(query), device=self.device, dtype=torch.long)
            for first in range(0, len(query), 262144):
                last = min(first + 262144, len(query))
                query_points = (raw[first_row+first:first_row+last] if query_indices is None else
                                raw[torch.as_tensor(query_indices[first:last].copy(),device=self.device,dtype=torch.long)])
                source[first:last] = table.query(query_points)[0]
            correspondence_s = self._clock() - tick
            del table, keys, raw, original
            prediction = np.empty(len(query), np.float32)
            feature_width = self.head.cost_head[0].in_features
            planning = np.empty((len(query), feature_width), np.float16) if return_features else None
            tick = self._clock()
            groups = partition(query, self.size)
            grouping_s = self._clock() - tick
            encoding_s = head_s = mapping_s = 0.
            calls, hits = 0, 0
            blocks = []
            for key, rows in groups:
                tick = self._clock()
                lower = torch.tensor(np.asarray(key) * self.size, device=self.device)
                upper = lower + torch.tensor(self.size, device=self.device)
                indices = torch.nonzero(((points >= lower - self.context) &
                                        (points < upper + self.context)).all(1)).flatten()
                context_points = points[indices]
                content = context_points.cpu().numpy()
                signature = hashlib.blake2b(content.tobytes(), digest_size=20).hexdigest()
                selected = source[torch.as_tensor(rows, device=self.device)]
                local = torch.searchsorted(indices, selected)
                if not bool((local < len(indices)).all()) or not torch.equal(indices[local], selected):
                    raise ValueError('Nearest query representative outside configured context')
                grouping_s += self._clock() - tick
                cached = self.cache.get(key)
                if (force_full or cached is None or cached[0] != signature
                        or (return_features and cached[2] is None)):
                    tick = self._clock()
                    # Cache every context representative, including ones first queried in later scans.
                    features, back, _ = encode_context(self.encoder, context_points,
                        torch.arange(len(indices), device=self.device), self.config)
                    encoding_s += self._clock() - tick
                    tick = self._clock()
                    batch = torch.zeros(len(features), device=self.device, dtype=torch.long)
                    values = self.head(features, self.condition, batch, return_features=return_features)
                    if return_features:
                        values, conditioned = values
                        context_features = conditioned[back.long(), 0].half().cpu().numpy()
                    else:
                        context_features = None
                    costs = values[back.long(), 0].cpu().numpy()
                    head_s += self._clock() - tick
                    self.cache[key] = (signature, costs, context_features)
                    calls += 1
                    self.encoder_calls += 1
                else:
                    hits += 1
                    costs = cached[1]
                    context_features = cached[2]
                tick = self._clock()
                matched = local.cpu().numpy()
                prediction[rows] = costs[matched]
                if return_features:
                    planning[rows] = context_features[matched]
                mapping_s += self._clock() - tick
                blocks.append(dict(id=block_name(key), signature=signature, points=len(rows),
                                   context_representatives=len(indices)))
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite network prediction')
        report = dict(points=len(query), context_points=len(coord),
            query_range=None if query_indices is not None else [first_row, last_row],
            geometry_points=len(points), blocks=blocks,
            encoder_calls=calls, encoder_calls_total=self.encoder_calls, cache_hits=hits,
            timing=dict(voxel_s=voxel_s, correspondence_s=correspondence_s, grouping_s=grouping_s,
                        encoding_s=encoding_s, head_s=head_s, stitching_s=mapping_s,
                        total_s=self._clock() - started))
        return (prediction, planning, report) if return_features else (prediction, report)


def predict_map(worker,snapshot,anchor=None,inference_frame='map',force_full=False):
    encoding=snapshot
    if inference_frame=='world':
        if anchor is None:raise ValueError('world -> map alignment is not ready')
        translation,quaternion=anchor
        coord=(snapshot.coord@rotation_matrix(quaternion).T+translation).astype(np.float32)
        encoding=Snapshot(snapshot.revision,snapshot.stamp,coord,snapshot.blocks)
    elif inference_frame!='map':raise ValueError('Expected world or map inference frame')
    _,prediction,report=worker.predict(encoding,force_full=True) if force_full else worker.predict(encoding)
    if prediction.shape!=(len(snapshot.coord),):raise ValueError('Whole PCD prediction size mismatch')
    if report.get('context_points',len(encoding.coord))!=len(encoding.coord):raise ValueError('Expected whole PCD context')
    if report.get('query_range',[0,len(encoding.coord)])!=[0,len(encoding.coord)]:raise ValueError('Expected all PCD rows')
    report=dict(report,inference_frame=inference_frame,output_frame='map',input_scope='whole_pcd',force_full=force_full)
    return snapshot,prediction,report


def final_prediction(worker, future, ready, previous, snapshot):
    if future is not None:
        result = future.result()
        if isinstance(result, dict) and result.get('state') == 'ready':
            ready = True
        else:
            previous = result
    if not ready:
        raise RuntimeError('Cost worker was not initialized')
    if len(snapshot.coord) and (previous is None or previous[0].revision != snapshot.revision):
        previous = worker.predict(snapshot)
    if previous is not None:
        geometry, prediction, report = previous
        if (geometry.revision != snapshot.revision or len(prediction) != len(snapshot.coord)
                or not np.array_equal(geometry.coord, snapshot.coord)):
            raise RuntimeError('Final cost does not cover the final map')
        if not np.isfinite(prediction).all():
            raise RuntimeError('Final cost contains nonfinite values')
        if report['revision'] != snapshot.revision:
            raise RuntimeError('Final inference report has the wrong revision')
    return previous


def predict_filtered(worker,filter_,snapshot):
    prepared=filter_.apply(snapshot)
    if len(prepared.coord):return worker.predict(prepared)
    # A completely removed ROI is an empty result, never a stale previous cost.
    timing={k:0. for k in ['voxel_s','correspondence_s','grouping_s','encoding_s','head_s','stitching_s','total_s']}
    report=dict(state='complete',points=0,context_points=len(prepared.context_coord),encoder_calls=0,
                cache_hits=0,blocks=[],timing=timing,transport_timing={},
                revision=prepared.revision,stamp=prepared.stamp,skip_reason='no_query_points_after_knn')
    report.update(prepared.metadata)
    return prepared,np.empty(0,np.float32),report
