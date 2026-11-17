"""One resident GPU process; JSON control on pipes, arrays in local NPY files."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback
import resource
import numpy as np
from Entry.utils.io import atomic_json, atomic_npy
from Entry.utils.core import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    wire = sys.stdout

    def send(value):
        print(json.dumps(value, allow_nan=False), file=wire, flush=True)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from Entry.utils.predict import BlockCostPredictor
            import yaml
            if yaml.safe_load(args.config.read_text()).get('input_scope') == 'local_voxel_box':
                from Entry.utils.local_map import load_local_config
                cfg = load_local_config(args.config)
            else:
                cfg = load_config(args.config)
            predictor = BlockCostPredictor(cfg['checkpoint'], cfg['condition'], cfg['core_size_m'],
                                           cfg['context_m'], cfg['device'], cfg['expected_epoch'],
                                           cfg.get('expected_encoder', 'utonia_small'))
        send(dict(state='ready', model=predictor.metadata))
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        send(dict(state='error', error=str(error)))
        return 1
    cache_sets = {'default': predictor.cache}
    for line in sys.stdin:
        request = json.loads(line)
        if request['op'] == 'quit':
            break
        try:
            namespace = request.get('cache_namespace', 'default')
            if not isinstance(namespace,str) or not namespace:
                raise ValueError('Invalid cache namespace')
            predictor.cache = cache_sets.setdefault(namespace,{})
            coord = np.load(request['input'], mmap_mode='r')
            queries = (None if 'query_indices' not in request else
                       np.load(request['query_indices'], mmap_mode='r'))
            with contextlib.redirect_stdout(sys.stderr):
                prediction, report = predictor.predict(coord, force_full=request.get('force_full', False),
                    query_range=request.get('query_range'), query_indices=queries)
            atomic_npy(request['output'], prediction)
            report.update(revision=request['revision'], stamp=request['stamp'],force_full=request.get('force_full',False))
            report.update(request.get('context_metadata', {}))
            report['memory'] = dict(worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_peak_allocated_bytes=predictor.torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=predictor.torch.cuda.max_memory_reserved(),
                scope='worker cumulative peak since initialization')
            if cfg.get('input_scope') == 'local_voxel_box':
                from Entry.utils.core import block_name
                meta = request.get('context_metadata', {})
                if meta.get('retained_core_lower') is not None:
                    lo, hi = meta['retained_core_lower'], meta['retained_core_upper']
                    predictor.cache = {key:value for key,value in predictor.cache.items()
                        if all(lo[i] <= key[i] < hi[i] for i in range(3))}
                else:
                    active = {row['id'] for row in report['blocks']}
                    predictor.cache = {key: value for key, value in predictor.cache.items()
                                       if block_name(key) in active}
            cache_sets[namespace] = predictor.cache
            atomic_json(request['report'], report)
            send(dict(state='complete', **report))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            send(dict(state='error', revision=request['revision'], error=str(error)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
