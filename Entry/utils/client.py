"""Blocking pipe client, used only from Entry's background inference thread."""
from Entry.utils.paths import ROOT
import json
import os
from pathlib import Path
import subprocess
import select
import time
import numpy as np
from Entry.utils.io import atomic_json, atomic_npy


class WorkerClient:
    def __init__(self, config, config_path, directory):
        self.directory = Path(directory)
        self.jobs = self.directory / 'jobs'
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / 'worker.log').open('w')
        self.process = subprocess.Popen([config['network_python'], '-u', '-m', 'Entry.predict',
            '--config', str(config_path)], cwd=ROOT,
            env=dict(os.environ, PYTHONUNBUFFERED='1'), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            # Terminal Ctrl+C belongs to the ROS receiver. Keep the worker
            # alive until close() drains and predicts the final snapshot.
            start_new_session=True)

    def response(self):
        if not select.select([self.process.stdout], [], [], 180)[0]:
            self.process.terminate()
            raise RuntimeError('Cost worker timed out after 180 seconds')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Cost worker exited ({self.process.poll()}); see {self.directory / "worker.log"}')
        value = json.loads(line)
        if value['state'] == 'error':
            raise RuntimeError(value['error'])
        return value

    def initialize(self):
        value = self.response()
        atomic_json(self.directory / 'model.json', value['model'])
        return value

    def predict(self, snapshot, query_range=None, input_path=None, force_full=False, cache_namespace=None):
        suffix = '' if query_range is None else '_'+('_'.join(map(str, query_range)))
        stem = self.jobs / f'r{snapshot.revision}{suffix}'
        source, dest, report = stem.with_suffix('.input.npy'), stem.with_suffix('.cost.npy'), stem.with_suffix('.json')
        query_path = stem.with_suffix('.query.npy')
        started = time.perf_counter()
        context = getattr(snapshot, 'context_coord', snapshot.coord)
        query_indices = getattr(snapshot, 'query_indices', None)
        if query_indices is not None and query_range is not None:
            raise ValueError('Context snapshots cannot also specify query_range')
        if input_path is None:
            atomic_npy(source, context)
        request = dict(op='predict', revision=snapshot.revision, stamp=snapshot.stamp,
                       input=str(source if input_path is None else input_path), output=str(dest), report=str(report))
        if query_range is not None:
            request['query_range'] = list(query_range)
        if query_indices is not None:
            atomic_npy(query_path, query_indices)
            request['query_indices'] = str(query_path)
            request['context_metadata'] = snapshot.metadata
        saved = time.perf_counter()
        request['force_full']=bool(force_full)
        if cache_namespace is not None:
            request['cache_namespace'] = cache_namespace
        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            value = self.response()
            response_ready = time.perf_counter()
            if value['revision'] != snapshot.revision:
                raise RuntimeError('Worker returned a different map revision')
            prediction = np.load(dest)
            expected = len(snapshot.coord) if query_range is None else query_range[1]-query_range[0]
            if prediction.shape != (expected,):
                raise RuntimeError('Cost result has a different point count')
            value['transport_timing'] = dict(input_write_s=saved-started,
                worker_roundtrip_s=response_ready-saved, output_read_s=time.perf_counter()-response_ready)
            return snapshot, prediction, value
        finally:
            for path in (source, dest, report, query_path):
                path.unlink(missing_ok=True)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"quit"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.stdout.close()
        self.log.close()
