"""Content-addressed native libraries; compilation never blocks every query."""
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time


def compile_cached(source, build_folder, name):
    source = Path(source)
    flags = ['-O3', '-std=c++17', '-ffp-contract=off', '-shared', '-fPIC']
    digest = hashlib.sha256(source.read_bytes() + '\0'.join(flags).encode()).hexdigest()
    cache = Path(__file__).resolve().parents[1] / '.cache/native' / digest
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / (name + '.so')
    began = time.perf_counter()
    compiled = False
    with (cache / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
            try:
                subprocess.run(['g++', *flags, str(source), '-o', str(temporary)], check=True)
                os.replace(temporary, library)
            finally:
                temporary.unlink(missing_ok=True)
            compiled = True
    folder = Path(build_folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + '.sha256')).write_text(digest + '\n')
    return library, time.perf_counter() - began, not compiled
