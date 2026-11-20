"""Streaming hashes and atomic JSON for data/checkpoint metadata."""
import hashlib
import json
from pathlib import Path
import tempfile

from ptv3_tomography.paths import asset_path, relocated_path


def file_sha256(path):
    digest = hashlib.sha256()
    with relocated_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(relocated_path(path).read_text())


def atomic_json(path, value):
    """Close then atomically replace metadata without forcing synchronous disk I/O."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
