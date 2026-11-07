#!/usr/bin/env python3
"""Build a headless PTv3 source/data/weight bundle without local tests or run artifacts."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PTV3 = ROOT.parents[1] / 'third_party/PointTransformerV3'
sys.path.insert(0,str(ROOT/'src'))
from ptv3_tomography.io import atomic_json,file_sha256


def write_manifest(output):
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name not in {'bundle_manifest.json','SHA256SUMS','TRANSFER_STATUS.json'}:
            files[path.relative_to(output).as_posix()] = file_sha256(path)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=PTV3,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=PTV3,text=True).strip())
    atomic_json(output/'bundle_manifest.json',{'status':'complete','source_repository_commit':revision,
                'source_repository_dirty_at_packaging':dirty,'files':files,
                'data_and_weights_separate_from_git':True,'single_gpu_training':True,
                'target_gpu':'H100', 'source_layout':'src/ptv3_tomography + third_party/PointTransformerV3'})
    (output/'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in files.items()))
    atomic_json(output/'TRANSFER_STATUS.json',{'status':'complete','files':len(files),
                'runtime_dataset_paths_relative':True,'git_commit':revision})
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dataset',type=Path,required=True)
    parser.add_argument('--dev-dataset',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    atomic_json(output/'TRANSFER_STATUS.json',{'status':'building'})
    code = output/'third_party/PointTransformerV3'
    code.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','*.pending','*.egg-info','outputs','tests','test_results')
    shutil.copytree(PTV3/'src/point_transformer_v3',code/'src/point_transformer_v3',ignore=ignore)
    for name in ['pyproject.toml','README.md','LICENSE','NOTICE','TOMOGRAPHY_UPSTREAM_PIN.json','README.tomography.md']:
        shutil.copyfile(PTV3/name,code/name)
    project = output/'src/ptv3_tomography'
    shutil.copytree(ROOT,project,ignore=ignore)
    entries = []
    for split,dataset,identifier in [('train',args.train_dataset,'ramp_130'),('dev',args.dev_dataset,'stairs_40')]:
        if dataset is None:
            continue
        destination = output/'data'/identifier
        subprocess.run([sys.executable,str(ROOT/'scripts/data/make_dataset_portable.py'),
                        '--dataset',str(dataset.resolve()),'--output',str(destination)],check=True)
        entries.append({'id':identifier,'building_id':identifier,'split':split,
                        'path':os.path.relpath(destination,project/'configs')})
    atomic_json(project/'configs/server_datasets.json',{'version':'single_frame_registry_v1','datasets':entries})
    if args.checkpoint:
        target = output/'weights/initial.pt'
        target.parent.mkdir()
        shutil.copyfile(args.checkpoint,target)
    readme = """# PTv3 training transfer bundle

Start with `src/ptv3_tomography/docs/SERVER_TRAINING.md`.
Install the local packages with:
`python -m pip install --no-build-isolation --no-deps -e third_party/PointTransformerV3 -e src/ptv3_tomography`.
Install `requirements-server.txt` for training and CPU Agg plots. No display is required.

The source, relative-path datasets and optional weights are self-contained for
single-frame offline training. Isaac Sim and ROS are not training dependencies.
The 40-frame stair scene is a development diagnostic. It has already been viewed
locally and is not a pristine final test set. No test split is evaluated by training.

The current native VST experiment rebuilds data and starts model parameters,
optimizer, scheduler and counters from scratch. The main repository manual gives
the complete fresh-start headless campaign commands.
The full headless collection campaign uses the main repository and requirements-capture.txt.
"""
    (output/'README.md').write_text(readme)
    count = write_manifest(output)
    print(f'Server bundle ready: {output} ({count} payload files)',flush=True)


if __name__=='__main__':
    main()
