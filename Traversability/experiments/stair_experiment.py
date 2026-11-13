"""Run the authorised A/B/C stair experiment, select on validation, continue to epoch 50."""
import copy
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.pointcloud import ROOT, read_json, write_json
from Traversability.experiments.stair_metrics import checkpoint_key

RUN = ROOT / "runs/stair_stand"
DATA = ROOT / "datasets/wfc_stair_stand_v1"
INITIAL = ROOT / "runs/traversability/best.pt"
GEOMETRY_PYTHON = "/home/cjh/glocal2-nav/.venv/bin/python"
VERSIONS = ["A_replay", "B_geometry", "C_geometry_conditions"]


def main():
    global RUN, DATA, INITIAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    RUN, DATA, INITIAL = args.output, args.data, args.initial
    RUN.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["MPLCONFIGDIR"] = str(ROOT / "datasets/cache/matplotlib")
    jobs = []

    def run(name, arguments, directory=RUN, python=sys.executable):
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print("START", name, flush=True)
        with (directory / f"{name}.log").open("a") as log:
            process = subprocess.Popen([python, *map(str, arguments)], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT)
            write_json(RUN / "execution.json", dict(complete=False, active=name, pid=process.pid, jobs=jobs))
            status = process.wait()
        jobs.append(dict(name=name, directory=str(directory), seconds=time.perf_counter()-started, status=status))
        write_json(RUN / "execution.json", dict(complete=False, active=None, jobs=jobs))
        if status:
            raise subprocess.CalledProcessError(status, process.args)
        print("DONE", jobs[-1], flush=True)

    write_json(RUN / "protocol.json", dict(
        initial_checkpoint=str(INITIAL),
        data=str(DATA), geometry_families=dict(train=8,val=3,test=3,variants=3),
        versions=VERSIONS, screening_epochs=20, scheduler_epochs=50, total_epochs=50,
        batch_size=2, blocks_per_epoch=342, stair_blocks=206, replay_blocks=136,
        validation_every_epochs=5, teacher_conditions_per_step=40, condition_banks=3,
        teacher_random_ranges=dict(clearance_min=[.2,1.],slope_max=[.18,.65],step_max=[.08,.4],stand_ratio=[.05,.5]),
        condition_difference_weight=1., response_weights=[1,2,1,2], geometry_seed_start=20262100,
        selection="eligible retention, then stair break rate, stand response MAE, global MAE",
        evaluation_endpoint_revision=2,
        endpoint_definition="fixed geometric pairs plus GT-connected components spanning at least 60% of stair height",
        test_use="only after validation selection", model_structure="original single FiLM",
        plan=str(ROOT/"Traversability/docs/STAIR_STAND_RATIO_FINETUNING_PLAN.md")))
    write_json(RUN / "execution.json", dict(complete=False, active="data_preparation", jobs=[]))
    while not (DATA / "data_ready.json").exists():
        time.sleep(5)
    run("prepare_connectivity", ["-m", "Traversability.experiments.stair_metrics", "--prepare", DATA/"evaluation"], python=GEOMETRY_PYTHON)
    base_args = ["-m", "Traversability.training.evaluate", "--data", DATA/"evaluation", "--split", "val"]
    run("initial_val", [*base_args, "--checkpoint", INITIAL, "--output", RUN/"initial_val.json",
                        "--predictions", RUN/"initial_val.npz"])
    baseline = read_json(RUN / "initial_val.json")
    for label, name in zip(["A", "B", "C"], VERSIONS):
        folder = RUN/name
        config = copy.deepcopy(read_json(ROOT/"Traversability/configs/train_response.json"))
        config.update(data=str(DATA/label), output=str(folder), validation_data=str(DATA/"evaluation"),
                      epochs=50, learning_rate=1e-4, encoder_learning_rate=1e-5,
                      evaluation_every_epochs=5, stair_selection_baseline=str(RUN/"initial_val.json"),
                      stair_sampling=dict(stair_blocks=206,source="replay" if label=="A" else "new"))
        write_json(folder/"train_config.json", config)
        if (folder/"last.pt").exists():
            import torch
            saved=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
            complete=saved["epoch"] >= 20
            del saved
            if complete:
                continue
            weights=["--resume",folder/"last.pt"]
        else:
            weights=["--config",folder/"train_config.json","--initialize",INITIAL]
        run("train_01_20", ["-m", "Traversability.training.train", *weights, "--stop-after-epoch", 20], folder)
    selection = {}
    for name in VERSIONS:
        history = read_json(RUN/name/"history.json")
        best = min((r for r in history if r["validation"] is not None),
                   key=lambda r:checkpoint_key(r["validation"],baseline))
        selection[name] = dict(epoch=best["epoch"], key=checkpoint_key(best["validation"],baseline),
                               metrics=best["validation"])
    winner = min(VERSIONS, key=lambda n:tuple(selection[n]["key"]))
    write_json(RUN/"validation_selection.json",dict(winner=winner, selected_before_test=True, versions=selection))
    run("train_21_50", ["-m","Traversability.training.train","--resume",RUN/winner/"last.pt"], RUN/winner)
    run("final_val", [*base_args,"--checkpoint",RUN/winner/"best.pt","--output",RUN/"final_val.json",
                      "--predictions",RUN/"final_val.npz"])
    for name, checkpoint in [("original",INITIAL),("selected",RUN/winner/"best.pt")]:
        run(f"{name}_test",["-m","Traversability.training.evaluate","--data",DATA/"evaluation","--split","test",
                             "--checkpoint",checkpoint,"--output",RUN/f"{name}_test.json",
                             "--predictions",RUN/f"{name}_test.npz"])
    regression=ROOT/"datasets/wfc_regression_20261127"
    run("known_stair_initial",["-m","Traversability.training.evaluate","--data",regression,"--split","test",
                               "--checkpoint",INITIAL,"--output",RUN/"known_stair_initial.json",
                               "--predictions",RUN/"known_stair_initial.npz"])
    run("known_stair_regression",["-m","Traversability.training.evaluate","--data",regression,"--split","test",
                                  "--checkpoint",RUN/winner/"best.pt","--output",RUN/"known_stair_regression.json",
                                  "--predictions",RUN/"known_stair_regression.npz"])
    for family in [11,12,13]:
        seed=(20262100+family)*10+1
        scene=DATA/"scenes"/f"scene_{seed}"
        run(f"predict_{seed}",["-m","Traversability.inference.predict","--checkpoint",RUN/winner/"best.pt",
                                "--pcd",scene/"terrain_downsampled.pcd","--geometry-python",GEOMETRY_PYTHON,
                                "--condition",.5,.6,.4,.05,"--output",RUN/"predictions"/scene.name])
    run("report",["-m","Traversability.experiments.stair_report","--run",RUN],python=GEOMETRY_PYTHON)
    write_json(RUN/"execution.json",dict(complete=True,winner=winner,jobs=jobs))
    print("EXPERIMENT COMPLETE",winner,flush=True)


if __name__ == "__main__":
    main()
