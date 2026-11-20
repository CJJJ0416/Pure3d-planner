"""Compare native and gap-augmented VST on the fixed straight-trench PCD."""

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

VST = Path(__file__).resolve().parents[2]
ROOT = VST.parents[1]
sys.path[:0] = [str(ROOT), str(VST / "src")]
DEFAULT_SCENE = ROOT / "datasets/simulation/gap_max_straight_trenches"
DEFAULT_OUTPUT = VST / "output/gap_max_straight_trenches"
REFERENCE = ROOT / "datasets/cache/vst_native_py310"
LIMITS = [0., .15, .25, .45]
THRESHOLD = 15.

from common.pointcloud_io import read_pcd_xyz, write_pcd
from vst_global_planner.gap_edges import gap_edge_table, grid_centers
from vst_global_planner.tomogram_cloud import tomogram_points


def load(path):
    return json.loads(path.read_text())


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def stats(values):
    return dict(median_ms=float(np.median(values) * 1e3),
                p95_ms=float(np.percentile(values, 95) * 1e3))


def native_build(output):
    native = VST / "native"
    folder = output / "native"
    folder.mkdir(exist_ok=True)
    import sysconfig
    library = folder / ("a_star_gap" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run([
        "g++", "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-DVST_ASTAR_MODULE=a_star_gap",
        "-I/usr/include/eigen3", "-I" + sysconfig.get_path("include"), "-I" + str(native / "src"),
        "-I" + str(native / "3rdparty/pybind11/include"),
        str(native / "src/a_star/a_star_search.cc"),
        str(native / "src/a_star/python_interface.cc"), "-o", str(library),
    ], check=True)
    return folder


def native_planner(payload, folder, module):
    sys.path.insert(0, str(folder))
    astar = importlib.import_module(module).Astar()
    cost, _, _, ground, _ = payload["data"].astype(np.float64)
    same = np.isfinite(ground[1:]) & np.isfinite(ground[:-1]) & (abs(np.diff(ground, axis=0)) < .1)
    delta = np.diff(cost, axis=0)
    gateway = np.zeros_like(cost)
    gateway[:-1][same & (delta < -8)] = 2.
    gateway[1:][same & (delta > 8)] = -2.
    astar.init(THRESHOLD, len(ground), payload["resolution"], 1.,
               cost.reshape(-1, cost.shape[-1]), ground.reshape(-1, ground.shape[-1]),
               gateway.reshape(-1, gateway.shape[-1]))
    return astar


def endpoint(payload, position):
    cost, _, _, ground, _ = payload["data"]
    pos = np.asarray(position)
    cell = np.rint((pos[:2] - payload["center"]) / payload["resolution"]).astype(int)
    i, j = cell + np.array(ground.shape[1:]) // 2
    height, values = ground[:, i, j], cost[:, i, j]
    valid = np.isfinite(height) & np.isfinite(values) & (abs(height - pos[2]) <= payload["slice_dh"])
    layers = np.flatnonzero(valid & (values >= 0.) & (values <= THRESHOLD))
    if not len(layers):
        return None
    selected = min(layers,key=lambda k:(values[k],abs(height[k] - pos[2])))
    return (int(selected), int(i), int(j))


def query(astar, start, goal, repeats):
    if start is None or goal is None:
        return dict(stage="endpoint", index_path=None, search_ms=None)
    times = []
    # Python interface expects layer, column, row at its normal search entrance.
    native_ends = [np.asarray(i,dtype=np.int32)[[0,2,1]] for i in (start,goal)]
    for _ in range(repeats + 1):
        tick = perf_counter()
        success = astar.search(*native_ends)
        times.append(perf_counter() - tick)
    if success:
        path = np.asarray(astar.get_result_matrix(), dtype=int).reshape(-1, 3)
        return dict(stage="complete", index_path=np.vstack((start,path)).tolist(), search_ms=stats(times[1:]))
    return dict(stage="astar", index_path=None, search_ms=stats(times[1:]))


def reference(payload, trial, output, repeats):
    astar = native_planner(payload, REFERENCE, "a_star")
    by_trench = {name: query(astar, row["start"], row["goal"], repeats)
                 for name,row in trial.items()}
    dump(output / "baseline.json", by_trench)


def result_metrics(result, payload, selected_edges):
    if result["index_path"] is None:
        return dict(result, length_m=None, used_edges=[])
    path = np.asarray(result["index_path"])
    ground = payload["data"][3]
    x, y = grid_centers(payload)
    points = np.column_stack((x[path[:, 1]],y[path[:, 2]],ground[tuple(path.T)]))
    used = [selected_edges[(tuple(a),tuple(b))] for a,b in zip(path,path[1:])
            if (tuple(a),tuple(b)) in selected_edges]
    return dict(result, length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                path_xyz=points.tolist(), used_edges=used)


def trial_queries(astar, queries, payload, records, repeats):
    return {name: result_metrics(query(astar, row["start"], row["goal"], repeats), payload, records)
            for name, row in queries.items()}


def plot_result(payloads, scenario, sweep, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = grid_centers(payloads[0])
    r = payloads[0]["resolution"]
    extent = (x[0] - r / 2, x[-1] + r / 2, y[0] - r / 2, y[-1] + r / 2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained", sharex=True, sharey=True)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#dbe2e9")
    for ax, trial, payload in zip(axes.flat, sweep, payloads):
        layer = trial["platform_layer"]
        cost = payload["data"][0, layer]
        ground = payload["data"][3, layer]
        mask = ~np.isfinite(ground) | (np.abs(ground - scenario["top_z_m"]) > .05)
        im = ax.imshow(np.ma.masked_where(mask, cost).T, origin="lower", extent=extent,
                       cmap=cmap, vmin=0, vmax=50)
        routes = trial["gap_aware"]
        count = sum(route["stage"] == "complete" for route in routes.values())
        ax.set_title(f"gap_max = {trial['gap_max_m']:.2f} m   ·   {count}/4 routes")
        for trench in scenario["trenches"]:
            key = trench["id"]
            route = routes[key]
            left, right = trench["x_span_m"]
            ax.text((left + right) / 2, 1.7, key, va="top", ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.85))
            if route["stage"] == "complete":
                path = np.asarray(route["path_xyz"])
                ax.plot(path[:, 0], path[:, 1], lw=1.8, c="#23394c")
                for edge in route["used_edges"]:
                    indices = np.asarray(edge["anchors"])
                    ax.plot(x[indices[:, 1]], y[indices[:, 2]], lw=3.4, c="#0468c8")
            else:
                goal = trench["goal_support_xyz"]
                ax.plot(goal[0], goal[1], marker="x", ms=6, c="#221d31")
        ax.set(xlabel="Map X (m)", ylabel="Map Y (m)")
    fig.colorbar(im, ax=axes, shrink=.68, label="VST trav_cost (raw units)")
    fig.suptitle("CUDA gap costs | native VST A* on four straight trenches", fontsize=16, fontweight="bold")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def trench_for_pair(x, row, scenario):
    _, a, _, b, _ = row
    left, right = sorted((x[a], x[b]))
    return next(t["id"] for t in scenario["trenches"]
                if t["start_support_xyz"][0] < left < right < t["goal_support_xyz"][0])


def rim_costs(trench, payload, platform_layer, baseline):
    x, y = grid_centers(payload)
    cost = payload["data"][0, platform_layer]
    ground = payload["data"][3, platform_layer]
    j = int(np.argmin(np.abs(y)))
    widths = payload.get("gap_pairs", {"cells": [], "widths_m": []})
    a, b = trench["x_span_m"]
    rows = [(row, width) for row, width in zip(widths["cells"], widths["widths_m"])
            if row[0] == platform_layer and row[2] == j and row[4] == j and x[row[1]] < (a+b)/2 < x[row[3]]]
    visible = np.isfinite(ground[:, j]) & (abs(ground[:, j] - trench["start_support_xyz"][2]) < .05)
    if rows:
        row, width = rows[0]
        takeoff, landing = int(row[1]), int(row[3])
    else:
        takeoff = int(np.flatnonzero(visible & (x < (a+b)/2))[-1])
        landing = int(np.flatnonzero(visible & (x > (a+b)/2))[0])
        width = None
    return dict(
        x_m=x[[takeoff, landing]].tolist(),
        cost_before=baseline["data"][0, platform_layer, [takeoff, landing], j].astype(float).tolist(),
        cost_after=cost[[takeoff, landing], j].astype(float).tolist(),
        cuda_width_m=None if width is None else float(width),
    )


def benchmark(scene, output, repeats):
    from third_party_lib.vst_global_planner.tomography.tomography import Tomography

    config = load(VST / "configs/teacher_building.json")
    config["trav"].update(step_max=.20, standable_ratio=0.)
    scenario = load(scene / "scene.json")
    pcd = scene / "terrain.pcd"
    points = read_pcd_xyz(pcd)

    def build_map(limit):
        config["trav"]["gap_max"] = limit
        cfg = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in config.items()})
        folder = output / f"cuda_gap_{int(round(limit * 100)):02d}"
        tick = perf_counter()
        instance = Tomography(None, cfg, pcd, folder / "tomogram.pickle")
        build_s = perf_counter() - tick
        with (folder / "tomogram.pickle").open("rb") as stream:
            payload = pickle.load(stream)
        cloud = tomogram_points(dict(payload, cost_scale=cfg.trav.cost_barrier))
        write_pcd(folder / "tomogram.pcd", cloud[:, :3], cloud[:, 3])
        return payload, dict(wall_s=build_s, stage_ms=instance.timing_ms), len(cloud)

    baseline, baseline_timing, count = build_map(0.)
    original = output / "tomogram.pickle"
    previous_map_available = original.exists()
    if previous_map_available:
        with original.open("rb") as stream:
            previous = pickle.load(stream)
        assert np.array_equal(previous["data"], baseline["data"], equal_nan=True)
    with original.open("wb") as stream:
        pickle.dump(baseline, stream)
    queries = {
        t["id"]: dict(start=endpoint(baseline, t["start_support_xyz"]), goal=endpoint(baseline, t["goal_support_xyz"]))
        for t in scenario["trenches"]
    }
    dump(output / "endpoints.json", queries)

    folder = native_build(output)
    subprocess.run([sys.executable, __file__, "--reference-only", "--output", str(output),
                    "--repeats", str(repeats)], check=True)
    old = load(output / "baseline.json")
    layer = queries["T1"]["start"][0]
    base_astar = native_planner(baseline, folder, "a_star_gap")
    payloads, sweep = [], []
    x, _ = grid_centers(baseline)
    expected = [set(), {"T1"}, {"T1", "T2"}, {"T1", "T2", "T3"}]
    for limit, accepted in zip(LIMITS, expected):
        payload, map_timing, _ = (baseline, baseline_timing, count) if limit == 0 else build_map(limit)
        payloads.append(payload)
        assert np.array_equal(baseline["data"][3:], payload["data"][3:], equal_nan=True)
        unchanged = ~np.isfinite(baseline["data"][3]) | (np.abs(baseline["data"][3] - scenario["top_z_m"]) > .05)
        assert np.array_equal(baseline["data"][0][unchanged], payload["data"][0][unchanged])
        pairs = payload.get("gap_pairs", {"cells": [], "widths_m": []})
        detected = {trench_for_pair(x, row, scenario) for row in pairs["cells"]}
        assert detected == accepted, (limit, detected)
        astar = native_planner(payload, folder, "a_star_gap")
        cost_only = trial_queries(astar, queries, payload, {}, repeats)
        times = []
        for _ in range(repeats + 1):
            tick = perf_counter()
            edges, weights, records = gap_edge_table(payload, THRESHOLD, config["trav"]["interval_min"])
            astar.set_gap_edges(edges, weights)
            times.append(perf_counter() - tick)
        combined = trial_queries(astar, queries, payload, records, repeats)
        edges_only, weights_only, records_only = gap_edge_table(
            payload, THRESHOLD, config["trav"]["interval_min"], cost_map=baseline["data"][0]
        )
        base_astar.set_gap_edges(edges_only, weights_only)
        edge_only = trial_queries(base_astar, queries, baseline, records_only, repeats)
        if limit == 0:
            for name, result in combined.items():
                assert result["stage"] == old[name]["stage"] and result["index_path"] == old[name]["index_path"]
        assert {name for name, route in combined.items() if route["stage"] == "complete"} == accepted
        assert all(edge["width_m"] <= limit + 1e-6 for route in combined.values() for edge in route["used_edges"])
        sweep.append(dict(
            gap_max_m=limit, platform_layer=layer, map_timing=map_timing,
            changed_cost_cells=int(np.count_nonzero(payload["data"][0] != baseline["data"][0])),
            gap_pairs=len(pairs["cells"]), detected_trenches=sorted(detected), directed_edges=len(edges),
            edge_table_ms=stats(times[1:]), cost_only=cost_only, edges_only=edge_only, gap_aware=combined,
            rims={t["id"]: rim_costs(t, payload, layer, baseline) for t in scenario["trenches"]},
        ))
        print("gap_max", limit, "pair_count", len(pairs["cells"]), "routed", sorted(accepted), flush=True)

    config["trav"].pop("gap_max")
    report = dict(
        input_pcd=str(pcd), input_points=len(points), tomogram_points=count,
        input_pcd_sha256=hashlib.sha256(pcd.read_bytes()).hexdigest(),
        native_cost_threshold=THRESHOLD, config=config, reference_matches_gap_disabled=True,
        native_zero_data_identical_to_previous_map=previous_map_available,
        lower_and_unknown_cost_unchanged=True,
        repeats=repeats, sweep=sweep,
    )
    dump(output / "summary.json", report)
    with (output / "routes.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gap_max_m", "mode", "trench", "status", "route_m",
                         "used_crossings", "astar_median_ms", "astar_p95_ms"])
        for trial in sweep:
            for mode in ("cost_only", "edges_only", "gap_aware"):
                for key, route in trial[mode].items():
                    timing = route["search_ms"] or {}
                    writer.writerow([trial["gap_max_m"], mode, key, route["stage"], route["length_m"],
                                     len(route["used_edges"]), timing.get("median_ms"), timing.get("p95_ms")])
    plot_result(payloads, scenario, sweep, output)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,default=DEFAULT_SCENE)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--reference-only",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    if args.reference_only:
        with (output/"tomogram.pickle").open("rb") as stream:
            reference(pickle.load(stream),load(output/"endpoints.json"),output,args.repeats)
    else:
        from third_party_lib.vst_global_planner.tomography.config.scene import SceneTrav

        if not hasattr(SceneTrav, "gap_max"):
            parser.error("This benchmark requires the experimental gap-cost generator. "
                         "The original VST cost pipeline is active.")
        output.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR",str(output/"cupy_cache"))
        os.environ.setdefault("MPLCONFIGDIR",str(output/"mpl_cache"))
        benchmark(args.scene.resolve(),output,args.repeats)


if __name__ == "__main__":
    main()
