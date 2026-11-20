#!/usr/bin/env python3
"""Inspect actual trained Plan A maps in the existing Isaac stage, with native comparison UI."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ptv3_tomography.io import atomic_json, file_sha256, read_json


from ptv3_tomography.paths import relocated_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--stage", type=Path, help="Existing USD stage; default loads the captured terrain mesh")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=52)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", type=int, default=0, help="0 prediction cost, 1 GT cost, 2 error, 3/4 height, 5 height error")
    args = parser.parse_args()
    args.maps, args.output = args.maps.resolve(), args.output.resolve()
    args.stage = args.stage.resolve() if args.stage else None
    manifest, plots = read_json(args.maps / "manifest.json"), read_json(args.maps / "plot_index.json")
    anchors = [row["frame_index"] for row in plots["frames"]]
    args.output.mkdir(parents=True, exist_ok=False)
    stage_hash = file_sha256(args.stage) if args.stage else None
    state = {"status": "starting", "pid": os.getpid(), "frame": args.frame,
             "source_stage": str(args.stage), "hardware_limits_imposed": False}
    atomic_json(args.output / "viewer_state.json", state)
    script_argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless, "width": 1280, "height": 800,
                             "window_width": 1800, "window_height": 1000, "multi_gpu": False,
                             "renderer": "RayTracedLighting", "denoiser": False,
                             "samples_per_pixel_per_frame": 1})
    finally:
        sys.argv = script_argv
    try:
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, UsdGeom, UsdLux, Vt
        context = omni.usd.get_context()
        import trimesh
        source_spec = read_json(relocated_path(manifest["source_dataset"]) / "dataset_spec.json")
        scene_dir = relocated_path(source_spec["scene_directory"])
        if not scene_dir.is_absolute():
            scene_dir = relocated_path(manifest["source_dataset"]) / scene_dir
        source_mesh = trimesh.load(scene_dir / "terrain.obj", force="mesh", process=False)
        if args.stage:
            context.open_stage(str(args.stage))
        else:
            context.new_stage()
        for _ in range(12):
            app.update()
        stage = context.get_stage()
        if not args.stage:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            terrain = UsdGeom.Mesh.Define(stage, "/World/Terrain")
            terrain.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(source_mesh.vertices.astype(np.float32)))
            terrain.CreateFaceVertexCountsAttr([3] * len(source_mesh.faces))
            terrain.CreateFaceVertexIndicesAttr(source_mesh.faces.ravel().tolist())
            terrain.CreateSubdivisionSchemeAttr("none")
            terrain.CreateDoubleSidedAttr(True)
            terrain.CreateDisplayColorAttr([Gf.Vec3f(.43, .47, .52)])
            UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(900.)
        stage.SetEditTarget(stage.GetSessionLayer())
        terrain = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Terrain"))
        terrain.CreateDisplayOpacityAttr([.35])
        for path in ("/World/SupervisedCostGreenToRed", "/World/BodyOrigin"):
            prim = stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()
        cloud = UsdGeom.Points.Define(stage, "/World/HistoryLidarCyan")
        cloud.CreatePointsAttr()
        cloud.CreateDisplayColorAttr([Gf.Vec3f(.05, .85, 1.)])
        cloud.CreateWidthsAttr([.025])
        cloud.SetWidthsInterpolation("constant")
        prediction_points = UsdGeom.Points.Define(stage, "/World/PlanA/PredictedSurface")
        prediction_points.CreateWidthsAttr([.085])
        prediction_points.SetWidthsInterpolation("constant")
        prediction_points.CreateDisplayColorAttr()
        prediction_points.GetDisplayColorPrimvar().SetInterpolation("vertex")
        axes = UsdGeom.BasisCurves.Define(stage, "/World/LocalXYAxes")
        axes.CreateTypeAttr("linear")
        axes.CreatePointsAttr()
        axes.CreateWidthsAttr([.015])
        axes.SetWidthsInterpolation("constant")
        axes.CreateCurveVertexCountsAttr([2, 2, 2, 2])
        axes.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, .3, 1), Gf.Vec3f(1, 1, 1)])
        axes.GetDisplayColorPrimvar().SetInterpolation("uniform")
        sensor_origin = UsdGeom.Xform.Define(stage, "/World/SensorOrigin")
        origin_ops = sensor_origin.GetOrderedXformOps()
        origin = origin_ops[0] if origin_ops else sensor_origin.AddTranslateOp()
        cfg, edges = manifest["map_config"], manifest["map_config"]["height_bin_edges_m"]
        modes = ["PTv3 cost", "Tomography cost", "Absolute cost error",
                 "PTv3 ground height", "Tomography ground height", "Absolute height error"]
        current = {"frame": args.frame, "band": -1, "mode": args.mode, "dense": False,
                   "cloud": True, "dirty": True, "camera": True}
        cache, stats_label, image_frame = {}, None, None

        def load(index):
            if index not in cache:
                row = next(row for row in manifest["frames"] if row["frame_index"] == index)
                file = args.maps / row["file"]
                with np.load(file, allow_pickle=False) as a:
                    cache[index] = {k: a[k].copy() for k in a.files}
            return cache[index]

        def plot_row():
            item = next(row for row in plots["frames"] if row["frame_index"] == current["frame"])
            band = item["best_band"] if current["band"] < 0 else current["band"]
            return item["bands"][band]

        def build_image():
            path = args.maps / plot_row()["file"]
            ui.Image(str(path), fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT)

        def change(name, value):
            current[name], current["dirty"] = value, True
            if name == "frame":
                current["camera"] = True

        def show():
            data, mode, band = load(current["frame"]), current["mode"], current["band"]
            tf, up = data["world_from_sensor"], data["gravity_up_sensor"]
            teacher = mode in (1, 4)
            cost_mode = mode < 3
            mask = data["mask_cost" if cost_mode else "mask_height"].copy()
            if current["dense"] and mode in (0, 3):
                mask[:] = True
            if band >= 0:
                mask[np.arange(len(mask)) != band] = False
            height = data["gt_ground_height_gravity_m"] if teacher else data["ground_height_gravity_m"]
            if teacher:
                b, x, y = np.nonzero(mask)
                xx = (x - cfg["grid_size_xy"][0] // 2) * cfg["resolution_m"]
                yy = (y - cfg["grid_size_xy"][1] // 2) * cfg["resolution_m"]
                zz = (height[mask] - up[0] * xx - up[1] * yy) / up[2]
                sensor_points = np.column_stack((xx, yy, zz))
            else:
                sensor_points = data["predicted_surface_sensor_m"][mask]
            world = sensor_points @ tf[:3, :3].T + tf[:3, 3]
            if mode in (0, 1):
                values = data["gt_cost_raw" if teacher else "cost_raw"][mask]
                ratio = np.clip(values / 60., 0, 1)
                colors = np.column_stack((np.minimum(2 * ratio, 1), np.minimum(2 * (1 - ratio), 1), np.full(len(ratio), .06)))
            elif mode in (3, 4):
                ratio = np.clip((height[mask] - edges[0]) / (edges[-1] - edges[0]), 0, 1)
                colors = np.column_stack((ratio, .25 + .65 * ratio, 1. - .85 * ratio))
            else:
                predicted = data["cost_raw" if cost_mode else "ground_height_gravity_m"][mask]
                ground_truth = data["gt_cost_raw" if cost_mode else "gt_ground_height_gravity_m"][mask]
                ratio = np.clip(np.abs(predicted - ground_truth) / (60. if cost_mode else .25), 0, 1)
                colors = np.column_stack((ratio, .12 * (1 - ratio), .9 * (1 - ratio)))
            prediction_points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(world.astype(np.float32)))
            prediction_points.GetDisplayColorAttr().Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
            world_cloud = data["points_sensor_m"] @ tf[:3, :3].T + tf[:3, 3]
            cloud.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(world_cloud.astype(np.float32)))
            (cloud.MakeVisible if current["cloud"] else cloud.MakeInvisible)()
            origin.Set(Gf.Vec3d(*tf[:3, 3]))
            endpoints = np.vstack((tf[:3, 3] + .8 * tf[:3, :3].T, tf[:3, 3] + [0, 0, .8]))
            line_points = np.stack((np.tile(tf[:3, 3], (4, 1)), endpoints), axis=1).reshape(-1, 3)
            axes.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(line_points.astype(np.float32)))
            if current["camera"]:
                set_camera_view(eye=tf[:3, 3] + [4, -5, 3.5], target=tf[:3, 3] + [0, 0, -.25], camera_prim_path="/OmniverseKit_Persp")
            current.update(dirty=False, camera=False)
            row = next(row for row in manifest["frames"] if row["frame_index"] == current["frame"])
            state.update(status="open" if not args.headless else "capturing", frame=current["frame"],
                         mode=modes[mode], band=band, displayed_cells=int(mask.sum()), dense_prediction=current["dense"],
                         cost_mae=row["cost_mae"], height_mae_m=row["height_mae"], plot_file=plot_row()["file"])
            if stats_label is not None:
                stats_label.text = (f"Frame {current['frame']} | {modes[mode]} | {mask.sum()} cells\n"
                                    f"All-band cost MAE {row['cost_mae']:.3f} /60 | height MAE {row['height_mae']:.4f} m\n"
                                    "GT uses every available native VST label. cost: 0 green / 30 yellow / 60 red.")
            if image_frame is not None:
                image_frame.rebuild()
            atomic_json(args.output / "viewer_state.json", state)

        if not args.headless:
            window = ui.Window("Native VST supervision | PTv3 maps", width=850, height=850)
            with window.frame:
                with ui.VStack(spacing=5):
                    ui.Label("Native VST labels | local cost + gravity height", height=25)
                    frame_model = ui.ComboBox(anchors.index(args.frame), *[f"Frame {i}" for i in anchors], height=25).model
                    frame_model.add_item_changed_fn(lambda model, _: change("frame", anchors[model.get_item_value_model().as_int]))
                    band_model = ui.ComboBox(0, "All bands in 3D / strongest supervised band in 2D",
                                             *[f"Band {b}: gravity h in [{lo:g}, {hi:g}) m" for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))], height=25).model
                    band_model.add_item_changed_fn(lambda model, _: change("band", model.get_item_value_model().as_int - 1))
                    mode_model = ui.ComboBox(args.mode, *modes, height=25).model
                    mode_model.add_item_changed_fn(lambda model, _: change("mode", model.get_item_value_model().as_int))
                    with ui.HStack(height=25):
                        dense_model = ui.CheckBox(width=24).model
                        dense_model.add_value_changed_fn(lambda model: change("dense", model.as_bool))
                        ui.Label("Show dense predicted cells (includes unobserved regions)")
                    with ui.HStack(height=25):
                        cloud_model = ui.CheckBox(width=24).model
                        cloud_model.set_value(True)
                        cloud_model.add_value_changed_fn(lambda model: change("cloud", model.as_bool))
                        ui.Label("Cyan: current LiDAR | RGB: sensor axes | white: gravity up")
                    ui.Label("3D cost: green 0 -> yellow 30 -> red 60. Terrain opacity 35%.", height=23)
                    stats_label = ui.Label("", height=62, word_wrap=True)
                    image_frame = ui.Frame()
                    image_frame.set_build_fn(build_image)
                    ui.Label("Grey panels = no supervision. Check all 6 gravity-height bands.", height=22)
                    ui.Button("Reset camera", height=24, clicked_fn=lambda: current.update(camera=True, dirty=True))
            for name in ("Stage", "Property", "Content", "Console"):
                other = ui.Workspace.get_window(name)
                if other is not None:
                    other.visible = False
            viewport_window = ui.Workspace.get_window("Viewport")
            if viewport_window is not None:
                window.dock_in(viewport_window, ui.DockPosition.RIGHT, .51)

        show()
        captures, snapshots = [], []
        mode_names = ["predicted_cost", "gt_cost", "cost_error", "predicted_height", "gt_height", "height_error"]
        for _ in range(32):
            app.update()
        file = args.output / f"frame_{args.frame:05d}_{mode_names[args.mode]}_native.png"
        capture = capture_viewport_to_file(get_active_viewport(), file_path=str(file))
        captures.append(capture)
        completion = asyncio.ensure_future(capture.wait_for_result(0))
        while not completion.done():
            app.update()
            time.sleep(.01)
        completion.result()
        import omni.renderer_capture
        omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        snapshots.append({"mode": mode_names[args.mode], "file": str(file)})
        stage.Export(str(args.output / "plan_a_scene.usda"))
        atomic_json(args.output / "report.json", {"status": "passed", "renderer": "Isaac Sim native viewport",
                    "source_stage_sha256": stage_hash, "source_scene": str(scene_dir),
                    "predicted_3d_surfaces_use_predicted_height": True, "display_height_offset_m": 0,
                    "hardware_limits_imposed": False,
                    "checkpoint_sha256": manifest["checkpoint_sha256"], "captures": snapshots,
                    "current_frame": args.frame, "source_dataset_modified": False,
                    "panel_content": "Tomography / PTv3 / absolute error for cost and physical ground height"})
        print("PLAN_A_VIEWER_READY " + str(state), flush=True)
        if not args.headless:
            while app.is_running():
                if current["dirty"]:
                    show()
                app.update()
                time.sleep(.03)
        state["status"] = "closed"
        atomic_json(args.output / "viewer_state.json", state)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        atomic_json(args.output / "viewer_state.json", state)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
