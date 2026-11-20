#!/usr/bin/env python3
"""Show current training-scene LiDAR and native VST supervision in Isaac Sim."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from ptv3_tomography.data.lidar_reference_pairs import LidarLabelIndex, load_lidar_config, query_lidar_labels
from ptv3_tomography.data.tomography_pairs import load_tomography
from ptv3_tomography.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.captures.read_text())
    scene_path = Path(manifest['scene'])
    payload, data, scene, _ = load_tomography(scene_path)
    cfg = load_lidar_config()
    source_index = LidarLabelIndex(data, payload, cfg['cost_scale'])
    episodes = manifest['captures']
    for episode in episodes:
        episode['report'] = json.loads((Path(episode['output']) / 'report.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'labels').mkdir(exist_ok=True)
    cache = {}

    def load(region, frame):
        key = (region, frame)
        if key not in cache:
            episode = episodes[region]
            with np.load(Path(episode['output']) / f'scan_{frame:05d}.npz', allow_pickle=False) as archive:
                scan = {k: archive[k].copy() for k in archive.files}
            labels = query_lidar_labels(data, payload, scan['world_from_sensor'], cfg, source_index=source_index)
            labels.update(frame_index=np.array(frame), time_start_s=scan['time_start_s'], time_end_s=scan['time_end_s'])
            np.savez_compressed(args.output / 'labels' / f'{episode["region_id"]}_frame_{frame:05d}.npz', **labels)
            cache[key] = scan, labels
        return cache[key]

    argv, sys.argv = sys.argv, sys.argv[:1]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=False, width=1440, height=900, window_width=1800,
                             window_height=1000, multi_gpu=False, renderer='RayTracedLighting',
                             denoiser=False, samples_per_pixel_per_frame=1))
    sys.argv = argv
    try:
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, UsdGeom, UsdLux, Vt
        import trimesh
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(root.GetPrim())
        mesh = trimesh.load(scene_path / 'terrain.obj', force='mesh', process=False)
        terrain = UsdGeom.Mesh.Define(stage, '/World/TrainingTerrain')
        terrain.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
        terrain.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        terrain.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
        terrain.CreateSubdivisionSchemeAttr('none')
        terrain.CreateDoubleSidedAttr(True)
        terrain.CreateDisplayColorAttr([Gf.Vec3f(.43, .47, .52)])
        terrain.CreateDisplayOpacityAttr([.35])
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(900.)

        gt = UsdGeom.Points.Define(stage, '/World/TrainingSupervision')
        gt.CreatePointsAttr(); gt.CreateDisplayColorAttr(); gt.CreateWidthsAttr([.085])
        gt.SetWidthsInterpolation('constant')
        gt.GetDisplayColorPrimvar().SetInterpolation('vertex')
        raw = UsdGeom.Points.Define(stage, '/World/RawLidar')
        raw.CreatePointsAttr(); raw.CreateWidthsAttr([.035]); raw.SetWidthsInterpolation('constant')
        raw.CreateDisplayColorAttr([Gf.Vec3f(.05, .85, 1.)])
        origins = UsdGeom.Points.Define(stage, '/World/LidarOrigins')
        origins.CreatePointsAttr(); origins.CreateWidthsAttr([.20])
        origins.SetWidthsInterpolation('constant'); origins.CreateDisplayColorAttr([Gf.Vec3f(1., 1., 1.)])
        current = dict(region=0, frame=0, band=-1, mode=0, cloud=True, gt=True, dirty=True, camera=True)
        stats_label, frame_control = None, None
        state = dict(pid=os.getpid(), status='starting', scene=str(scene_path), seed=scene['seed'])

        def change(name, value):
            current[name] = value
            current['dirty'] = True
            if name in ('region', 'frame'):
                current['camera'] = True
            if name == 'region':
                current['frame'] = len(episodes[value-1]['report']['frames'])//2 if value else 0
                frame_control.rebuild()

        def build_frames():
            if current['region'] == 0:
                ui.Label('Overview: one midpoint scan per training region', height=25)
            else:
                count = len(episodes[current['region']-1]['report']['frames'])
                model = ui.ComboBox(current['frame'], *[f'Frame {i}' for i in range(count)], height=25).model
                model.add_item_changed_fn(lambda m, _: change('frame', m.get_item_value_model().as_int))

        def show():
            chosen = [(i, len(e['report']['frames'])//2) for i, e in enumerate(episodes)] if current['region'] == 0 else [(current['region']-1, current['frame'])]
            points, colors, returns, positions, detail = [], [], [], [], []
            for region, frame in chosen:
                scan, label = load(region, frame)
                mask = label['mask_cost' if current['mode'] == 0 else 'mask_height'].copy()
                if current['band'] >= 0:
                    mask[np.arange(len(mask)) != current['band']] = False
                points.append(label['surface_points_world_m'][mask])
                cost = label['cost_raw'][mask]
                if current['mode'] == 0:
                    ratio = cost / cfg['cost_scale']
                    color = np.column_stack((np.minimum(2*ratio, 1), np.minimum(2*(1-ratio), 1), np.full(len(ratio), .06)))
                else:
                    ratio = (label['ground_height_gravity_m'][mask]-cfg['height_bin_edges_m'][0]) / (cfg['height_bin_edges_m'][-1]-cfg['height_bin_edges_m'][0])
                    color = np.column_stack((ratio, .25+.65*ratio, 1.-.85*ratio))
                colors.append(color)
                returns.append(scan['points_world_m'][scan['hit_mask']])
                positions.append(scan['world_from_sensor'][:3, 3])
                detail.append(dict(region=episodes[region]['region_id'], frame=frame, supervised_cells=int(mask.sum()),
                                   raw_valid_returns=int(scan['hit_mask'].sum()), cost60_cells=int((cost == 60).sum())))
            xyz = np.concatenate(points).astype(np.float32)
            gt.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz))
            gt.GetDisplayColorAttr().Set(Vt.Vec3fArray.FromNumpy(np.concatenate(colors).astype(np.float32)))
            raw.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.concatenate(returns).astype(np.float32)))
            origins.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(positions, np.float32)))
            (gt.MakeVisible if current['gt'] else gt.MakeInvisible)()
            (raw.MakeVisible if current['cloud'] else raw.MakeInvisible)()
            if current['camera']:
                if current['region'] == 0:
                    target = mesh.bounds.mean(0)
                    eye = target + max(mesh.extents[:2])*np.array([.8, -1., .8])
                else:
                    target = np.mean(positions, axis=0) + [0., 0., -.3]
                    eye = target + [7., -9., 6.]
                set_camera_view(eye=eye, target=target, camera_prim_path='/OmniverseKit_Persp')
            state.update(status='open', region='all' if not current['region'] else episodes[current['region']-1]['region_id'],
                         displayed_supervision_cells=len(xyz), raw_valid_returns=sum(d['raw_valid_returns'] for d in detail),
                         mode='GT cost' if current['mode'] == 0 else 'GT gravity-relative ground height', band=current['band'],
                         frames=detail, mask_definition='mask_cost = mask_height = target_available')
            stats_label.text = (f'{len(detail)} scan(s) | {len(xyz):,} supervised cells\n'
                                f'{state["raw_valid_returns"]:,} original valid LiDAR returns\n'
                                f'{sum(d["cost60_cells"] for d in detail):,} cells at cost 60')
            atomic_json(args.output / 'viewer_state.json', state)
            current.update(dirty=False, camera=False)

        window = ui.Window('Training data | native VST supervision', width=480, height=440)
        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label('Nine-region training map | freshly collected scans', height=26)
                region_model = ui.ComboBox(0, 'All training regions', *[e['region_id'] for e in episodes], height=26).model
                region_model.add_item_changed_fn(lambda m, _: change('region', m.get_item_value_model().as_int))
                frame_control = ui.Frame(height=28)
                frame_control.set_build_fn(build_frames)
                mode_model = ui.ComboBox(0, 'GT cost', 'GT ground height relative to LiDAR', height=26).model
                mode_model.add_item_changed_fn(lambda m, _: change('mode', m.get_item_value_model().as_int))
                edges = cfg['height_bin_edges_m']
                band_model = ui.ComboBox(0, 'All height bands', *[f'Band {b}: [{lo:g}, {hi:g}) m' for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))], height=26).model
                band_model.add_item_changed_fn(lambda m, _: change('band', m.get_item_value_model().as_int-1))
                for name, text in (('gt', 'Show supervision grid'), ('cloud', 'Show original LiDAR returns (cyan)')):
                    with ui.HStack(height=25):
                        model = ui.CheckBox(width=24).model
                        model.set_value(True)
                        model.add_value_changed_fn(lambda m, key=name: change(key, m.as_bool))
                        ui.Label(text)
                ui.Label('Cost: green 0 / yellow 30 / red 60', height=24)
                ui.Label('mask_cost = mask_height = target_available', height=24)
                stats_label = ui.Label('', height=70, word_wrap=True)
                ui.Button('Reset camera', height=27, clicked_fn=lambda: current.update(camera=True, dirty=True))
        for name in ('Stage', 'Property', 'Content', 'Console'):
            other = ui.Workspace.get_window(name)
            if other is not None:
                other.visible = False
        viewport = ui.Workspace.get_window('Viewport')
        if viewport is not None:
            window.dock_in(viewport, ui.DockPosition.RIGHT, .30)
        show()
        for _ in range(40):
            app.update()
        output = args.output / 'training_supervision_overview_native.png'
        handle = capture_viewport_to_file(get_active_viewport(), file_path=str(output))
        completion = asyncio.ensure_future(handle.wait_for_result(0))
        while not completion.done():
            app.update()
            time.sleep(.01)
        completion.result()
        import omni.renderer_capture
        omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        stage.Export(str(args.output / 'training_supervision.usda'))
        atomic_json(args.output / 'report.json', dict(state, renderer='Isaac Sim native viewport', capture=str(output),
                    supervision_source='production query_lidar_labels', captured_frames=sum(len(e['report']['frames']) for e in episodes)))
        print('TRAINING_SUPERVISION_READY ' + json.dumps(state), flush=True)
        while app.is_running():
            if current['dirty']:
                show()
            app.update()
            time.sleep(.03)
        state['status'] = 'closed'
        atomic_json(args.output / 'viewer_state.json', state)
    finally:
        app.close()


if __name__ == '__main__':
    main()
