"""Native Isaac Sim viewer of unchanged dense LIO returns and Mesh GT PCD."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import time

import numpy as np

from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', default='lio_error')
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root/args.profile
    output = folder/'isaac_view'
    output.mkdir(exist_ok=True)
    config = read_json(root/'config.actual.json')
    # Load full XYZ once. No original point is removed, moved or merged.
    scan = read_pcd_xyz(folder/'rough.pcd')
    gt = read_pcd_xyz(root/'clean.pcd')
    increments = dict(np.load(folder/'incremental_index.npz'))
    duration = read_json(folder/'generation.json')['frames']/config['scan_hz']
    state = dict(pid=os.getpid(), status='starting', scan_points=len(scan), gt_points=len(gt),
        coordinates='map, metres, Z up', source_scan=str(folder/'rough.pcd'),
        source_gt=str(root/'clean.pcd'), source_positions_unchanged=True, downsampling=False)
    write_json(output/'state.json', state)
    saved_argv, sys.argv = sys.argv, sys.argv[:1]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=False, width=1280, height=900, window_width=1700,
        window_height=1000, multi_gpu=False, renderer='RayTracedLighting', denoiser=False,
        samples_per_pixel_per_frame=1))
    sys.argv = saved_argv
    try:
        import omni.usd
        import omni.ui as ui
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        from pxr import UsdGeom, UsdLux, Gf, Vt
        import trimesh
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(900.)

        def points(path, xyz, color, width):
            item = UsdGeom.Points.Define(stage, path)
            item.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(xyz, dtype=np.float32)))
            item.CreateWidthsAttr([width])
            item.SetWidthsInterpolation('constant')
            item.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            item.GetDisplayColorPrimvar().SetInterpolation('constant')
            return item

        scan_prim = points('/World/SCAN_FasterLIO_Magenta', scan, [.95, .08, .65], .007)
        gt_prim = points('/World/GT_MeshPCD_Green', gt, [.08, .95, .3], .018)
        mesh = trimesh.load(root/'static_mesh.ply', process=False)
        terrain = UsdGeom.Mesh.Define(stage, '/World/StaticReferenceMesh')
        terrain.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, np.float32)))
        terrain.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
        terrain.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.ravel().astype(np.int32)))
        terrain.CreateSubdivisionSchemeAttr('none')
        terrain.CreateDoubleSidedAttr(True)
        terrain.CreateDisplayColorAttr([Gf.Vec3f(.4, .44, .5)])
        terrain.CreateVisibilityAttr('invisible')
        current = dict(seconds=duration, playing=False, dirty=False)
        window = ui.Window('Faster-LIO | scan and GT overlay', width=340, height=400)
        label = None

        def visibility(prim, value):
            UsdGeom.Imageable(prim.GetPrim()).MakeVisible() if value else UsdGeom.Imageable(prim.GetPrim()).MakeInvisible()

        def camera(detail=False):
            eye, target = ([5., 5., 4.], [1., 1.9, 1.]) if detail else ([11., -13., 12.], [0., .3, 1.])
            set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path='/OmniverseKit_Persp')

        def toggle_row(text, prim, enabled):
            with ui.HStack(height=26):
                model = ui.CheckBox(width=24).model
                model.set_value(enabled)
                ui.Label(text)
                model.add_value_changed_fn(lambda m: visibility(prim, m.as_bool))

        def show():
            relative = increments['stamp_s']-config['time_origin_s']
            frame_count = np.searchsorted(relative, current['seconds'], side='right')
            count = int(increments['offsets'][frame_count])
            scan_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(scan[:count]))
            label.text = (f"Elapsed: {current['seconds']:.2f} / {duration:.1f} s\n"
                          f"LIO frames: {frame_count} | scan: {count:,}\n"
                          f"GT: {len(gt):,} | all original points\nMap coordinates, metres, Z up")
            current['dirty'] = False
            state.update(status='open', displayed_scan_points=count, displayed_gt_points=len(gt),
                         elapsed_s=current['seconds'], lio_frames=int(frame_count))
            write_json(output/'state.json', state)

        with window.frame:
            with ui.VStack(spacing=8):
                ui.Label('Magenta: registered scan\nGreen: Mesh-sampled GT', height=46)
                toggle_row('Scan / magenta', scan_prim, True)
                toggle_row('GT PCD / green', gt_prim, True)
                toggle_row('Reference Mesh', terrain, False)
                label = ui.Label('', height=95, word_wrap=True)
                slider = ui.FloatSlider(min=0., max=duration, height=25).model
                slider.set_value(duration)
                slider.add_value_changed_fn(lambda m: current.update(seconds=m.as_float, dirty=True))
                ui.Button('Play / pause incremental scan', height=28,
                          clicked_fn=lambda: current.update(playing=not current['playing']))
                ui.Button('Reset to scan start', height=28, clicked_fn=lambda: slider.set_value(0.))
                ui.Button('Overview camera', height=28, clicked_fn=lambda: camera(False))
                ui.Button('Stair detail camera', height=28, clicked_fn=lambda: camera(True))
        for name in ['Stage', 'Property', 'Content', 'Console']:
            other = ui.Workspace.get_window(name)
            if other is not None:
                other.visible = False
        viewport = ui.Workspace.get_window('Viewport')
        if viewport is not None:
            window.dock_in(viewport, ui.DockPosition.RIGHT, .23)
        camera()
        show()
        stage.Export(str(output/'scan_gt_overlay.usdc'))
        for _ in range(20):
            app.update()
        capture = capture_viewport_to_file(get_active_viewport(), file_path=str(output/'overlay_native.png'))
        completion = asyncio.ensure_future(capture.wait_for_result(0))
        while not completion.done():
            app.update()
        completion.result()
        print('ISAAC_SCAN_OVERLAY_READY', state, flush=True)
        tick = time.monotonic()
        while app.is_running():
            now = time.monotonic()
            if current['playing'] and now-tick >= .1:
                slider.set_value((current['seconds'] + now-tick) % duration)
                tick = now
            elif not current['playing']:
                tick = now
            if current['dirty']:
                show()
            app.update()
            time.sleep(.01)
        state['status'] = 'closed'
        write_json(output/'state.json', state)
    finally:
        app.close()


if __name__ == '__main__':
    main()
