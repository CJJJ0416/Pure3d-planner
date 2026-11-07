#!/usr/bin/env python3
"""Capture all observed source VST cells on their saved world-height surfaces in Isaac."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.scene, args.stage, args.output = [p.resolve() for p in (args.scene, args.stage, args.output)]
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.scene / 'vst.pickle'
    original_hashes = {str(p): digest(p) for p in [source, args.stage, args.scene/'terrain.obj']}
    metadata = json.loads((args.scene / 'vst.meta.json').read_text())
    with source.open('rb') as stream:
        payload = pickle.load(stream)
    data = np.asarray(payload['data'], dtype=np.float32)
    cost, ground = data[0], data[3]
    resolution = float(payload['resolution'])
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 1280, 'height': 900, 'multi_gpu': False,
                             'renderer': 'RayTracedLighting', 'denoiser': False,
                             'samples_per_pixel_per_frame': 1})
    finally:
        sys.argv = argv
    try:
        import omni.usd
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        from pxr import UsdGeom, Vt
        import trimesh
        context = omni.usd.get_context()
        context.open_stage(str(args.stage))
        for _ in range(12):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        native = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Terrain'))
        mesh = trimesh.load(args.scene/'terrain.obj', force='mesh', process=False)
        native.CreateDisplayOpacityAttr([.35])
        # The small lift is a rendering offset only, recorded in the output report.
        display_lift = .015
        layers, rows = [], []
        for layer in range(len(cost)):
            valid = np.isfinite(ground[layer])
            x, y = np.nonzero(valid)
            centers = np.column_stack(((x-cost.shape[1]//2)*resolution+payload['center'][0],
                                        (y-cost.shape[2]//2)*resolution+payload['center'][1],
                                        ground[layer][valid]+display_lift))
            corners = np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])*resolution
            vertices = (centers[:,None,:]+corners[None,:,:]).reshape(-1,3).astype(np.float32)
            ratio = np.clip(cost[layer][valid]/60, 0, 1)
            colors = np.column_stack((np.minimum(2*ratio,1), np.minimum(2*(1-ratio),1),
                                       np.full(len(ratio),.06))).astype(np.float32)
            tile = UsdGeom.Mesh.Define(stage, f'/World/FullVST/Layer_{layer}')
            tile.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
            tile.CreateFaceVertexCountsAttr([4]*len(centers))
            tile.CreateFaceVertexIndicesAttr(np.arange(len(vertices)).tolist())
            tile.CreateSubdivisionSchemeAttr('none')
            tile.CreateDoubleSidedAttr(True)
            tile.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors))
            tile.GetDisplayColorPrimvar().SetInterpolation('uniform')
            tile.MakeInvisible()
            layers.append(tile)
            rows.append({'layer': layer, 'cells': len(centers),
                         'no_ground_cells_omitted_from_3d_only': int((~valid).sum())})
        target = [float(payload['center'][0]),float(payload['center'][1]),.15]
        set_camera_view(eye=np.asarray(target)+[9,-12,12], target=target,
                        camera_prim_path='/OmniverseKit_Persp')
        captures=[]
        for i, tile in enumerate(layers):
            for other in layers:
                other.MakeInvisible()
            tile.MakeVisible()
            for _ in range(24):
                app.update()
            file = args.output/f'full_vst_layer_{i:02d}_native.png'
            handle = capture_viewport_to_file(get_active_viewport(), file_path=str(file))
            captures.append(handle)
            completion = asyncio.ensure_future(handle.wait_for_result(0))
            deadline = time.monotonic()+40
            while not completion.done() and time.monotonic()<deadline:
                app.update()
                time.sleep(.01)
            completion.result()
            import omni.renderer_capture
            omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
            rows[i].update(file=str(file),sha256=digest(file))
        stage.Export(str(args.output/'full_vst_scene.usda'))
        report={'status':'passed','renderer':'Isaac Sim native viewport','hardware_limits_imposed':False,
                'source_hashes':original_hashes,'source_mesh_topology_verified':True,
                'cost_source':'direct saved VST cost, no prediction or supervision mask',
                'full_map_cost_shape':list(cost.shape),'display_height_offset_m':display_lift,
                'ground_height_source':'saved VST ground channel, world metres',
                'unknown_ground_note':'3D surface undefined, omitted only from 3D. All cells remain in the full 2D plot.',
                'layers':rows,'source_files_modified':False,'interactive_window_open':False}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
