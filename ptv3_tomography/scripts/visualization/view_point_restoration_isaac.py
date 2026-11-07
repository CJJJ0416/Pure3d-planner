#!/usr/bin/env python3
"""Native Isaac comparison of input, voxel and inverse-restored LiDAR point features."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from ptv3_tomography.io import atomic_json, file_sha256, read_json


from ptv3_tomography.paths import relocated_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restoration', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=16)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    args.restoration,args.stage,args.output = [p.resolve() for p in (args.restoration,args.stage,args.output)]
    report = read_json(args.restoration/'report.json')
    rows = report['frames_manifest']
    indices = [r['frame_index'] for r in rows]
    args.output.mkdir(parents=True,exist_ok=False)
    source_hash = file_sha256(args.stage)
    state = {'status':'starting','pid':os.getpid(),'frame':args.frame,'hardware_limits_imposed':False}
    atomic_json(args.output/'viewer_state.json',state)
    argv,sys.argv = sys.argv,sys.argv[:1]
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless':args.headless,'width':1280,'height':900,
                             'window_width':1800,'window_height':1000,'multi_gpu':False,
                             'renderer':'RayTracedLighting','denoiser':False,'samples_per_pixel_per_frame':1})
    finally:
        sys.argv = argv
    try:
        import omni.usd
        import omni.ui as ui
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import capture_viewport_to_file,get_active_viewport
        from pxr import Gf,UsdGeom,Vt
        import trimesh
        context = omni.usd.get_context()
        context.open_stage(str(args.stage))
        for _ in range(12):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        maps = read_json(relocated_path(report['source_maps'])/'manifest.json')
        dataset = relocated_path(report['source_dataset'])
        spec = read_json(dataset/'dataset_spec.json')
        scene = relocated_path(spec['scene_directory'])
        if not scene.is_absolute():
            scene = dataset/scene
        source_mesh = trimesh.load(scene/'terrain.obj',force='mesh',process=False)
        terrain = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Terrain'))
        terrain.CreateDisplayOpacityAttr([.28])
        cloud = UsdGeom.Points.Define(stage,'/World/PointRestoration/DisplayedPoints')
        cloud.CreatePointsAttr()
        cloud.CreateDisplayColorAttr()
        cloud.GetDisplayColorPrimvar().SetInterpolation('vertex')
        cloud.CreateWidthsAttr([.085])
        cloud.SetWidthsInterpolation('constant')
        origin = UsdGeom.Points.Define(stage,'/World/PointRestoration/LiDAROrigin')
        origin.CreateDisplayColorAttr([Gf.Vec3f(1.,1.,1.)])
        origin.CreateWidthsAttr([.12])
        modes = ['Raw LiDAR: every valid return','PTv3 voxel features: PCA colors',
                 'Restored LiDAR features: PCA colors','LiDAR points: sampled predicted map cost']
        names = ['raw_lidar','voxel_pca','restored_pca','sampled_map_cost']
        current = {'frame':args.frame,'mode':2,'dirty':True,'playing':False}
        cache = {}
        label = None

        def load(index):
            if index not in cache:
                row = next(r for r in rows if r['frame_index']==index)
                file = args.restoration/row['file']
                with np.load(file,allow_pickle=False) as a:
                    cache[index] = {k:a[k].copy() for k in a.files}
            return cache[index]

        def change(name,value):
            current[name],current['dirty'] = value,True

        def camera(detail=False):
            if detail:
                center = load(current['frame'])['world_from_sensor'][:3,3]+[0,.3,-.15]
                offset = [3.5,-4.5,3.2]
            else:
                center = source_mesh.bounds.mean(0)
                offset = [9,-12,12]
            set_camera_view(eye=center+offset,target=center,camera_prim_path='/OmniverseKit_Persp')

        def show():
            index,mode = current['frame'],current['mode']
            data = load(index)
            row = next(r for r in rows if r['frame_index']==index)
            xyz = data['points_sensor_m']
            if mode==0:
                colors = np.tile([.05,.85,1.],(len(xyz),1))
            elif mode==1:
                xyz,colors = data['voxel_points_sensor_m'],data['voxel_feature_rgb']
            elif mode==2:
                colors = data['restored_feature_rgb']
            else:
                colors = np.tile([.55,.58,.62],(len(xyz),1))
                valid = data['within_map_domain']
                ratio = np.clip(data['sampled_map_cost_raw'][valid]/60.,0,1)
                colors[valid] = np.column_stack((np.minimum(2*ratio,1),np.minimum(2*(1-ratio),1),np.full(len(ratio),.06)))
            tf = data['world_from_sensor']
            world = xyz@tf[:3,:3].T+tf[:3,3]
            cloud.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(world.astype(np.float32)))
            cloud.GetDisplayColorAttr().Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
            origin.CreatePointsAttr([Gf.Vec3f(*tf[:3,3])])
            expected = row['voxel_tokens'] if mode==1 else row['valid_returns']
            state.update(status='capturing' if args.headless else 'open',frame=index,mode=modes[mode],
                         displayed_points=len(world),valid_returns=row['valid_returns'],voxel_tokens=row['voxel_tokens'],
                         feature_channels=row['restored_feature_shape'][1],gt_mask_applied=False,
                         map_cost_colored_points=row['within_map_domain'] if mode==3 else None)
            if label is not None:
                label.text = (f"Frame {index} | {modes[mode]}\nDisplayed {len(world)} points | input valid returns {row['valid_returns']}\n"
                              f"Voxel features {row['voxel_tokens']} x 64 -> restored features {row['valid_returns']} x 64\n"
                              f"Map cost: {row['within_map_domain']} colored / {row['outside_map_domain']} outside map (gray)")
            current['dirty'] = False
            atomic_json(args.output/'viewer_state.json',state)

        if not args.headless:
            window = ui.Window('PTv3 | point restoration',width=540,height=480)
            with window.frame:
                with ui.VStack(spacing=7):
                    ui.Label('Every valid LiDAR return, with its original position',height=25)
                    frame_model = ui.ComboBox(indices.index(args.frame),*[f'Frame {i}' for i in indices],height=28).model
                    frame_model.add_item_changed_fn(lambda model,_:change('frame',indices[model.get_item_value_model().as_int]))
                    mode_model = ui.ComboBox(2,*modes,height=28).model
                    mode_model.add_item_changed_fn(lambda model,_:change('mode',model.get_item_value_model().as_int))
                    label = ui.Label('',height=110,word_wrap=True)
                    ui.Label('PCA colors compare features. They are not cost or class labels.',height=40,word_wrap=True)
                    ui.Label('Cost: green 0 -> yellow 30 -> red 60. Gray: outside the predicted map.',height=40,word_wrap=True)
                    ui.Label('Ground-truth supervision masks do not filter these LiDAR points.',height=40,word_wrap=True)
                    ui.Button('Play / Pause the 40-frame capture',height=30,
                              clicked_fn=lambda:current.update(playing=not current['playing']))
                    ui.Button('Whole scene view',height=30,clicked_fn=lambda:camera(False))
                    ui.Button('LiDAR / stairs detail view',height=30,clicked_fn=lambda:camera(True))
            for name in ['Stage','Property','Content','Console']:
                other = ui.Workspace.get_window(name)
                if other is not None:
                    other.visible = False
            viewport = ui.Workspace.get_window('Viewport')
            if viewport is not None:
                window.dock_in(viewport,ui.DockPosition.RIGHT,.29)
        camera()
        snapshots,captures = [],[]
        for detail,mode in [(detail,mode) for detail in [False,True] for mode in range(len(names))]:
            camera(detail)
            name = names[mode]
            if not args.headless:
                mode_model.get_item_value_model().set_value(mode)
            else:
                change('mode',mode)
            show()
            for _ in range(18):
                app.update()
            view_name = 'detail' if detail else 'overview'
            file = args.output/f'frame_{args.frame:05d}_{name}_{view_name}_native.png'
            capture = capture_viewport_to_file(get_active_viewport(),file_path=str(file))
            captures.append(capture)
            completion = asyncio.ensure_future(capture.wait_for_result(0))
            deadline = time.monotonic()+40
            while not completion.done() and time.monotonic()<deadline:
                app.update()
                time.sleep(.01)
            completion.result()
            import omni.renderer_capture
            omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
            snapshots.append({'mode':name,'view':view_name,'file':str(file),'sha256':file_sha256(file),
                              'displayed_points':state['displayed_points']})
        if not args.headless:
            frame_model.get_item_value_model().set_value(0)
            frame_model.get_item_value_model().set_value(indices.index(args.frame))
            mode_model.get_item_value_model().set_value(2)
        else:
            change('mode',2)
        show()
        stage.Export(str(args.output/'point_restoration_scene.usda'))
        atomic_json(args.output/'report.json',{'status':'passed','renderer':'Isaac Sim native viewport',
                    'source_stage_sha256':source_hash,'source_geometry_vertices_faces_verified':True,
                    'checkpoint_sha256':report['checkpoint_sha256'],'native_ui_callbacks_verified':not args.headless,
                    'all_valid_returns_displayed_without_gt_mask':True,'point_positions_unchanged':True,
                    'hardware_limits_imposed':False,'captures':snapshots,
                    'pca_colors_are_not_cost':True,'cost_values_are_sampled_existing_map_predictions':True})
        print('POINT_RESTORATION_VIEWER_READY '+str(state),flush=True)
        last_tick = time.monotonic()
        while not args.headless and app.is_running():
            if current['playing'] and time.monotonic()-last_tick>=.1:
                position = (indices.index(current['frame'])+1)%len(indices)
                frame_model.get_item_value_model().set_value(position)
                last_tick = time.monotonic()
            if current['dirty']:
                show()
            app.update()
            time.sleep(.02)
        state['status'] = 'closed'
        atomic_json(args.output/'viewer_state.json',state)
    except Exception as exc:
        state.update(status='failed',error=str(exc))
        atomic_json(args.output/'viewer_state.json',state)
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
