"""Native VST random paths -> batched Warp rays -> isolated LIO -> paired data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Semaphore
from time import perf_counter

import numpy as np
import trimesh

from common.geneator.filter_wfc import object_proxies
from common.pointcloud_io import read_pcd_xyz
from utils.pointcloud import read_json, write_json
from .generate import load_config, imu_stream
from .raycast import RollingRayCaster
from .trajectory import Trajectory


def terrain_assets(settings, root):
    source = Path(settings['terrain']).resolve()
    folder = root / 'scene_assets'
    folder.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(source/'terrain.obj', process=False, force='mesh')
    mesh.export(folder/'static_mesh.ply')
    (folder/'clean.pcd').symlink_to(source/settings['clean_pcd'])
    # Existing WFC export has no per-triangle semantic identity. Preserve one
    # static-building instance, without inventing stair/wall class labels.
    np.savez(folder/'mesh_faces.npz', instance_id=np.ones(len(mesh.faces),np.int32),
        semantic_id=np.zeros(len(mesh.faces),np.int32))
    np.savez(folder/'clean_attributes.npz', metadata_available=np.array(False))
    templates = dict(object_proxies([0,0,0]))
    actors = [templates[name] for name in ['person_proxy','movable_clutter']]
    for name, actor in zip(['person_proxy','movable_clutter'], actors):
        actor.export(folder/f'dynamic_{name}.ply')
    scene = read_json(source/'scene.json')
    scene.update(bounds=mesh.bounds.tolist(), geometry_group_id=f"fasterlio_{scene['seed']}",
        source_scene=str(source), static_face_identity='one building instance, semantic unknown',
        reference_pcd=settings['clean_pcd'])
    write_json(folder/'scene.json', scene)
    return folder, mesh, actors, scene


def query_check(caster, trajectories, frame):
    """Compare actual full ray results for one batch against separate queries."""
    seeds = [9000+i for i in range(len(trajectories))]
    caster.scan_batch(frame, trajectories, True, True, [np.random.default_rng(s) for s in seeds])
    tick = perf_counter()
    batch = caster.scan_batch(frame, trajectories, True, True, [np.random.default_rng(s) for s in seeds])
    batch_s = perf_counter()-tick
    tick = perf_counter()
    singles = [caster.scan(frame,t,True,True,np.random.default_rng(s)) for t,s in zip(trajectories,seeds)]
    single_s = perf_counter()-tick
    for a,b in zip(batch,singles):
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
    return dict(frame=frame, trajectories=len(trajectories), outputs_equal=True,
        fields=list(batch[0]), batch_wall_s=batch_s, separate_wall_s=single_s)


def generate_group(settings, base, root, assets, mesh, actors, routes, group_index, prefix_frames, on_ready=None):
    tick = perf_counter()
    configurations = []
    # Share scan timestamps, not motion durations. Finished trajectories leave
    # the active ray batch immediately and can start their own LIO consumer.
    for route in routes:
        config = {**base, 'support_path':route['support_path'], 'duration_s':1.,
            'max_speed_m_s':settings['max_speed_m_s'], 'max_yaw_rate_deg_s':settings['max_yaw_rate_deg_s']}
        configurations.append(config)
    minimum = [Trajectory(None,c).minimum_duration_s for c in configurations]
    durations = np.ceil(np.asarray(minimum)*base['scan_hz'])/base['scan_hz']
    counts = np.rint(durations*base['scan_hz']).astype(int)
    if prefix_frames is not None:
        counts = np.minimum(counts, prefix_frames)
    actor_paths = [np.load(route['support_path']).tolist() for route in routes]
    actor_paths = [actor_paths[i%len(actor_paths)] for i in range(2)]
    jobs, trajectories, streams, rngs = [], [], [], []
    profile = settings['profile']
    for index,(route,c) in enumerate(zip(routes,configurations)):
        folder = root/route['id']
        folder.mkdir(parents=True,exist_ok=True)
        for name in ['static_mesh.ply','clean.pcd','clean_attributes.npz','mesh_faces.npz','scene.json',
                     'dynamic_person_proxy.ply','dynamic_movable_clutter.ply']:
            (folder/name).symlink_to(assets/name)
        trajectory_index = int(route['id'].rsplit('_',1)[1])
        c.update(output=str(folder), duration_s=float(durations[index]), frames=int(counts[index]), split='train',
            seed=read_json(assets/'scene.json')['seed'], trajectory_index=trajectory_index,
            actor_support_paths=actor_paths, playback_rate=settings['playback_rate'],
            explicit_ghost_probability=settings['explicit_ghost_probability'],
            explicit_ghost_pull_m=settings['explicit_ghost_pull_m'], profiles=[profile],
            gt_pcd_diagnostics=settings['gt_pcd_diagnostics'],
            warp_cache=str(root/'.warp_cache'))
        trajectory = Trajectory(None,c)
        write_json(folder/'trajectory_report.json', {**trajectory.save(folder/'truth_trajectory.npz'),
            'minimum_duration_s':float(minimum[index]), 'captured_prefix_s':int(counts[index])/c['scan_hz'],
            'complete_path_captured':bool(prefix_frames is None or counts[index]==round(durations[index]*c['scan_hz'])),
            'support_reference':'VST ground nearest GPMP height, not highest XY surface'})
        write_json(folder/'config.actual.json',c)
        stream = {key:int(np.random.SeedSequence([c['seed'],settings['route_seed'],trajectory_index,i]).generate_state(1)[0])
            for key,i in dict(measurement=0,imu=1,imu_jitter=3).items()}
        imu,biases = imu_stream(trajectory,c,profile in ['mixed','lio_error'],stream)
        scan_folder = folder/profile/'scans'
        scan_folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/profile/'imu.npz',**imu)
        job = dict(id=route['id'],root=str(folder),config=str(folder/'config.actual.json'),
            ros_domain_id=settings['ros_domain_base']+trajectory_index, records=[], biases=biases,
            trajectory_index=trajectory_index, stream=stream)
        jobs.append(job)
        trajectories.append(trajectory)
        streams.append(stream)
        rngs.append(np.random.default_rng(stream['measurement']))
    caster = RollingRayCaster(mesh,actors,configurations[0])
    # Check start and midpoint poses, including their selected building floors.
    checks = [query_check(caster,trajectories,f) for f in [0,int(min(counts)//2)]]
    launches_before, query_before = caster.launch_count,caster.query_wall_s
    query_s = save_s = 0.
    capture_started = perf_counter()
    faces = dict(np.load(assets/'mesh_faces.npz'))
    transitions=[]
    previous_active=[]

    def finish(index):
        job=jobs[index]
        folder=Path(job['root'])/profile
        write_json(folder/'scans.json',job['records'])
        write_json(folder/'generation.json',dict(profile=profile,streams=job['stream'],biases=job['biases'],
            frames=int(counts[index]),returns=sum(r['points'] for r in job['records']),
            dynamic_returns=sum(r['dynamic_points'] for r in job['records']),
            edge_mixed_returns=sum(r['edge_mixed_points'] for r in job['records']),
            explicit_ghost_returns=sum(r['explicit_ghost_points'] for r in job['records']),
            batching='shared scan timestamps, independent trajectory durations',group_index=group_index,
            captured_prefix_s=int(counts[index])/base['scan_hz'],
            full_trajectory_duration_s=float(durations[index]),
            complete_path_captured=bool(counts[index]==round(durations[index]*base['scan_hz']))))
        del job['records']
        if on_ready is not None:
            on_ready(job)

    for frame in range(int(max(counts))):
        active=np.flatnonzero(frame<counts).tolist()
        if active!=previous_active:
            transitions.append(dict(frame=frame,time_s=frame/base['scan_hz'],route_ids=[jobs[i]['id'] for i in active]))
            previous_active=active
        q = perf_counter()
        results = caster.scan_batch(frame,[trajectories[i] for i in active],profile in ['dynamic','mixed'],
            profile in ['edge_returns','mixed'],[rngs[i] for i in active])
        query_s += perf_counter()-q
        q = perf_counter()
        for index,result in zip(active,results):
            job=jobs[index]
            static = result['dynamic_instance']==0
            instance = result['dynamic_instance'].copy()
            semantic = np.zeros(len(static),np.int32)
            instance[static] = faces['instance_id'][result['face_id'][static]]
            semantic[~static] = np.where(instance[~static]==1001,8,10)
            result.update(instance_id=instance,semantic_id=semantic,
                trajectory_id=np.full(len(static),job['trajectory_index'],np.int32))
            folder = Path(job['root'])/profile
            np.savez(folder/'scans'/f'{frame:05d}.npz',**result)
            end = frame/base['scan_hz']+float(np.float32(result['offset_ns'][-1]/1e6))/1000
            job['records'].append(dict(frame=frame,start_s=frame/base['scan_hz'],end_s=end,
                trajectory_id=job['trajectory_index'],points=len(static),dynamic_points=int((~static).sum()),
                edge_mixed_points=int(result['edge_mixed'].sum()),explicit_ghost_points=int(result['explicit_ghost'].sum()),
                blind_near_rays=int((result['return_status']==2).sum()),blind_fov_rays=int((result['return_status']==3).sum()),
                path=f'scans/{frame:05d}.npz'))
        save_s += perf_counter()-q
        for index in active:
            if frame+1==counts[index]:
                finish(index)
        if frame%100==0:
            print('BATCH_RAYS',group_index,frame,'trajectories',len(active),flush=True)
    capture_wall = perf_counter()-capture_started
    report = dict(group_index=group_index,route_ids=[r['id'] for r in routes],
        frames_per_route={r['id']:int(n) for r,n in zip(routes,counts)},
        duration_s={r['id']:float(t) for r,t in zip(routes,durations)},
        captured_prefix_s={r['id']:int(n)/base['scan_hz'] for r,n in zip(routes,counts)},
        active_batch_transitions=transitions,rays_per_scan=base['rays_per_scan'],
        footprint_beams=5 if profile in ['edge_returns','mixed'] else 1,
        launch_count=caster.launch_count-launches_before,
        total_query_rays=int(counts.sum())*base['rays_per_scan']*(5 if profile in ['edge_returns','mixed'] else 1),
        query_and_sensor_s=query_s,kernel_launch_transfer_and_sync_s=caster.query_wall_s-query_before,
        save_s=save_s,capture_wall_s=capture_wall,whole_group_wall_s=perf_counter()-tick,
        single_vs_batched_checks=checks,
        flatten_order=['trajectory','footprint_beam','ray'],sensor_timestamps_shared=True)
    write_json(root/f'ray_batch_{group_index:03d}.json',report)
    return jobs,report


def run(settings,root,reports,prefix_frames,stages):
    root.mkdir(parents=True,exist_ok=True)
    reports.mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    settings['output'],settings['report_output']=str(root),str(reports)
    write_json(root/'batch.actual.json',settings)
    workspace = Path(__file__).resolve().parents[2]
    events,lock=[],Lock()
    prepare_slot=Semaphore(settings['prepare_workers'])

    def command(route,stage,args,env=None):
        begin=perf_counter()-started
        with (reports/f'{route}_{stage}.log').open('w') as log:
            subprocess.run(args,cwd=workspace,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
        row=dict(route=route,stage=stage,start_s=begin,end_s=perf_counter()-started)
        with lock:
            events.append(row)
            write_json(reports/'timeline.json',sorted(events,key=lambda r:r['start_s']))
        print('COMPLETE',row,flush=True)

    if 'plan' in stages:
        native=str(Path(settings['native_dir']).resolve())
        env={**os.environ,'LD_LIBRARY_PATH':native+':'+os.environ.get('LD_LIBRARY_PATH','')}
        command('terrain','plan',[settings['planner_python'],'-m','common.scanning.terrain_routes',
            '--config',str(root/'batch.actual.json'),'--output',str(root/'routes')],env)
    planning=read_json(root/'routes/routes.json')
    routes=[r for r in planning['routes'] if r['success']][:settings['capture_routes']]

    def consume(job):
        if 'replay' in stages:
            command(job['id'],'lio',[sys.executable,'-m','common.scanning.pipeline',
                '--config',job['config'],'--profiles',settings['profile'],
                '--ros-domain-id',str(job['ros_domain_id']),'--stages','replay'])
        if 'prepare' in stages:
            with prepare_slot:
                command(job['id'],'prepare',[sys.executable,'-m','common.scanning.prepare',
                    '--root',job['root'],'--profile',settings['profile']])

    jobs,groups,futures=[],[],[]
    with ThreadPoolExecutor(max_workers=settings['replay_workers']) as workers:
        def on_ready(job):
            futures.append(workers.submit(consume,job))
        if 'scan' in stages:
            assets,mesh,actors,scene=terrain_assets(settings,root)
            base=load_config(settings['base_config'])
            width=settings['ray_batch_routes']
            for group,start in enumerate(range(0,len(routes),width)):
                begin=perf_counter()-started
                ready,report=generate_group(settings,base,root,assets,mesh,actors,routes[start:start+width],group,prefix_frames,on_ready)
                groups.append(report)
                with lock:
                    events.append(dict(route=f'batch_{group}',stage='warp',start_s=begin,end_s=perf_counter()-started))
                    write_json(reports/'timeline.json',events)
                jobs+=ready
        else:
            groups=[read_json(path) for path in sorted(root.glob('ray_batch_*.json'))]
            for route in routes:
                folder=root/route['id']
                index=int(route['id'].rsplit('_',1)[1])
                job=dict(id=route['id'],root=str(folder),config=str(folder/'config.actual.json'),ros_domain_id=settings['ros_domain_base']+index)
                jobs.append(job)
                futures.append(workers.submit(consume,job))
        for future in futures:
            future.result()
    scenes=[dict(scene_id=f"{job['id']}/{settings['profile']}",split='train') for job in jobs]
    write_json(root/'manifest.json',dict(version='vst_random_paths_batched_scan_v1',scenes=scenes,test_suites=['nominal'],
        split_unit='building seed, never split routes of one building',input='XYZ only',training_started=False))
    results=[]
    if 'prepare' in stages:
        for job in jobs:
            folder=Path(job['root'])/settings['profile']
            results.append(dict(id=job['id'],root=job['root'],replay=read_json(folder/'replay.json'),metrics=read_json(folder/'metrics.json')))
    report=dict(wall_s=perf_counter()-started,settings=settings,planning=planning,ray_batches=groups,
        routes=results,events=events,requested_prefix_frames=prefix_frames,
        training_started=False,source_terrain=str(Path(settings['terrain']).resolve()))
    write_json(reports/'report.json',report)
    print('TERRAIN_BATCH_COMPLETE',report['wall_s'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('common/scanning/terrain_batch_config.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report-output',type=Path)
    parser.add_argument('--frames',type=int,help='Capture a physical-time prefix, without accelerating the planned motion')
    parser.add_argument('--stages',nargs='+',default=['plan','scan','replay','prepare'])
    args=parser.parse_args()
    settings=read_json(args.config)
    run(settings,Path(args.output or settings['output']).resolve(),Path(args.report_output or settings['report_output']).resolve(),args.frames,args.stages)


if __name__=='__main__':main()
