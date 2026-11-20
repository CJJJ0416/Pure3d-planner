"""Separate reconstruction archive, goal nodes, per-node cost and merge stages."""
from common.paths import ROOT
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import yaml
from common.pointcloud_io import write_pcd, read_pcd_fields, fields_xyz
from Simulator.unitree_gazebo.scripts.frame_config import compose, rotation_matrix
from common.mapping.core import Snapshot, load_config, partition, block_name
from common.io import atomic_json, atomic_npy


class ReconstructionArchive:
    """Archive LIO outputs without running node construction or inference."""
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.frames, self.odometry, self.goals = [], [], []
        self.points = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = []
        self.error = None

    def cloud(self, coord, stamp, frame):
        if frame != 'camera_init': raise ValueError('LIO cloud must use camera_init')
        if self.frames and stamp <= self.frames[-1]['stamp']:
            raise ValueError('LIO cloud timestamps are not strictly increasing')
        coord = np.asarray(coord, np.float32)
        coord = np.ascontiguousarray(coord[np.isfinite(coord).all(1)])
        if not len(coord): return
        name = f'scans/{len(self.frames):06d}.npy'
        self.futures.append(self.pool.submit(atomic_npy, self.output/name, coord))
        self.frames.append(dict(stamp=float(stamp), path=name, first_point=self.points,
                                end_point=self.points+len(coord), points=len(coord)))
        self.points += len(coord)

    def finish(self, complete, transform=None):
        self.pool.shutdown(wait=True)
        for future in self.futures: future.result()
        atomic_json(self.output/'scans.json', self.frames)
        atomic_json(self.output/'odometry.json', self.odometry)
        atomic_json(self.output/'goals.json', self.goals)
        atomic_json(self.output/'frames.json', transform)
        if self.points:
            path = self.output/'map.npy'
            mmap = np.lib.format.open_memmap(path.with_suffix('.tmp.npy'), mode='w+', dtype=np.float32, shape=(self.points, 3))
            for frame in self.frames:
                mmap[frame['first_point']:frame['end_point']] = np.load(self.output/frame['path'])
            mmap.flush(); del mmap
            path.with_suffix('.tmp.npy').replace(path)
            write_pcd(self.output/'map.pcd', np.load(path, mmap_mode='r'))
        report = dict(state='complete' if complete else 'incomplete', frame_id='map',
            map_camera_identity=True if transform else None, revision=len(self.frames), points=self.points,
            first_scan_stamp=self.frames[0]['stamp'] if self.frames else None,
            last_scan_stamp=self.frames[-1]['stamp'] if self.frames else None,
            xyz='map.npy', pcd='map.pcd', scans='scans.json', error=self.error)
        atomic_json(self.output/'reconstruction.json', report)
        return report


def load_reconstruction(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory/'reconstruction/reconstruction.json').read_text())
    if report['state'] != 'complete': raise ValueError('Faster-LIO reconstruction has not completed')
    return directory/'reconstruction', report


def build_nodes(directory, mission_path):
    reconstruction, info = load_reconstruction(directory)
    output = Path(directory)/'nodes'
    output.mkdir(parents=True, exist_ok=False)
    mission_path = Path(mission_path)
    if mission_path.is_dir(): mission_path = mission_path/'mission.json'
    mission = json.loads(mission_path.read_text())
    execution = json.loads((mission_path.parent/'execution.json').read_text())
    arrivals = [e for e in execution['events'] if e['event']=='arrived']
    if execution['state'] != 'complete' or [e['segment'] for e in arrivals] != list(range(len(mission['segments']))):
        raise ValueError('Need every measured goal arrival before building complete nodes')
    frames = json.loads((reconstruction/'scans.json').read_text())
    odometry = json.loads((reconstruction/'odometry.json').read_text())
    recorded_goals = json.loads((reconstruction/'goals.json').read_text())
    transform = json.loads((reconstruction/'frames.json').read_text())
    if not frames or not odometry or not transform or not transform['map_camera_identity']:
        raise ValueError('Missing reconstruction geometry, body poses or identity map/camera TF')
    goals=[]
    for goal in recorded_goals:
        if goal['frame_id'] != 'world': raise ValueError('Recorded mission goals must use world')
        if not goals or np.linalg.norm(np.asarray(goal['position'])-goals[-1]['position']) >= .01:
            goals.append(goal)
    if len(goals) != len(mission['segments']) or any(
            np.linalg.norm(np.asarray(goal['position'])-segment['goal_body_xyz']) > .01
            for goal,segment in zip(goals,mission['segments'])):
        raise ValueError('Mission goals differ from the goals archived during reconstruction')
    arrival_times=[event['stamp'] for event in arrivals]
    if (any(b <= a for a,b in zip(arrival_times,arrival_times[1:])) or
            any(event['stamp'] < goal['stamp'] for event,goal in zip(arrivals,goals))):
        raise ValueError('Goal arrival timestamps are not in execution order')
    world_map = (np.asarray(transform['world_map_translation']), np.asarray(transform['world_map_quaternion_xyzw']))
    from Simulator.unitree_gazebo.scripts.frame_config import inverse
    map_world = inverse(world_map)
    xyz = np.load(reconstruction/'map.npy', mmap_mode='r')
    times = np.array([f['stamp'] for f in frames])
    pose_times = np.array([o['stamp'] for o in odometry])
    stepwise=mission.get('planning_mode')=='stepwise_short_goals'
    publications=[e for e in execution['events'] if e['event']=='published']
    holds=[e for e in execution['events'] if e['event']=='hold_complete']
    if stepwise:
        if len(publications)!=len(arrivals) or len(holds)!=len(arrivals):
            raise ValueError('Short-goal nodes need every publication, measured arrival and completed hold')
        boundaries=[int(np.searchsorted(times,e['stamp'],side='left')) for e in publications]+[len(frames)]
    else:
        boundaries = [1]+[int(np.searchsorted(times, e['stamp'], side='right')) for e in arrivals]
    # Attach the final stationary tail to the last reached node so every raw
    # point belongs to exactly one node, including post-arrival observations.
    boundaries[-1] = len(frames)
    if any(b < a for a, b in zip([0]+boundaries, boundaries)):
        raise ValueError('Goal times precede the initial reconstructed scan')
    rows, first_frame = [], 0
    for index, end_frame in enumerate(boundaries):
        name = f'n{index}'
        event = None if index==0 else arrivals[index-1]
        pose_stamp = times[0] if event is None else event['stamp']
        pose_index = int(np.argmin(np.abs(pose_times-pose_stamp)))
        if abs(pose_times[pose_index]-pose_stamp) > .25:
            raise ValueError(f'No LIO body pose near {name} at {pose_stamp}')
        first_point = frames[first_frame]['first_point'] if first_frame < len(frames) else len(xyz)
        end_point = frames[end_frame-1]['end_point'] if end_frame > first_frame else first_point
        coord = xyz[first_point:end_point]
        node_dir = output/name
        atomic_npy(node_dir/'cloud.npy', coord)
        write_pcd(node_dir/'cloud.pcd', coord)
        goal = None if event is None else mission['segments'][index-1]['goal_body_xyz']
        goal_map = None if goal is None else compose(map_world, (np.asarray(goal), np.array([0.,0.,0.,1.])))[0].tolist()
        blocks = [block_name(key) for key, _ in partition(coord, [4.,4.,3.])]
        world_coord=coord@rotation_matrix(world_map[1]).T+world_map[0]
        eroi_blocks=[dict(id=block_name(key),key=list(key),points=len(indices))
                     for key,indices in partition(world_coord,[4.,4.,3.])]
        eroi=dict(id=name,frame_id='world',definition=('all initial stationary scans' if stepwise and index==0
            else 'all raw scans during this node motion, stationary hold and planning wait'),
            grid_size_m=[4.,4.,3.],core_blocks=eroi_blocks,points=len(coord),
            bounds_world=[np.min(world_coord,0).tolist(),np.max(world_coord,0).tolist()] if len(coord) else None,
            cropped=False)
        atomic_json(node_dir/'eroi.json',eroi)
        row = dict(id=name, parent=f'n{index-1}' if index else None, frame_id='map',
            goal_world=goal, goal_map=goal_map, arrival_stamp=None if event is None else event['stamp'],
            arrival_source='recorded Gazebo measured arrival' if event else 'first valid LIO frame',
            body_pose=odometry[pose_index], first_frame=first_frame, end_frame=end_frame,
            first_point=first_point, end_point=end_point, points=len(coord), blocks=blocks,
            first_scan_stamp=times[first_frame] if end_frame > first_frame else None,
            last_scan_stamp=times[end_frame-1] if end_frame > first_frame else None,
            motion_start_stamp=publications[index-1]['stamp'] if stepwise and index else None,
            hold_complete_stamp=holds[index-1]['stamp'] if stepwise and index else None,
            hold_seconds=holds[index-1]['stationary_s'] if stepwise and index else None,
            eroi=str(node_dir/'eroi.json'),
            includes_final_stationary_tail=index==len(boundaries)-1,
            xyz=str(node_dir/'cloud.npy'), pcd=str(node_dir/'cloud.pcd'))
        atomic_json(node_dir/'node.json', row)
        rows.append(row); first_frame=end_frame
    if sum(row['points'] for row in rows) != len(xyz): raise ValueError('Node coverage is incomplete')
    report = dict(state='complete', reconstruction=str(reconstruction), revision=info['revision'],
        points=len(xyz), nodes=rows, mission=str(mission_path.resolve()),
        ownership=('n0 initial stationary scans; each following node owns all motion/hold scans until next path publication; final tail belongs to last node'
                   if stepwise else 'n0 first scan; subsequent goal intervals; final tail belongs to last node'))
    atomic_json(output/'nodes.json', report)
    return report


def compute_node_cost(directory, config_path, inference_frame='map', variant='', context_source='final'):
    from common.cost.client import WorkerClient
    reconstruction, geometry = load_reconstruction(directory)
    nodes = json.loads((Path(directory)/'nodes/nodes.json').read_text())
    if nodes['state'] != 'complete' or nodes['revision'] != geometry['revision']:
        raise ValueError('Nodes and reconstructed map have different versions')
    if inference_frame not in ('map','world'): raise ValueError('Expected map or world inference frame')
    if context_source not in ('final','node'): raise ValueError('Expected final or node context source')
    suffix = '_'+variant if variant else ''
    output = Path(directory)/('node_cost'+suffix)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    config['checkpoint'] = str((ROOT/config['checkpoint']).resolve())
    resolved_config = output/'config.yaml'
    resolved_config.write_text(yaml.safe_dump(config))
    xyz = np.load(reconstruction/'map.npy', mmap_mode='r')
    inference_path = reconstruction/'map.npy'
    transform = None
    if inference_frame == 'world':
        transform = json.loads((reconstruction/'frames.json').read_text())
        rotation = rotation_matrix(np.array(transform['world_map_quaternion_xyzw']))
        translation = np.array(transform['world_map_translation'])
        inference_path = output/'world_xyz.npy'
        world = np.lib.format.open_memmap(inference_path,mode='w+',dtype=np.float32,shape=xyz.shape)
        for first in range(0,len(xyz),262144):
            last=min(first+262144,len(xyz))
            world[first:last]=xyz[first:last]@rotation.T+translation
        world.flush();del world
    inference_xyz = np.load(inference_path,mmap_mode='r')
    snapshot = Snapshot(geometry['revision'], geometry['last_scan_stamp'], inference_xyz, ())
    report = dict(state='incomplete', revision=geometry['revision'], points=len(xyz), nodes=[],
        frame_id='map',inference_frame=inference_frame,inference_xyz=str(inference_path),transform=transform,
        context_source=context_source,
        context_policy=(f'own node observations only; spatial core + 1m halo in {inference_frame}' if context_source=='node' else
                        f'final geometry spatial core + 1m halo in {inference_frame}; each node queries its owned raw points'))
    worker = WorkerClient(config, resolved_config, output)
    started = time.perf_counter()
    try:
        worker.initialize()
        for node in nodes['nodes']:
            first, last = node['first_point'], node['end_point']
            if last > first:
                if context_source=='node':
                    node_input=output/node['id']/'inference_xyz.npy'
                    atomic_npy(node_input,inference_xyz[first:last])
                    local_snapshot=Snapshot(node['end_frame'],node['last_scan_stamp'],inference_xyz[first:last],())
                    _,cost,inference=worker.predict(local_snapshot,input_path=node_input)
                else:
                    _, cost, inference = worker.predict(snapshot, query_range=[first,last], input_path=inference_path)
            else:
                cost, inference = np.empty(0,np.float32), dict(points=0, encoder_calls=0)
            node_dir = output/node['id']
            atomic_npy(node_dir/'cost.npy', cost)
            write_pcd(node_dir/'prediction.pcd', xyz[first:last], cost)
            record = dict(id=node['id'], first_point=first, end_point=last, points=last-first,
                cost=str(node_dir/'cost.npy'), pcd=str(node_dir/'prediction.pcd'), report=inference)
            atomic_json(node_dir/'prediction.json', record)
            report['nodes'].append(record)
            atomic_json(output/'cost.json', report)
            print(json.dumps(dict(stage='cost',node=node['id'],points=last-first,context_source=context_source,
                wall_s=inference.get('timing',{}).get('total_s'))),flush=True)
        report['state']='complete'
    except BaseException as error:
        report['error']=str(error) or type(error).__name__
        raise
    finally:
        worker.close()
        report['wall_s']=time.perf_counter()-started
        atomic_json(output/'cost.json', report)
    return report


def merge_node_cost(directory, variant=''):
    reconstruction, geometry = load_reconstruction(directory)
    suffix = '_'+variant if variant else ''
    cost_directory = Path(directory)/('node_cost'+suffix)
    report = json.loads((cost_directory/'cost.json').read_text())
    nodes = json.loads((Path(directory)/'nodes/nodes.json').read_text())
    if report['state'] != 'complete' or report['revision'] != geometry['revision']:
        raise ValueError('Per-node cost is incomplete or belongs to a different map')
    if [n['id'] for n in report['nodes']] != [n['id'] for n in nodes['nodes']]:
        raise ValueError('Cost node sequence differs from node ownership')
    xyz = np.load(reconstruction/'map.npy', mmap_mode='r')
    costs = np.empty(len(xyz), np.float32)
    cursor = 0
    for row, node in zip(report['nodes'], nodes['nodes']):
        first, last = row['first_point'], row['end_point']
        if first != cursor or (first,last) != (node['first_point'],node['end_point']):
            raise ValueError('Node ranges overlap or leave a gap')
        values = np.load(row['cost'])
        cloud = read_pcd_fields(row['pcd'])
        if (values.shape != (last-first,) or not np.isfinite(values).all() or
                not np.array_equal(fields_xyz(cloud), xyz[first:last]) or
                not np.array_equal(cloud['trav_cost'], values)):
            raise ValueError(f'{row["id"]}: prediction points/cost do not match final geometry')
        costs[first:last]=values; cursor=last
    if cursor != len(xyz): raise ValueError('Cost does not cover every reconstructed point')
    output = Path(directory)/('merged'+suffix)
    output.mkdir(parents=True, exist_ok=False)
    atomic_npy(output/'cost.npy', costs)
    write_pcd(output/'prediction.pcd', xyz, costs)
    merged = dict(state='complete', frame_id='map', revision=geometry['revision'], points=len(xyz),
        nodes=len(report['nodes']), xyz_order_verified=True, context_duplicates=False,
        reconstruction=str(reconstruction), prediction=str(output/'prediction.pcd'),
        inference_frame=report.get('inference_frame','map'),context_source=report.get('context_source','final'),
        model=str(cost_directory/'model.json'))
    atomic_json(output/'merge.json', merged)
    return merged
