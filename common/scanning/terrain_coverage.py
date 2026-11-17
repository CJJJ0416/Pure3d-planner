"""Audit complete motion and actual returns on known inter-floor structures."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.pointcloud import read_json,write_json


def triangle_keys(triangles):
    # OBJ components and the combined float32 PLY share triangles. Quantize
    # vertex serialization differences only for this teacher-side audit.
    vertices=np.rint(triangles*1e4).astype(np.int32)
    order=np.lexsort((vertices[:,:,2],vertices[:,:,1],vertices[:,:,0]),axis=1)
    canonical=np.take_along_axis(vertices,order[:,:,None],axis=1)
    return np.ascontiguousarray(canonical).reshape(-1,9).view('V36').reshape(-1)


def audit(report_path):
    report=read_json(report_path)
    root=Path(report['settings']['output'])
    source=Path(report['source_terrain'])
    scene=read_json(source/'scene.json')
    mesh=trimesh.load(root/'scene_assets/static_mesh.ply',process=False)
    keys=triangle_keys(mesh.triangles)
    connection_id=np.full(len(keys),-1,np.int16)
    connections=[]
    for index,row in enumerate(scene['stairs']):
        component=trimesh.load(source/(row['id']+'.obj'),process=False)
        mask=np.isin(keys,triangle_keys(component.triangles))
        connection_id[mask]=index
        connections.append(dict(id=row['id'],kind=row['kind'],from_floor=row['from_floor'],
            to_floor=row['to_floor'],component_faces=len(component.faces),matched_global_faces=int(mask.sum())))
    del mesh,keys
    results=[]
    colors=['#1b9abd','#d85b54','#a35ad1','#c59b21']
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    counts_per_route=[]
    for index,row in enumerate(report['routes']):
        folder=Path(row['root'])
        config=read_json(folder/'config.actual.json')
        generation=read_json(folder/'mixed/generation.json')
        records=read_json(folder/'mixed/registered.json')
        scans=read_json(folder/'mixed/scans.json')
        truth=np.load(folder/'truth_trajectory.npz')
        position=truth['position']
        t=truth['time']
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(position,axis=0),axis=1))]
        support_z=position[:,2]-config['sensor_height_m']
        hits=np.zeros(len(connections),np.int64)
        frames=np.zeros(len(connections),np.int64)
        for record in records:
            with np.load(folder/'mixed/scans'/f"{record['frame']:05d}.npz") as raw:
                keep=(raw['dynamic_instance']==0)&~raw['explicit_ghost']&~raw['edge_mixed']
                groups=connection_id[raw['face_id'][keep]]
                counts=np.bincount(groups[groups>=0],minlength=len(connections))
                hits+=counts
                frames+=counts>0
        assert generation['complete_path_captured']
        assert len(scans)==round(config['duration_s']*config['scan_hz'])
        assert records[-1]['frame']==len(scans)-1
        elapsed=scans[-1]['end_s']
        last=np.array([np.interp(elapsed,t,position[:,axis]) for axis in range(3)])
        result=dict(route=row['id'],duration_s=config['duration_s'],input_frames=len(scans),
            registered_frames=len(records),full_path_captured=True,
            distance_traveled_m=float(distance[-1]),distance_in_first20s_m=float(np.interp(20,t,distance)),
            support_z_start_m=float(support_z[0]),support_z_end_m=float(support_z[-1]),
            support_z_range_m=[float(support_z.min()),float(support_z.max())],
            last_scan_pose_to_motion_end_m=float(np.linalg.norm(last-position[-1])),
            static_connector_returns=int(hits.sum()),
            observed_connections=[dict(**connection,returns=int(hits[i]),registered_frames_with_returns=int(frames[i]))
                for i,connection in enumerate(connections) if hits[i]>0])
        results.append(result)
        counts_per_route.append(hits)
        color=colors[index%len(colors)]
        axes[0].plot(t,distance,label=row['id'],color=color)
        axes[1].plot(t,support_z,label=row['id'],color=color)
    axes[0].axvline(20,color='#777',linestyle='--',lw=1,label='Old 20 s cutoff')
    axes[0].set(xlabel='Physical time (s)',ylabel='Travelled distance (m)',title='Independent durations, complete motion')
    axes[1].set(xlabel='Physical time (s)',ylabel='Support height (m)',title='Full ascent and descent segments')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    bottoms=np.zeros(len(connections),np.int64)
    for index,(row,hits) in enumerate(zip(results,counts_per_route)):
        axes[2].barh(range(len(connections)),hits,left=bottoms,label=row['route'],color=colors[index%len(colors)])
        bottoms+=hits
    axes[2].set(yticks=range(len(connections)),yticklabels=[c['id'] for c in connections],
        xlabel='Static returns in registered PCDs',title='Exact source-triangle hits on connections')
    axes[2].tick_params(axis='y',labelsize=7)
    axes[2].legend(fontsize=8)
    for axis in axes:axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(report_path.parent/'full_path_coverage.png',dpi=150)
    plt.close(fig)
    result=dict(routes=results,connections=connections,
        count_scope='registered static primary-ray hits, excluding explicit flying and mixed-edge returns',
        not_an_accuracy_metric=True)
    write_json(report_path.parent/'coverage.json',result)
    print('FULL_PATH_COVERAGE',[(r['route'],r['duration_s'],r['static_connector_returns']) for r in results],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    audit(args.report.resolve())


if __name__=='__main__':main()
