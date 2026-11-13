"""正式规划结果与紧凑地图可视化产物。"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from common.pointcloud_io import write_pcd


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_map(planner,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_pcd(output/'voxel_cost.pcd',planner.xyz,planner.cost)
    write_pcd(output/'dc_surface_cost.pcd',planner.search_surface,planner.surface_cost)
    graph=planner.graph_data();np.savez(output/'graph_view.npz',**graph)
    metadata=dict(source_pcd=planner.source_pcd,input_points=planner.input_points,
        original_points_in_roi=planner.original_point_count,voxel_points=len(planner.xyz),
        dc_vertices=len(planner.surface),dc_triangles=len(planner.triangles),search_nodes=len(planner.node_xyz),
        component_counts=graph['component_counts'].tolist(),base_undirected_edges=len(graph['base_edges']),
        new_undirected_step_edges=len(graph['added_edges']),step_rule=planner.link_info,
        geometry_field=planner.field_info,setup=planner.setup,coordinate_frame='map',
        endpoint_anchors=len(planner.anchor_voxel_indices),
        endpoint_selection='nearest input PCD voxel anchor, then its fixed nearest upward-facing DC candidate',
        step_sources_with_clearance=int(planner.source_clear.sum()),
        step_source_interval_free_m=planner.config.interval_free,
        step_source_geometry='Known occupied voxel intersections along gravity-up; affects added step edges only, not base nodes or triangles',
        geometry_source='input PCD only, observed-voxel shell, no Mesh or cavity filling')
    save_json(output/'map.json',metadata)
    return metadata


def save_result(planner,result,output,start,goal,wall_s=None,step_edges=None):
    """交互规划与CLI共用。失败保存状态，不保留上一条成功路径。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    enabled=planner.config.step_edges if step_edges is None else step_edges
    config=asdict(planner.config);config['step_edges']=enabled
    save_json(output/'config.json',config)
    save_json(output/'queries.json',dict(pcd=planner.source_pcd,start=np.asarray(start).tolist(),
        goal=np.asarray(goal).tolist(),coordinate_frame='map',coordinates='input PCD support positions in map XYZ metres'))
    save_json(output/'endpoint_snap.json',result['astar']['endpoint_snap'])
    metrics=dict(success=result['success'],method='DC_with_step' if enabled else 'DC_without_step',
        source_pcd=planner.source_pcd,astar=result['astar']['metrics'],
        planning_wall_s=result['planning_wall_s'] if wall_s is None else wall_s)
    if not result['success']:
        for filename in ['astar_path.csv','support_path.csv','gpmp_trajectory.csv','gpmp_support.csv','planning.npz','endpoints.json']:
            (output/filename).unlink(missing_ok=True)
        save_json(output/'metrics.json',metrics)
        return metrics
    a,g=result['astar'],result['gpmp'];metrics['gpmp']=g['metrics']
    np.savetxt(output/'astar_path.csv',np.column_stack((a['xyz'],a['cost'],a['point_indices'])),
        delimiter=',',header='x,y,z,trav_cost,point_index',comments='')
    np.savetxt(output/'support_path.csv',a['support_xyz'],delimiter=',',header='x,y,z',comments='')
    np.savetxt(output/'gpmp_trajectory.csv',np.column_stack((g['time'],g['xyz'],g['velocity'],g['yaw'],g['cost'])),
        delimiter=',',header='t,x,y,z,vx,vy,vz,yaw,trav_cost',comments='')
    np.savetxt(output/'gpmp_support.csv',np.column_stack((g['time'],g['support_xyz'],g['normal'],g['cost'])),
        delimiter=',',header='t,x,y,z,nx,ny,nz,trav_cost',comments='')
    np.savez(output/'planning.npz',astar_xyz=a['xyz'],astar_point_indices=a['point_indices'],
        time=g['time'],xyz=g['xyz'],velocity=g['velocity'],cost=g['cost'],
        initial_state=g['initial_state'],state=g['state'],surface_endpoints=g['surface_endpoints'],
        endpoint_normals=g['endpoint_normals'],endpoint_targets=g['endpoint_targets'])
    save_json(output/'endpoints.json',dict(offset_m=planner.config.endpoint_offset_m,
        point_indices=a['point_indices'][[0,-1]].tolist(),surface_xyz=g['surface_endpoints'].tolist(),
        support_reference='input PCD voxel anchors',pcd_anchor_xyz=a['endpoint_snap']['pcd_anchor_xyz'],
        dc_surface_xyz=a['surface_endpoints'].tolist(),
        unit_normals=g['endpoint_normals'].tolist(),gpmp_xyz=g['endpoint_targets'].tolist()))
    save_json(output/'metrics.json',metrics)
    return metrics
