"""正式DC规划入口，--ablation仅比较是否添加跨阶边。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import PlannerConfig
from .io import save_json,save_map,save_result
from .planner_wrapper import DCSurfacePlanner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcd',type=Path,required=True,help='含XYZ和归一化trav_cost的建筑PCD')
    parser.add_argument('--start',nargs=3,type=float,required=True,help='输入PCD上的支撑位置XYZ，地图坐标，米')
    parser.add_argument('--goal',nargs=3,type=float,required=True,help='输入PCD上的支撑位置XYZ，地图坐标，米')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'config/dc.json')
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--without-step',action='store_true',help='只运行无跨阶边消融')
    mode.add_argument('--ablation',action='store_true',help='复用同一张DC图，同时运行正式方案和无跨阶边消融')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=PlannerConfig(**json.loads(args.config.read_text()))
    if args.without_step:cfg.step_edges=False
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',asdict(cfg))
    query=dict(pcd=str(args.pcd.resolve()),start=args.start,goal=args.goal,coordinate_frame='map',
        coordinates='input PCD support positions in map XYZ metres')
    save_json(output/'queries.json',query)
    planner=DCSurfacePlanner(cfg,output/'native');planner.load_pcd(args.pcd.resolve())
    map_info=save_map(planner,output/'map')
    variants={};modes=[True,False] if args.ablation else [cfg.step_edges]
    for enabled in modes:
        name='with_step' if enabled else 'without_step'
        result=planner.plan(args.start,args.goal,enabled)
        variants[name]=save_result(planner,result,output/name,args.start,args.goal,step_edges=enabled)
        print(name,json.dumps(variants[name],ensure_ascii=False),flush=True)
    summary=dict(default_method='DC_with_step',only_ablation='DC_without_step',config=asdict(cfg),
        query=query,map=map_info,variants=variants)
    save_json(output/'summary.json',summary)
    lines=['# DC正式方案与无跨阶边消融','','两组共用PCD、DC表面、节点、代价、起终点、A*边权和GPMP，仅连边规则不同。','',
        '| 方案 | 成功 | A*长度(m) | cost积分 | A*目标 | GPMP长度(m) | GPMP收敛 |',
        '|---|---|---:|---:|---:|---:|---|']
    for name,r in variants.items():
        if r['success']:
            a,g=r['astar'],r['gpmp'];lines.append(f'| {name} | True | {a["length_m"]:.6f} | {a["integral_cost_m"]:.6f} | {a["objective"]:.6f} | {g["length_m"]:.6f} | {g["converged"]} |')
        else:lines.append(f'| {name} | False | — | — | — | — | — |')
    lines+=['',f'原始点数{map_info["input_points"]:,}，ROI对应原始点数{map_info["original_points_in_roi"]:,}，'
        f'占用体素{map_info["voxel_points"]:,}，DC搜索节点{map_info["search_nodes"]:,}。',
        '几何只来自实测PCD占用壳，不读取Mesh、不填洞、不推断未知空间。此为参考路径结果，不是实机执行或碰撞验收。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if args.plot:
        from .visualization.plot import plot_run
        plot_run(output)
    print(output,flush=True)


if __name__=='__main__':main()
