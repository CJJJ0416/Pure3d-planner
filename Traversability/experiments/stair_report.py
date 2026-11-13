"""Plot the completed stair experiment and full-query GT/prediction comparisons."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.pointcloud import read_json, write_json
from utils.pointcloud import ROOT


def plot_teacher_examples(data, output):
    """Show the actual low/strict stand labels for the three pilot stair types."""
    cmap=LinearSegmentedColormap.from_list("teacher_cost",["#169c50","#f5df43","#f0443c"])
    fig=plt.figure(figsize=(16,10))
    axes=[]
    for column,seed in enumerate([202621001,202621041,202621061]):
        folder=Path(data)/"scenes"/f"scene_{seed}"
        scene=read_json(folder/"scene.json")
        xyz=np.load(folder/"geometry/query_coord.npy")
        cost=np.load(folder/"conditioned/cost.npy",mmap_mode="r")
        for row,profile in enumerate([1,2]):
            ax=fig.add_subplot(2,3,row*3+column+1,projection="3d")
            im=ax.scatter(*xyz.T,c=cost[:,profile],s=2,cmap=cmap,vmin=0,vmax=1,linewidths=0)
            ax.set_box_aspect(np.ptp(xyz,axis=0))
            ax.view_init(32,-55)
            ax.set(title=f"{scene['kind']} | stand_ratio={[.05,.3][row]}",
                   xlabel="World X (m)",ylabel="World Y (m)",zlabel="Z (m)")
            axes.append(ax)
    fig.suptitle("Native VST teacher | identical geometry within each column\nclearance=0.5 m, slope=0.4 rad, step=0.3 m | all native Tomogram points")
    fig.colorbar(im,ax=axes,shrink=.6,pad=.02,label="Normalised cost")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=140)
    plt.close(fig)


def known_stair_report(run, plots):
    import heapq
    from scipy.spatial import cKDTree
    example=np.load(ROOT/"datasets/wfc_regression_20261127/stair_case.npz")
    initial=np.load(run/"known_stair_initial.npz")
    result=np.load(run/"known_stair_regression.npz")
    column=result["profiles"].tolist().index("clearance05_slope06_step04_stand005")
    distance,indices=cKDTree(result["coord"]).query(example["coord"])
    pred=result["prediction"][indices,column]
    xyz=example["coord"]
    adjacency=cKDTree(xyz).query_ball_point(xyz,.45)
    start,goal=int(example["start"]),int(example["goal"])
    def bottleneck(cost):
        best=np.full(len(cost),np.inf)
        best[start]=cost[start]
        queue=[(float(cost[start]),start)]
        while queue:
            value,node=heapq.heappop(queue)
            if value!=best[node]:continue
            if node==goal:return float(value)
            for neighbor in adjacency[node]:
                candidate=max(value,float(cost[neighbor]))
                if candidate<best[neighbor]:
                    best[neighbor]=candidate
                    heapq.heappush(queue,(candidate,neighbor))
        return None
    _, initial_indices=cKDTree(initial["coord"]).query(example["coord"])
    initial_column=initial["profiles"].tolist().index("clearance05_slope06_step04_stand005")
    values=dict(GT=example["GT"],Initial=initial["prediction"][initial_indices,initial_column],Selected=pred)
    thresholds={name:bottleneck(cost) for name,cost in values.items()}
    write_json(run/"known_stair_connectivity.json",dict(minimum_required_threshold=thresholds,
        connected_at_03={name:value is not None and value<=.3 for name,value in thresholds.items()},
        max_query_match_distance_m=float(distance.max()),radius_m=.45,threshold=.3))
    path=example["gt_path"]
    length=np.r_[0,np.linalg.norm(np.diff(xyz[path],axis=0),axis=1).cumsum()]
    fig,ax=plt.subplots(figsize=(12,4.5))
    for name,cost in values.items():
        ax.plot(length,cost[path],label=f"{name}: bottleneck {thresholds[name]:.3f}",marker=".",lw=1.2)
    ax.axhline(.3,ls="--",color="red",label="cost threshold 0.3")
    ax.set(xlabel="Distance along the same GT route (m)",ylabel="Normalised cost",ylim=(-.03,1.03),
           title="Known regression: seed 20261127 / F1_F2_straight_1 / same XYZ and endpoints")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots/"known_stair_longitudinal.png",dpi=150)
    plt.close(fig)
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    selection = read_json(run / "validation_selection.json")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for name in selection["versions"]:
        history = read_json(run/name/"history.json")
        axes[0,0].plot([r["epoch"] for r in history],[r["loss"] for r in history],label=name)
        val = [r for r in history if r["validation"] is not None]
        axes[0,1].plot([r["epoch"] for r in val],[r["validation"]["cost_mae_raw"] for r in val],marker="o",label=name)
        axes[1,0].plot([r["epoch"] for r in val],[r["validation"]["stair_metrics"]["radii"]["0.45"]["break_pct"] for r in val],marker="o",label=name)
        axes[1,1].plot([r["epoch"] for r in val],[r["validation"]["parameter_response"]["stand_ratio"]["changed_delta_mae"] for r in val],marker="o",label=name)
    for axis,title in zip(axes.flat,["Training loss","Validation cost MAE (0-50)",
                                    "GT-connected stair pairs broken in prediction (%)","Stand response delta MAE (0-1)"]):
        axis.set(title=title,xlabel="Epoch")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots/"training_comparison.png",dpi=150)
    plt.close(fig)
    original=np.load(run/"original_test.npz")
    selected=np.load(run/"selected_test.npz")
    cmap=LinearSegmentedColormap.from_list("cost",["#169c50","#f5df43","#f0443c"])
    for scene in np.unique(selected["scene"]):
        mask=selected["scene"]==scene
        old_mask=original["scene"]==scene
        xyz=selected["coord"][mask]
        # Both evaluations use the same deterministic dataset/block order.
        fig=plt.figure(figsize=(17,6))
        data=[selected["target"][mask,3],original["prediction"][old_mask,3],selected["prediction"][mask,3]]
        for column,(cost,title) in enumerate(zip(data,["Native VST GT","Original checkpoint",selection["winner"]]),1):
            ax=fig.add_subplot(1,3,column,projection="3d")
            im=ax.scatter(*xyz.T,c=cost,cmap=cmap,vmin=0,vmax=1,s=2,linewidths=0)
            ax.set_box_aspect(np.ptp(xyz,axis=0))
            ax.view_init(30,-55)
            ax.set(title=title,xlabel="World X (m)",ylabel="World Y (m)",zlabel="Z (m)")
        fig.suptitle(f"{scene} | identical native query XYZ | [clearance,slope,step,stand]=[0.5,0.6,0.4,0.05]")
        fig.colorbar(im,ax=fig.axes,shrink=.6,pad=.03,label="Normalised cost 0-1")
        fig.savefig(plots/f"{scene}_gt_prediction.png",dpi=150)
        plt.close(fig)
    baseline=read_json(run/"original_test.json")
    final=read_json(run/"selected_test.json")
    history=read_json(run/selection["winner"]/"history.json")
    chosen=min((r for r in history if r["validation"] is not None),key=lambda r:tuple(r["validation"]["selection_key"]))
    summary=dict(winner=selection["winner"],best_epoch=chosen["epoch"],last_epoch=history[-1]["epoch"],
                 baseline_test=baseline,selected_test=final,
                 retention_eligible=not chosen["validation"]["selection_key"][0])
    summary["known_stair_thresholds"]=known_stair_report(run,plots)
    write_json(run/"summary.json",summary)
    rows=[]
    for label,value in [("Original",baseline),("Selected",final)]:
        stair=value["stair_metrics"]["radii"]["0.45"]
        rows.append(f"| {label} | {value['cost_mae_raw']:.3f} | {stair['break_pct']:.2f}% | "
                    f"{stair['false_pass_pct']:.2f}% | {value['parameter_response']['stand_ratio']['changed_delta_mae']:.4f} |")
    (run/"REPORT.md").write_text(
        "# 楼梯与机器人参数增强微调结果\n\n"
        f"验证选定方案：{selection['winner']}。最佳权重 epoch {chosen['epoch']}，最后权重 epoch {history[-1]['epoch']}。\n\n"
        "| 模型 | Test cost MAE / 50 | 楼梯断裂率 | 错误放行率 | stand 差值 MAE |\n"
        "|---|---:|---:|---:|---:|\n"+"\n".join(rows)+"\n\n"
        "连通指标基于固定原生查询点、0.45 m 诊断图及 cost 阈值 0.3。0.3 m 图结果保存在 JSON。\n\n"
        "模型选择使用验证集，测试场景在选定方案后评估。当前没有执行 GPMP 或机器人动态 rollout。\n\n"
        f"保留原模型能力的验证判据满足情况：{summary['retention_eligible']}。\n\n"
        f"最佳权重：{run/selection['winner']/'best.pt'}\n\n"
        f"完整预测点云：{run/'predictions'}\n\n"
        f"对比图：{plots}\n",encoding="utf-8")


if __name__=="__main__":
    main()
