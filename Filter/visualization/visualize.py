"""Render D0 Mesh roles and D1 clean/rough PCD with matching map views."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from common.geneator.filter_components import COLORS, SEMANTICS
from common.pointcloud_io import read_pcd_xyz
from Filter.data.augment import SOURCE_ROLES, SOURCE_COLORS
from utils.pointcloud import read_json, workspace_path


BACKGROUND = "#f6f8fb"


def set_view(axis, bounds, title):
    low, high = bounds
    span = high - low
    axis.set(xlim=(low[0] - .1, high[0] + .1), ylim=(low[1] - .1, high[1] + .1),
             zlim=(low[2] - .1, high[2] + .1), xlabel="Map X (m)",
             ylabel="Map Y (m)", zlabel="Map Z (m)", title=title)
    axis.set_box_aspect(span + .2)
    axis.view_init(elev=27, azim=-125)
    axis.set_proj_type("ortho")
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(labelsize=8)


def role_legend(codes):
    return [Patch(color=SOURCE_COLORS[code], label=name.replace("_", " "))
            for name, code in SOURCE_ROLES.items() if code in codes]


def point_view(axis, coord, semantic, colors, bounds, title, groups=None, marker_size=.6):
    # Drawing alone hides the ceiling and limits marker count. PCD files and
    # all provenance arrays retain every sampled return, including ceilings.
    indices = np.flatnonzero(semantic != SEMANTICS["ceiling"])
    if groups is None:
        indices = indices[::max(1, len(indices) // 80000)]
    else:
        indices = np.concatenate([part[::max(1, len(part) // 35000)]
            for code in np.unique(groups[indices])
            for part in [indices[groups[indices] == code]]])
    values = coord[indices]
    rgba = np.c_[colors[indices], np.full(len(indices), .8)]
    wall_context = semantic[indices] == SEMANTICS["wall"]
    if groups is not None:
        wall_context &= groups[indices] <= 1
    rgba[wall_context, 3] = .16
    axis.scatter(values[:, 0], values[:, 1], values[:, 2], c=rgba, s=marker_size,
                 depthshade=False, linewidths=0, rasterized=True)
    set_view(axis, bounds, title)


def render_overview(folder, mesh, metadata, scene):
    visible = metadata["semantic_id"] != SEMANTICS["ceiling"]
    faces = mesh.triangles[visible]
    semantic = metadata["semantic_id"][visible]
    target = metadata["target_role"][visible]
    fig = plt.figure(figsize=(15, 8), facecolor=BACKGROUND)
    palettes = [COLORS[semantic] / 255., np.where(target[:, None] == 1,
                np.array([[.24, .64, .51]]), np.array([[.88, .24, .49]]))]
    for column, (palette, title) in enumerate(zip(palettes, ["Mesh | component identity", "Mesh | all stationary surfaces retained"])):
        axis = fig.add_subplot(1, 2, column + 1, projection="3d")
        rgba = np.c_[palette, np.ones(len(palette))]
        rgba[semantic == SEMANTICS["wall"], 3] = .15
        axis.add_collection3d(Poly3DCollection(faces, facecolors=rgba, edgecolors="none", zsort="average"))
        set_view(axis, np.asarray(scene["bounds_world"]), title)
    semantic_handles = [Patch(color=COLORS[code] / 255., label=name.replace("_", " "))
        for name, code in SEMANTICS.items() if code in semantic]
    fig.legend(handles=semantic_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .035))
    name = "WFC multi-floor terrain + corridors" if scene["family"] == "wfc_terrain" else "identity-labelled stair room"
    if scene.get('kind')=='block_ghost_example': name='complete local stair/platform component'
    fig.suptitle(f"Filter D0 | {name} | seed {scene['seed']}\n"
                 "Building and stationary objects are both static reference", fontsize=17, y=.97)
    fig.text(.5, .015, "Display cutaway: ceiling hidden, walls translucent. Exported Mesh and PCD contain the full geometry.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.02, right=.97, bottom=.17, top=.86, wspace=.04)
    fig.savefig(folder / "overview.png", dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def render_corridors(folder, mesh, metadata, scene):
    corridors = scene["corridors"]
    cfg = read_json(folder / "wfc_config.resolved.json")
    frame = np.asarray(scene["transform"])
    centers = mesh.triangles_center
    local = (centers - frame[:3, 3]) @ frame[:3, :3]
    fig = plt.figure(figsize=(7 * len(corridors), 8), facecolor=BACKGROUND)
    for column, corridor in enumerate(corridors):
        cells = np.asarray(corridor["cells"])
        tile = cfg["tile_dim_m"][0]
        x = (cells[:, 1].mean() - (cfg["shape"][1] - 1) / 2) * tile
        y = ((cfg["shape"][0] - 1) / 2 - cells[:, 0].mean()) * tile
        z = cfg["floor_z_m"][corridor["floor"]]
        length, width = corridor["length_m"], corridor["width_m"]
        half = np.array([length / 2 + .4, width / 2 + .8] if corridor["axis"] == "x" else [width / 2 + .8, length / 2 + .4])
        selected = ((np.abs(local[:, :2] - [x, y]) < half).all(1)
                    & (local[:, 2] > z - .2) & (local[:, 2] < z + corridor["height_m"] + .2))
        faces = mesh.triangles[selected]
        semantic = metadata["semantic_id"][selected]
        rgba = np.c_[COLORS[semantic] / 255., np.ones(len(semantic))]
        rgba[semantic == SEMANTICS["wall"], 3] = .3
        rgba[semantic == SEMANTICS["ceiling"], 3] = .12
        axis = fig.add_subplot(1, len(corridors), column + 1, projection="3d")
        axis.add_collection3d(Poly3DCollection(faces, facecolors=rgba, edgecolors="none"))
        bounds = np.array([faces.reshape(-1, 3).min(0), faces.reshape(-1, 3).max(0)])
        set_view(axis, bounds, f"Floor {corridor['floor']} | {length:.1f} × {width:.1f} m\n"
                 f"Clear height {corridor['height_m']:.2f} m | two side doorways")
    fig.suptitle(f"WFC corridor additions | seed {scene['seed']} | flat reserved WFC cells", fontsize=17, y=.97)
    fig.text(.5, .035, "Walls and ceilings are translucent for inspection. Stationary object controls are included inside each corridor.", ha="center", fontsize=11)
    fig.subplots_adjust(left=.02, right=.98, top=.85, bottom=.12, wspace=.08)
    fig.savefig(folder / "corridors.png", dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def render_comparison(folder, variant, coord, semantic, roles, target, bounds, name):
    """Identical XYZ and camera across input, provenance, GT and retained rows."""
    fig = plt.figure(figsize=(17, 13), facecolor=BACKGROUND)
    blue = np.tile([.25, .48, .8], (len(coord), 1))
    selections = [np.ones(len(coord), bool)] * 3 + [target >= .5]
    colors = [blue, SOURCE_COLORS[roles], plt.get_cmap('RdYlGn')(target)[:, :3],
              plt.get_cmap('RdYlGn')(target)[:, :3]]
    for index, title in enumerate(['Input XYZ', 'Source provenance (GT only)',
                                   'Continuous retention target', 'Retained by GT >= 0.5']):
        selected = selections[index]
        axis = fig.add_subplot(2, 2, index + 1, projection='3d')
        # Keep the same display rows in the first three views.
        point_view(axis, coord[selected], semantic[selected], colors[index][selected], bounds,
                   f'{title} | {int(selected.sum()):,} points')
    fig.suptitle(f'Filter trails v2 | variant {variant:03d} | {name}\n'
                 'Single fused XYZ input. Colors and labels are not model inputs.', fontsize=16, y=.98)
    fig.legend(handles=role_legend(np.unique(roles)), loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .045))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='RdYlGn'),
                 cax=fig.add_axes([.92, .27, .015, .46]), label='keep_target')
    fig.text(.5, .012, 'Same map coordinates, camera and bounds. Ceiling hidden and display markers subsampled only. PCD files retain all points.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.02, right=.88, bottom=.12, top=.91, hspace=.15, wspace=.06)
    fig.savefig(folder / f'{name}_{variant:03d}.png', dpi=130, facecolor=BACKGROUND)
    plt.close(fig)


def render_details(folder, variant, clean, rough, record):
    coord, role = rough['coord'], rough['source_role']
    # Each type gets independent close-ups, so dense building context cannot
    # hide sparse motion trails or correct disconnected fragments.
    groups = [
        ('dynamic_trails', record['dynamic_trails'], role == SOURCE_ROLES['dynamic_object_trail']),
        ('registration_ghosts', record['registration_trails'], role == SOURCE_ROLES['registration_trail']),
        ('static_controls', record['disconnected_controls'], rough['is_disconnected_control']),
    ]
    for name, entries, mask in groups:
        if not entries:
            continue
        fig = plt.figure(figsize=(18, 6), facecolor=BACKGROUND)
        # The first corridor supplies person, chair and movable-object examples.
        for column, entry in enumerate(entries[:3]):
            anchor = np.asarray(entry['anchor_xyz'] if 'anchor_xyz' in entry else entry['center'])
            if name == 'static_controls':
                close = np.linalg.norm(coord - anchor, axis=1) < entry['radius_m'] + entry['gap_m'] + .2
                selection = mask & close
            else:
                selection = mask & (rough['trail_id'] == entry['id'])
            points = coord[selection]
            low, high = points.min(0) - .35, points.max(0) + .35
            if name == 'static_controls':
                context = coord[((coord >= low) & (coord <= high)).all(1) & ~mask]
            else:
                context = clean[((clean >= low) & (clean <= high)).all(1)][::4]
            axis = fig.add_subplot(1, 3, column + 1, projection='3d')
            axis.scatter(*context.T, color='#b0b7c0', s=.7, alpha=.15, depthshade=False, linewidths=0)
            color = rough['keep_target'][selection] if name == 'static_controls' else rough['trail_step'][selection]
            scatter = axis.scatter(*points.T, c=color, cmap='RdYlGn' if name == 'static_controls' else 'plasma',
                                   vmin=0, vmax=1 if name == 'static_controls' else entry['copies'],
                                   s=3 if name == 'static_controls' else 1.5, depthshade=False, linewidths=0)
            kind = {value: key for key, value in SEMANTICS.items()}[entry['semantic_id']]
            set_view(axis, np.array([low, high]), f'{kind} | {len(points):,} points')
            fig.colorbar(scatter, ax=axis, shrink=.5, pad=.02,
                         label='keep_target' if name == 'static_controls' else 'Accumulation step')
        context_label = 'observed context (gap retained)' if name == 'static_controls' else 'static reference'
        fig.suptitle(f'{name.replace("_", " ")} | variant {variant:03d}\n'
                     f'Array-generated supervision. Grey: {context_label}.', fontsize=16, y=.98)
        fig.subplots_adjust(left=.02, right=.98, top=.83, bottom=.12, wspace=.18)
        fig.savefig(folder / f'{name}_{variant:03d}.png', dpi=140, facecolor=BACKGROUND)
        plt.close(fig)


def render_block_example(folder, rough, window, record):
    """Render every full-resolution point. Box wires never determine GT colors."""
    from itertools import product
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    xyz,score=rough['coord'],rough['keep_target']
    core_lower,core_upper=np.asarray(window['core_lower']),np.asarray(window['core_upper'])
    context_lower,context_upper=np.asarray(window['context_lower']),np.asarray(window['context_upper'])
    bounds=np.array([np.minimum(xyz.min(0),context_lower),np.maximum(xyz.max(0),context_upper)])
    source_color=SOURCE_COLORS[rough['source_role']].copy()
    floating_ids=[entry['id'] for entry in record['floating_ghosts'] if entry['kind'] in ['vertical_scatter','volume_scatter']]
    source_color[np.isin(rough['ghost_placement_id'],floating_ids)]=[.84,.15,.7]
    source_color[rough['is_disconnected_control']]=[.0,.72,.85]
    colors=[np.tile([.22,.48,.78],(len(xyz),1)),source_color,
            plt.get_cmap('RdYlGn')(score)[:,:3],plt.get_cmap('RdYlGn')(score)[:,:3]]
    sizes=np.where(np.isin(rough['source_role'],[2,6]),4.5,.45)
    selections=[np.ones(len(xyz),bool)]*3+[score>=.5]
    titles=['Input XYZ | full points','Source identity | supervision only',
            'Continuous GT | complete Mesh reference','Retained by GT >= 0.5']
    for view in ['comparison','top','side']:
        fig=plt.figure(figsize=(17,12),facecolor=BACKGROUND)
        for column,(title,selected,palette) in enumerate(zip(titles,selections,colors)):
            ax=fig.add_subplot(2,2,column+1,projection='3d' if view=='comparison' else None)
            points=xyz[selected]
            if view=='comparison':
                ax.scatter(*points.T,c=palette[selected],s=sizes[selected],depthshade=False,linewidths=0,rasterized=True)
                set_view(ax,bounds,f'{title} | {len(points):,} points')
                ax.view_init(elev=26,azim=-55)
            else:
                dims=[0,1] if view=='top' else [1,2]
                ax.scatter(points[:,dims[0]],points[:,dims[1]],c=palette[selected],s=sizes[selected],linewidths=0,rasterized=True)
                ax.set(xlim=(bounds[0,dims[0]]-.1,bounds[1,dims[0]]+.1),
                       ylim=(bounds[0,dims[1]]-.1,bounds[1,dims[1]]+.1),
                       xlabel=f'Map {"XYZ"[dims[0]]} (m)',ylabel=f'Map {"XYZ"[dims[1]]} (m)',
                       title=f'{title} | {len(points):,} points')
                ax.set_aspect('equal');ax.set_facecolor(BACKGROUND);ax.grid(alpha=.16)
            for lower,upper,color in [(core_lower,core_upper,'#00a4bf'),(context_lower,context_upper,'#505b70')]:
                if view=='comparison':
                    corners=np.array(list(product(*zip(lower,upper))))
                    for i in range(8):
                        for bit in [1,2,4]:
                            j=i^bit
                            if i<j: ax.plot(*corners[[i,j]].T,color=color,lw=.8,alpha=.8)
                else:
                    ax.add_patch(Rectangle(lower[dims],*(upper-lower)[dims],fill=False,
                                           edgecolor=color,linewidth=1.,linestyle='--'))
        handles=[Patch(color=SOURCE_COLORS[0],label='Accurate structure'),
            Patch(color=SOURCE_COLORS[1],label='Measurement noise'),
            Patch(color=SOURCE_COLORS[2],label='Sparse displaced points'),
            Patch(color=SOURCE_COLORS[6],label='Sparse ghost trails'),
            Patch(color=[.84,.15,.7],label='Floating scattered points'),
            Patch(color=[.0,.72,.85],label='Accurate disconnected controls'),
            Line2D([],[],color='#00a4bf',label='Core 4 x 4 x 3 m'),
            Line2D([],[],color='#505b70',label='Context 6 x 6 x 5 m')]
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.48,.035),fontsize=10)
        fig.suptitle(f'Complete block + ghosts inside/outside | seed {record["seed"]} | {view}\n'
            'Full surface samples and augmentation points. No voxel downsampling. Same XYZ, bounds and camera.',fontsize=16,y=.98)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='RdYlGn'),
            cax=fig.add_axes([.93,.22,.012,.53]),label='keep_target')
        fig.text(.5,.012,'Every point is plotted. Ghost markers are enlarged for visibility. Labels use the complete Mesh.',
            ha='center',fontsize=10)
        fig.subplots_adjust(left=.04,right=.9,bottom=.13,top=.9,hspace=.19,wspace=.12)
        fig.savefig(folder/f'{view}.png',dpi=150,facecolor=BACKGROUND)
        plt.close(fig)


def render_mesh_gt_demo(folder, mesh, clean, rough):
    """Show the complete geometric GT separately from same-row input scores."""
    coord,score=rough['coord'],rough['keep_target']
    bounds=np.array([coord.min(0)-.2,coord.max(0)+.2])
    ghost=rough['ghost_placement_id']>=0
    sizes=np.where(ghost,4.,.35)
    green=np.tile([.12,.62,.35],(len(clean),1))
    blue=np.tile([.24,.48,.8],(len(coord),1))
    gt_color=plt.get_cmap('RdYlGn')(score)[:,:3]
    fig=plt.figure(figsize=(16,12),facecolor=BACKGROUND)
    titles=['Clean generator Mesh | geometric reference',
        f'Mesh -> complete GT PCD | {len(clean):,} points',
        f'Intact structure + sparse ghosts | {len(coord):,} points',
        f'Mesh-derived continuous GT | {len(coord):,} same input points']
    for i,title in enumerate(titles):
        ax=fig.add_subplot(2,2,i+1,projection='3d')
        if i==0:
            ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolors='#a7b4c5',
                edgecolors='#63758b',linewidths=.15))
        elif i==1:
            ax.scatter(*clean.T,c=green,s=.35,depthshade=False,linewidths=0,rasterized=True)
        else:
            ax.scatter(*coord.T,c=blue if i==2 else gt_color,s=sizes,
                depthshade=False,linewidths=0,rasterized=True)
        set_view(ax,bounds,title);ax.view_init(elev=28,azim=-55)
    fig.suptitle('Generator Mesh as GT | complete geometry + sparse residuals\n'
        'No holes, stripe deletion or voxel downsampling. Every input point receives a continuous score.',
        fontsize=16,y=.98)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='RdYlGn'),
        cax=fig.add_axes([.935,.2,.013,.56]),label='keep_target | geometric supervision')
    fig.text(.5,.03,'Same map coordinates and camera. Red: low retention. Green: high retention. '
        'Ghost markers enlarged for visibility. No network prediction or point deletion.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.025,right=.9,bottom=.085,top=.9,hspace=.12,wspace=.08)
    fig.savefig(folder/'mesh_gt_comparison.png',dpi=150,facecolor=BACKGROUND)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,6),facecolor=BACKGROUND)
    for ax,points,palette,size,title in zip(axes,[clean,coord,coord],[green,blue,gt_color],
        [np.full(len(clean),.35),sizes,sizes],['Complete GT PCD','Intact input + sparse ghosts','Continuous GT at input XYZ']):
        ax.scatter(points[:,1],points[:,2],c=palette,s=size,linewidths=0,rasterized=True)
        ax.set(xlim=(bounds[0,1],bounds[1,1]),ylim=(bounds[0,2],bounds[1,2]),
            xlabel='Map Y (m)',ylabel='Map Z (m)',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_facecolor(BACKGROUND)
    fig.suptitle('Mesh GT versus labelled input | side view | all points',fontsize=16)
    fig.text(.5,.025,'Geometry is unchanged between input and its score labels. Missing-surface reconstruction is outside this demo.',
        ha='center',fontsize=10)
    fig.subplots_adjust(left=.04,right=.98,bottom=.14,top=.86,wspace=.18)
    fig.savefig(folder/'mesh_gt_side.png',dpi=150,facecolor=BACKGROUND)
    plt.close(fig)


def render_scene(folder):
    folder = Path(folder)
    scene = read_json(folder / 'scene.json')
    mesh = trimesh.load(folder / 'scene.obj', force='mesh', process=False)
    with np.load(folder / 'face_metadata.npz') as meta:
        render_overview(folder, mesh, meta, scene)
        render_corridors(folder, mesh, meta, scene)
    clean = read_pcd_xyz(folder / 'clean.pcd')
    for path in sorted(folder.glob('augmentation_[0-9][0-9][0-9].json')):
        variant = int(path.stem.rsplit('_', 1)[1])
        record = read_json(path)
        with np.load(folder / f'rough_attributes_{variant:03d}.npz') as values:
            rough = dict(values)
        coord = rough['coord']
        bounds = np.array([coord.min(0), coord.max(0)])
        render_comparison(folder, variant, coord, rough['semantic_id'], rough['source_role'],
                          rough['keep_target'], bounds, 'comparison')
        index = np.load(folder / f'windows_{variant:03d}.npz')
        windows = read_json(folder / f'windows_{variant:03d}.json')
        selected = windows['selected_window']
        a, b = index['offsets'][selected:selected + 2]
        ids = index['point_index'][a:b]
        window = windows['windows'][selected]
        render_comparison(folder, variant, coord[ids], rough['semantic_id'][ids], rough['source_role'][ids],
            rough['keep_target'][ids], np.array([window['context_lower'], window['context_upper']]), 'window')
        render_details(folder, variant, clean, rough, record)
        print(f'Plots | variant={variant:03d} | comparison/window/details', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    args = parser.parse_args()
    render_scene(workspace_path(args.scene))


if __name__ == '__main__':
    main()
