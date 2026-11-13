"""Identity-preserving solid union and role definitions for Filter Mesh data."""
import manifold3d
import numpy as np
import trimesh

SEMANTICS = {
    "floor": 0, "wall": 1, "ceiling": 2, "stair": 3,
    "ramp": 4, "railing": 5, "column_beam": 6,
    "large_block_obstacle": 7, "person_proxy": 8,
    "chair": 9, "movable_clutter": 10,
}
COLORS = np.array([
    [120, 160, 185], [165, 178, 195], [190, 201, 215], [225, 153, 62],
    [57, 165, 158], [45, 100, 150], [139, 120, 166], [102, 157, 101],
    [219, 77, 116], [184, 77, 179], [230, 93, 64],
], dtype=np.uint8)


def surface_union(parts):
    """Union solids while retaining their component IDs on output triangles.

    Manifold's original-ID runs carry the source component through Boolean
    splitting. Internal contact faces are removed before surface sampling.
    """
    solids, identities = [], {}
    for part in parts:
        leaves = part["mesh"].split(only_watertight=False)
        first = manifold3d.Manifold.reserve_ids(len(leaves))
        for offset, leaf in enumerate(leaves):
            source_id = first + offset
            data = manifold3d.Mesh(
                vert_properties=np.asarray(leaf.vertices, dtype=np.float32),
                tri_verts=np.asarray(leaf.faces, dtype=np.uint32),
                run_index=np.array([0, leaf.faces.size], dtype=np.uint32),
                run_original_id=np.array([source_id], dtype=np.uint32),
            )
            solids.append(manifold3d.Manifold(data))
            identities[source_id] = (part["instance_id"], part["semantic_id"], part["target_role"])
    union = manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add)
    result = union.to_mesh()
    mesh = trimesh.Trimesh(np.asarray(result.vert_properties)[:, :3],
                           np.asarray(result.tri_verts), process=False)
    runs = np.asarray(result.run_index, dtype=np.int64) // 3
    values = np.concatenate([
        np.tile(identities[int(source_id)], (end - start, 1))
        for source_id, start, end in zip(result.run_original_id, runs[:-1], runs[1:])
    ])
    metadata = dict(instance_id=values[:, 0].astype(np.int32),
                    semantic_id=values[:, 1].astype(np.int32),
                    target_role=values[:, 2].astype(np.uint8))
    # Boolean cuts can create zero-width slivers at coincident tile edges.
    # Remove those faces with their metadata before sampling or exporting IDs.
    nondegenerate = mesh.nondegenerate_faces(height=1e-6)
    mesh.metadata["degenerate_faces_removed"] = int((~nondegenerate).sum())
    mesh.update_faces(nondegenerate)
    mesh.remove_unreferenced_vertices()
    metadata = {key: value[nondegenerate] for key, value in metadata.items()}
    mesh.visual.face_colors = COLORS[metadata["semantic_id"]]
    return mesh, metadata
