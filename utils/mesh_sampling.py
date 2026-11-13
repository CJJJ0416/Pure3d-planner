"""Deterministic samples of every source triangle, including thin structures."""
import numpy as np


def triangle_rows(triangles, spacing):
    """Use the longest edge as a base and place rows across its altitude."""
    lengths = np.linalg.norm(np.roll(triangles, -1, axis=1) - triangles, axis=2)
    first = lengths.argmax(axis=1)
    vertices = np.take_along_axis(
        triangles, ((first[:, None] + np.arange(3)) % 3)[:, :, None], axis=1,
    )
    origin = vertices[:, 0]
    base = vertices[:, 1] - origin
    side = vertices[:, 2] - origin
    width = np.linalg.norm(base, axis=1)
    altitude = np.linalg.norm(np.cross(base, side), axis=1) / width
    rows = np.maximum(1, np.ceil(altitude / spacing).astype(np.int64))
    face = np.repeat(np.arange(len(triangles)), rows + 1)
    row = np.arange(len(face)) - np.repeat(np.cumsum(rows + 1) - rows - 1, rows + 1)
    fraction = row / rows[face]
    segments = np.ceil(width[face] * (1 - fraction) / spacing).astype(np.int64)
    return origin, base, side, face, fraction, segments


def sample_surface(mesh, spacing, batch_faces=4096, *, return_face_ids=False):
    """Sample all triangles with row/segment spacing at most ``spacing`` metres.

    Triangle boundaries and original vertices are included, so small railing
    faces are retained even when their area is below spacing squared. This
    samples the source mesh as supplied, including contact/interior faces in
    concatenated solids. It does not perform a Boolean union or ray casting.

    With return_face_ids=True, return (XYZ, face_id, report). The selected
    source triangle follows each sample through sorting and deduplication.
    At shared edges the first sampled triangle supplies the identity.
    """
    triangles = mesh.triangles
    starts = range(0, len(triangles), batch_faces)
    counts = [
        int((triangle_rows(triangles[start:start + batch_faces], spacing)[-1] + 1).sum())
        for start in starts
    ]
    points = np.empty((sum(counts), 3), dtype=np.float32)
    face_ids = np.empty(len(points), dtype=np.int64) if return_face_ids else None
    offset = 0
    for start, count in zip(starts, counts):
        origin, base, side, face, fraction, segments = triangle_rows(
            triangles[start:start + batch_faces], spacing,
        )
        row = np.repeat(np.arange(len(face)), segments + 1)
        column = np.arange(count) - np.repeat(np.cumsum(segments + 1) - segments - 1, segments + 1)
        along = column / np.maximum(segments[row], 1)
        across = fraction[row]
        triangle = face[row]
        points[offset:offset + count] = (
            origin[triangle] + across[:, None] * side[triangle]
            + ((1 - across) * along)[:, None] * base[triangle]
        )
        if return_face_ids:
            face_ids[offset:offset + count] = start + triangle
        offset += count

    # Remove repeated triangle-edge samples without moving them to voxel centres.
    # Sorting this array in place avoids a second full-size coordinate buffer.
    np.round(points, decimals=6, out=points)
    if return_face_ids:
        _, first = np.unique(points, axis=0, return_index=True)
        return points[first], face_ids[first], {"samples_before_deduplication": len(points)}
    points.view([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).reshape(-1).sort()
    keep = np.empty(len(points), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points[keep], {"samples_before_deduplication": len(points)}
