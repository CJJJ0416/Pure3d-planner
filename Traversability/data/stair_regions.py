"""Stair-region membership for production training replay sampling."""
import numpy as np

def legacy_stair_mask(xyz, metadata):
    """Only identifies stair blocks for replay sampling, never gates their targets."""
    angle = np.deg2rad(metadata["config"]["yaw_deg"])
    c, s = np.cos(angle), np.sin(angle)
    xy = xyz[:, :2] @ np.array([[c, -s], [s, c]])
    selected = np.zeros(len(xyz), bool)
    for r in metadata["stairs"]:
        local = xy - r["center_local_xy"]
        a = np.deg2rad(r.get("rotation_deg", 0))
        ca, sa = np.cos(a), np.sin(a)
        local = local @ np.array([[ca, -sa], [sa, ca]])
        w, h = r["unrotated_footprint_m"]
        z0, z1 = [metadata["config"]["floor_z_m"][r[k]] for k in ("from_floor", "to_floor")]
        selected |= ((abs(local[:,0]) < w/2) & (abs(local[:,1]) < h/2)
                     & (xyz[:,2] > z0+.03) & (xyz[:,2] < z1+.1))
    return selected
