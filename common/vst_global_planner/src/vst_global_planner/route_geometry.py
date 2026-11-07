"""Offline checks of a planned reference against the authoritative mesh."""
import numpy as np
import open3d as o3d
import trimesh


class RouteGeometry:
    def __init__(self,mesh_file,config):
        mesh=trimesh.load(mesh_file,force="mesh",process=False)
        self.caster=o3d.t.geometry.RaycastingScene(nthreads=4)
        legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                        o3d.utility.Vector3iVector(mesh.faces))
        self.caster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        self.radius=float(config["body_radius"])
        self.height=float(config["body_height"])
        self.step=float(config["max_step"])

    def check(self,path):
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        samples=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/.05))+1))
        dense=np.column_stack([np.interp(samples,arc,path[:,i]) for i in range(3)])
        expected=dense[:,2]-.1
        angles=np.arange(8)*2*np.pi/8
        offsets=np.vstack([[0.,0.],np.c_[np.cos(angles),np.sin(angles)]*self.radius*.75])
        xy=dense[:,None,:2]+offsets[None,:,:]
        origins=np.concatenate([xy,np.broadcast_to((expected+self.step+.02)[:,None,None],(*xy.shape[:2],1))],axis=2)
        directions=np.broadcast_to([0.,0.,-1.],origins.shape)
        rays=np.concatenate([origins,directions],axis=2).astype(np.float32)
        hit=self.caster.cast_rays(o3d.core.Tensor(rays))
        distance=hit["t_hit"].numpy()
        ground=origins[:,:,2]-distance
        valid=np.isfinite(ground)&(abs(ground-expected[:,None])<=self.step+.02)
        valid&=hit["primitive_normals"].numpy()[:,:,2]>.5
        support_ok=valid[:,0]&(valid.sum(axis=1)>=6)
        if not support_ok.all():
            raise ValueError(f"Mesh support check failed at {int((~support_ok).sum())}/{len(dense)} samples")
        support=ground[:,0]
        ring=np.arange(16)*2*np.pi/16
        ring_xy=np.c_[np.cos(ring),np.sin(ring)]*self.radius
        heights=np.linspace(self.step+.02,self.height,3)
        points=np.concatenate([np.c_[dense[:,:2]+d,(support+h)] for d in ring_xy for h in heights])
        points=np.vstack([points,np.c_[dense[:,:2],support+self.height]])
        signed=self.caster.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
        if np.any(signed<-.005):
            raise ValueError(f"Mesh body-clearance check failed at {int((signed<-.005).sum())} samples")
        return dict(passed=True,samples=len(dense),spacing_m=.05,minimum_signed_distance_m=float(signed.min()),
            support_height_error_max_m=float(abs(support-expected).max()),
            radius_m=self.radius,step_limit_m=self.step,body_top_above_support_m=self.height,
            body_check_lower_above_support_m=self.step+.02,
            scope="Offline sampled support and upper-body geometry near the native height reference. Not a dynamics or gait rollout.")

