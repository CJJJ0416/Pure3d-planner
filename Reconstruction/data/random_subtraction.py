"""Seeded, irregular surface deletion on terrain blocks. XYZ never changes."""
import numpy as np
from scipy.spatial import cKDTree


VERSION = 'noise_warped_surface_subtraction_v1'
DEFAULTS=dict(light_regions=[12,20],heavy_regions=[32,48],scale_m=[.1,.6],large_scale_m=[.6,1.],large_probability=.1,irregularity=.38,band_width_m=[.02,.08])


def noise_field(xyz, seed, scale):
    """Stationary multiscale random Fourier field, independent of point order."""
    rng=np.random.default_rng(seed)
    result=np.zeros(len(xyz),np.float64)
    for wavelength,amplitude in [(scale,.6),(scale*.5,.3),(scale*.25,.15)]:
        direction=rng.normal(size=(3,6));direction/=np.linalg.norm(direction,axis=0)
        phase=rng.uniform(0,2*np.pi,6)
        result+=amplitude*np.cos(xyz@direction*(2*np.pi/wavelength)+phase).sum(1)/np.sqrt(3)
    return result/np.sqrt(.6**2+.3**2+.15**2)


class SubtractionGenerator:
    """A generator seed owns a reproducible stream of independent mask seeds.

    Training may pass an explicit per-epoch sample_seed. Recompute PTv3 from
    that input if the mask changes. Never slice features encoded from full GT.
    """
    def __init__(self,seed,settings=None):
        self.rng=np.random.default_rng(seed)
        self.settings=dict(DEFAULTS,**(settings or {}))

    def sample(self,xyz,severity='heavy',sample_seed=None,center_bounds=None,spatial_index=None,target_removed_fraction=None):
        seed=int(self.rng.integers(0,2**32)) if sample_seed is None else int(sample_seed)
        rng=np.random.default_rng(seed)
        if spatial_index is None:
            tree=cKDTree(xyz)
            _,representatives=np.unique(np.floor(xyz/.1).astype(np.int64),axis=0,return_index=True)
        else:
            tree,representatives=spatial_index
        if center_bounds is not None:
            representatives=representatives[((xyz[representatives]>=center_bounds[0])&(xyz[representatives]<center_bounds[1])).all(1)]
        limits=self.settings[f'{severity}_regions']
        count=int(rng.integers(limits[0],limits[1]+1))
        weights=rng.dirichlet([3.,2.,2.]);removed=np.zeros(len(xyz),bool);regions=[]
        removed_count=0
        while (removed_count<len(xyz)*target_removed_fraction if target_removed_fraction is not None else len(regions)<count):
            kind=str(rng.choice(['irregular_hole','curved_band','soft_sparse'],p=weights))
            center_index=int(rng.choice(representatives));center=xyz[center_index].astype(np.float64)
            scale=float(rng.uniform(*self.settings['large_scale_m']) if rng.random()<self.settings['large_probability'] else rng.uniform(*self.settings['scale_m']))
            noise_seed=int(rng.integers(0,2**32));region=dict(kind=kind,center_index=center_index,center=center.tolist(),scale_m=scale,noise_seed=noise_seed)
            if kind=='irregular_hole':
                radius=scale/2
                ids=np.asarray(tree.query_ball_point(center,radius*1.65),np.int64)
                delta=xyz[ids]-center
                basis,_=np.linalg.qr(rng.normal(size=(3,3)))
                aspect=rng.uniform(.55,1.,3)
                distance=np.linalg.norm((delta@basis)/aspect,axis=1)
                boundary=radius*np.clip(1+self.settings['irregularity']*noise_field(delta,noise_seed,scale*.7),.25,1.6)
                chosen=ids[distance<boundary]
                region.update(aspect=aspect.tolist(),basis=basis.tolist())
            elif kind=='curved_band':
                nearby=xyz[tree.query(center,k=min(64,len(xyz)))[1]]
                _,axes=np.linalg.eigh(np.cov((nearby-center).T))
                normal=axes[:,0];basis=axes[:,1:]
                angle=rng.uniform(0,2*np.pi);steps=int(rng.integers(4,12));path=[np.zeros(2)]
                for step in range(steps):
                    angle+=rng.normal(0,.7)
                    path.append(path[-1]+scale/steps*np.array([np.cos(angle),np.sin(angle)]))
                path=np.asarray(path);path-=path.mean(0)
                width=float(rng.uniform(*self.settings['band_width_m']));radius=width/2
                ids=np.asarray(tree.query_ball_point(center,np.linalg.norm(path,axis=1).max()+.15),np.int64)
                delta=xyz[ids]-center;uv=delta@basis;distance=np.full(len(ids),np.inf)
                for a,b in zip(path[:-1],path[1:]):
                    vector=b-a;fraction=np.clip(((uv-a)@vector)/(vector@vector),0,1)
                    distance=np.minimum(distance,np.linalg.norm(uv-(a+fraction[:,None]*vector),axis=1))
                boundary=radius*np.clip(1+.4*noise_field(delta,noise_seed,.12),.25,1.8)
                chosen=ids[(distance<boundary)&(np.abs(delta@normal)<.05)]
                region.update(path_world=(path@basis.T+center).tolist(),width_m=width,normal=normal.tolist())
            else:
                radius=max(.1,scale/2)
                ids=np.asarray(tree.query_ball_point(center,radius*1.5),np.int64)
                delta=xyz[ids]-center
                d=np.linalg.norm(delta,axis=1)/radius
                soft=np.clip(1-d/1.5,0,1)**.5
                correlation=1/(1+np.exp(-1.3*noise_field(delta,noise_seed,scale*.5)))
                fraction=float(rng.uniform(.25,.5) if severity=='light' else rng.uniform(.5,.85))
                probability=np.clip(fraction*soft*(.4+1.2*correlation),0,1)
                chosen=ids[rng.random(len(ids))<probability]
                region.update(peak_deletion_probability=fraction)
            added=int((~removed[chosen]).sum())
            removed[chosen]=True;removed_count+=added
            region.update(selected_points=len(chosen),newly_removed_points=added);regions.append(region)
        metadata=dict(version=VERSION,seed=seed,severity=severity,region_count=len(regions),type_probabilities=weights.tolist(),target_removed_fraction=target_removed_fraction,
                      regions=regions,removed_points=int(removed.sum()),removed_fraction=float(removed.mean()),
                      coordinate_changes=False,settings=self.settings)
        return np.flatnonzero(~removed),np.flatnonzero(removed),metadata
