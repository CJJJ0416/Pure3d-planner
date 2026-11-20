"""Seeded building yaw and rectangle packing without resizing the geometry."""
import numpy as np


def _pack(sizes,container,order):
    free=[(0.,0.,float(container[0]),float(container[1]))]
    placed={}
    eps=1e-8
    for name in order:
        w,h=sizes[name]
        choices=[]
        for rect in free:
            x,y,rw,rh=rect
            if w<=rw+eps and h<=rh+eps:
                choices.append(((min(rw-w,rh-h),max(rw-w,rh-h),y,x),rect))
        if not choices:
            return None
        _,(x,y,_,_)=min(choices)
        placed[name]=np.array([x,y])
        split=[]
        for fx,fy,fw,fh in free:
            if x>=fx+fw-eps or x+w<=fx+eps or y>=fy+fh-eps or y+h<=fy+eps:
                split.append((fx,fy,fw,fh))
                continue
            if x>fx+eps: split.append((fx,fy,x-fx,fh))
            if x+w<fx+fw-eps: split.append((x+w,fy,fx+fw-x-w,fh))
            if y>fy+eps: split.append((fx,fy,fw,y-fy))
            if y+h<fy+fh-eps: split.append((fx,y+h,fw,fy+fh-y-h))
        split=list(dict.fromkeys(split))
        free=[a for i,a in enumerate(split) if not any(i!=j and
            a[0]>=b[0]-eps and a[1]>=b[1]-eps and
            a[0]+a[2]<=b[0]+b[2]+eps and a[1]+a[3]<=b[1]+b[3]+eps
            for j,b in enumerate(split))]
    return placed


def random_layout(groups,zone_groups,cfg,zone_probe_points):
    names=list(zone_groups)
    rng=np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]),42719]))
    gap=float(cfg["minimum_structure_gap_m"])
    edge=float(cfg["layout_edge_margin_m"])
    container=np.array(cfg["size_m"],float)-2*edge+gap
    vertices={name:np.vstack([m.vertices for group in zone_groups[name] for m in groups[group]])
              for name in names}
    for name,points in zone_probe_points.items():
        if len(points):
            vertices[name]=np.vstack([vertices[name],points])
    pivots={name:(points[:,:2].min(axis=0)+points[:,:2].max(axis=0))/2
            for name,points in vertices.items()}
    for attempt in range(int(cfg["layout_attempts"])):
        angles={name:float(rng.uniform(0.,2*np.pi)) for name in names}
        rotations={name:np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
                   for name,yaw in angles.items()}
        bounds={}
        for name,points in vertices.items():
            xy=(points[:,:2]-pivots[name])@rotations[name].T
            bounds[name]=np.array([xy.min(axis=0),xy.max(axis=0)])
        sizes={name:np.diff(bounds[name],axis=0)[0]+gap for name in names}
        orders=[sorted(names,key=lambda n:float(np.prod(sizes[n])),reverse=True),
                sorted(names,key=lambda n:float(max(sizes[n])),reverse=True)]
        orders.extend([list(rng.permutation(names)) for _ in range(16)])
        for order in orders:
            positions=_pack(sizes,container,order)
            if positions is None:
                continue
            extents=np.array([positions[n]+sizes[n] for n in names]).max(axis=0)
            origin=-extents/2+gap/2
            transforms={}
            for name in names:
                desired_min=positions[name]+origin
                shift=desired_min-bounds[name][0]-pivots[name]@rotations[name].T
                transform=np.eye(4)
                transform[:2,:2]=rotations[name]
                transform[:2,3]=shift
                transforms[name]=dict(matrix=transform,yaw_rad=angles[name],pivot=pivots[name])
            return transforms,dict(attempts=attempt+1,edge_margin_m=edge,
                sampling="Independent uniform yaw proposals in [0, 360), accepted when the compact layout fits")
    raise ValueError("Random building orientations did not fit. Increase ground size or change the layout seed.")


def transform_xy(points,transform):
    matrix=np.asarray(transform)
    return np.asarray(points)@matrix[:2,:2].T+matrix[:2,3]

