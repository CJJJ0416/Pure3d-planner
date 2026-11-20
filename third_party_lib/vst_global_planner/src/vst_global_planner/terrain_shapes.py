"""Curved, varying-width stair solids and real platform slab openings."""
import numpy as np
import trimesh
from manifold3d import Manifold,Mesh64,OpType,CrossSection,FillRule


def add_curved_flight(scene, flight, bend_degrees, width_ratios, add_prism):
    n, depth = flight["n_steps"], flight["step_depth"]
    run = (n+1)*depth
    angle = np.deg2rad(bend_degrees)
    x, y, direction = flight["x"], flight["y"], flight["direction"]
    base, rise = flight["base_z"], flight["riser"]
    thickness = scene["config"]["stair_structure"]["tread_thickness"]
    meshes = []

    def frame(q):
        q = np.asarray(q,float)
        theta = angle*q/run
        # sinc form covers straight and curved flights with one continuous formula.
        dx = q*(theta/2)*np.sinc(theta/(2*np.pi))**2
        dy = direction*q*np.sinc(theta/np.pi)
        center = np.stack((x+dx,y+dy),axis=-1)
        normal = np.stack((np.cos(theta),-direction*np.sin(theta)),axis=-1)
        width = flight["width"]*np.interp(np.clip(q/run,0,1),[0,.5,1],width_ratios)
        return center,normal,width

    def footprint(q0,q1):
        c,norm,w=frame(np.array([q0,q1]))
        left,right=c-norm*w[:,None]/2,c+norm*w[:,None]/2
        return np.array([left[0],right[0],right[1],left[1]])

    for j in range(n):
        top=base+j*rise
        add_prism(scene,meshes,footprint(j*depth,(j+1)*depth),
                  [0,0,top],[0,0,top-thickness],"stair",flight_id=flight["id"],tread_index=j)
    top_footprints=[footprint(n*depth,run),footprint(run,flight["length"])]

    # Continuous lofted beams avoid internal seams at every curved segment.
    qs=np.linspace(0,run,max(2,int(np.ceil(run/.12)))+1)
    center,norm,width=frame(qs)
    z=base+(qs/depth-1)*rise-thickness+.02
    for side in (-1,1):
        edge=center+side*norm*(width[:,None]/2-.025)
        left,right=edge-norm*.05,edge+norm*.05
        vertices=np.stack((np.c_[left,z-.24],np.c_[right,z-.24],
                           np.c_[right,z],np.c_[left,z]),axis=1).reshape(-1,3)
        faces=[]
        for j in range(len(qs)-1):
            for k in range(4):
                a,b=4*j+k,4*j+(k+1)%4
                c,d=a+4,b+4
                faces.extend([[a,b,d],[a,d,c]])
        end=4*(len(qs)-1)
        faces.extend([[0,2,1],[0,3,2],[end,end+1,end+2],[end,end+2,end+3]])
        beam=trimesh.Trimesh(vertices,np.asarray(faces),process=False)
        beam.fix_normals()
        meshes.append(beam)
    q=(np.arange(n+1)+.5)*depth
    centers,_,widths=frame(q)
    probes=np.c_[centers,base+np.arange(n+1)*rise]
    flight.update(construction="curved_floating",bend_degrees=float(bend_degrees),
                  tread_widths=widths.tolist(),surface_probe_local_xyz=probes.tolist(),
                  centerline_local_xy=frame(np.linspace(0,run,97))[0].tolist())
    attach,normal,_=frame(n*depth)
    tangent=np.array([-direction*normal[1],direction*normal[0]])
    return meshes,probes,attach,normal,tangent,top_footprints


def slab_from_footprints(footprints,height,slab):
    """Union horizontal outlines before extruding a single constant-height slab."""
    section=CrossSection(footprints,FillRule.NonZero)
    plate=section.extrude(slab).translate([0,0,height-slab]).to_mesh64()
    return trimesh.Trimesh(plate.vert_properties[:,:3],plate.tri_verts,process=False)


def cut_platform(plate, lo, hi, height, slab, mode, size_fraction, offset_fraction):
    lo,hi=np.asarray(lo,float),np.asarray(hi,float)
    size=hi-lo
    center=lo+size*np.array([.5+offset_fraction[0],.55+offset_fraction[1]])
    hole_size=size*np.asarray(size_fraction)
    records=[]
    cutters=[]
    if mode=="solid":
        return plate,records
    centers=[center]
    if mode=="double_hole":
        hole_size=hole_size*np.array([.5,.8])
        centers=[center+[-size[0]*.23,0],center+[size[0]*.23,0]]
    for c in centers:
        c=np.asarray(c,float)
        if mode=="ellipse":
            cutter=trimesh.creation.cylinder(radius=.5,height=slab+.2,sections=48)
            cutter.apply_scale([*hole_size,1.])
            cutter.apply_translation([*c,height-slab/2])
            kind="ellipse"
        else:
            lower,upper=c-hole_size/2,c+hole_size/2
            if mode=="notch":
                lower[1]=hi[1]-hole_size[1]
                upper[1]=hi[1]+.1
                c[1]=hi[1]-hole_size[1]/2
            cutter=trimesh.creation.box([*(upper-lower),slab+.2],
                trimesh.transformations.translation_matrix([*((lower+upper)/2),height-slab/2]))
            kind="notch" if mode=="notch" else "rectangle"
        cutters.append(cutter)
        records.append(dict(kind=kind,construction_center_xy=c.tolist(),size_xy_m=hole_size.tolist()))
    return trimesh.boolean.difference([plate,*cutters],engine="manifold"),records


def union_buildings(buildings, base):
    """Keep each boolean solid intact until the final mesh export."""
    def solid(mesh):
        return Manifold(Mesh64(
            np.array(mesh.vertices,dtype=np.float64,order="C",copy=True),
            np.array(mesh.faces,dtype=np.uint64,order="C",copy=True)))

    structures=[Manifold.batch_boolean([solid(mesh) for mesh in parts],OpType.Add)
                for parts in buildings]
    merged=Manifold.batch_boolean([*structures,solid(base)],OpType.Add).to_mesh64()
    return trimesh.Trimesh(merged.vert_properties[:,:3],merged.tri_verts,process=False)
