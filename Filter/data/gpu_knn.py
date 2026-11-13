"""Exact original six KNN statistics on CUDA, with no query-by-point matrix."""
import cupy as cp
import torch

from Filter.third_party.cupy_kdtree._kdtree_utils import asm_kd_tree,compute_knn


_COUNT_SOURCE=r'''
extern "C" __global__ void subtree_bounds(
    int n,int level,const double* tree,double* bounds) {
    int node=(1<<level)-1+blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=n || node>=((1<<(level+1))-1)) return;
    int left=2*node+1,right=left+1;
    for(int axis=0;axis<3;axis++) {
        double lo=tree[3*node+axis],hi=lo;
        if(left<n) {lo=fmin(lo,bounds[6*left+2*axis]);hi=fmax(hi,bounds[6*left+2*axis+1]);}
        if(right<n) {lo=fmin(lo,bounds[6*right+2*axis]);hi=fmax(hi,bounds[6*right+2*axis+1]);}
        bounds[6*node+2*axis]=lo;bounds[6*node+2*axis+1]=hi;
    }
}

__device__ long long radius_count(
    int n,int height,const double* tree,const double* bounds,
    const double* point,double radius2) {
    int curr=0,prev=-1;
    long long count=0;
    while(curr>=0) {
        int parent=(curr+1)/2-1,left=2*curr+1,right=left+1,next=parent;
        if(prev<left) {
            double nearest=0.,farthest=0.;
            for(int axis=0;axis<3;axis++) {
                double lo=bounds[6*curr+2*axis],hi=bounds[6*curr+2*axis+1],p=point[axis];
                double near=fmax(fmax(lo-p,p-hi),0.);
                double far=fmax(fabs(p-lo),fabs(p-hi));
                nearest+=near*near;farthest+=far*far;
            }
            if(farthest<=radius2) {
                int depth=31-__clz((unsigned int)(curr+1));
                long long span=1LL<<(height-depth),first=(curr+1)*span-1;
                long long last=n-first;
                last=last<0?0:(last>span?span:last);
                count+=span-1+last;
            } else if(nearest<=radius2) {
                double dx=point[0]-tree[3*curr],dy=point[1]-tree[3*curr+1],dz=point[2]-tree[3*curr+2];
                double distance=dx*dx+dy*dy+dz*dz;
                count+=distance<=radius2;
                if(left<n) next=left;
            }
        } else if(prev==left && right<n) next=right;
        prev=curr;curr=next;
    }
    return count;
}

extern "C" __global__ void two_radius_counts(
    int n,int queries,int height,const double* tree,const double* bounds,
    const double* points,long long* counts) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=queries) return;
    counts[2*row]=radius_count(n,height,tree,bounds,points+3*row,0.15*0.15)-1;
    counts[2*row+1]=radius_count(n,height,tree,bounds,points+3*row,0.5*0.5)-1;
}
'''
_COUNT_MODULE=cp.RawModule(code=_COUNT_SOURCE,options=('-std=c++17','--fmad=false'))


@torch.inference_mode()
def geometry_features_cuda(context,core,voxel=None):
    """Same feature order/units/self exclusion as data.features.geometry_features.

    Context is the original unscaled context window, not the full map. CuPy
    shares Torch's CUDA stream and tensor storage through DLPack. The KDTree
    and distance arithmetic use float64, matching SciPy's source coordinates.
    """
    device=context.device
    with cp.cuda.Device(device.index),cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        points=cp.from_dlpack(context).astype(cp.float64)
        query_indices=cp.from_dlpack(core)
        back=None
        if voxel is not None:
            cells=cp.floor(points/voxel).astype(cp.int64)
            cells-=cells.min(0);shape=cells.max(0)+1
            keys=(cells[:,0]*shape[1]+cells[:,1])*shape[2]+cells[:,2]
            occupied,inverse=cp.unique(keys,return_inverse=True)
            first=cp.full(len(occupied),len(points),dtype=cp.int32)
            cp.minimum.at(first,inverse,cp.arange(len(points),dtype=cp.int32))
            points=points[first]
            query_indices,back=cp.unique(inverse[query_indices],return_inverse=True)
        if len(points)==1:
            return torch.zeros((len(core),6),device=device,dtype=torch.float32)
        query=points[query_indices]
        tree,index=asm_kd_tree(points)
        box=cp.full(3,cp.inf,dtype=cp.float64);empty=cp.empty(0,dtype=cp.float64)
        distances,_=compute_knn(query,tree,index,box,empty,k=min(65,len(points)))
        distances=distances[:,1:]
        d16=distances[:,min(15,distances.shape[1]-1)]
        d64=distances[:,-1];mean=distances[:,:16].mean(1)
        bounds=cp.empty((len(points),3,2),dtype=cp.float64)
        height=len(points).bit_length()-1
        build=_COUNT_MODULE.get_function('subtree_bounds')
        for level in range(height,-1,-1):
            build(((2**level+127)//128,),(128,),(len(points),level,tree,bounds))
        counts=cp.empty((len(query),2),dtype=cp.int64)
        count=_COUNT_MODULE.get_function('two_radius_counts')
        count(((len(query)+127)//128,),(128,),(len(points),len(query),height,tree,bounds,query,counts))
        features=cp.column_stack((mean,d16,d64,counts,mean/d64)).astype(cp.float32)
        if back is not None: features=features[back]
        return torch.from_dlpack(features)
