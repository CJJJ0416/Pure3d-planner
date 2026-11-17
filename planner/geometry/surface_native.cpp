// Cost-only sparse surface graph. XYZ organizes layers; it never determines
// traversability through slope, step height, clearance or body collision.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Key {int64_t x,y,z;bool operator==(const Key&o)const{return x==o.x&&y==o.y&&z==o.z;}};
uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
struct Hash {size_t operator()(const Key&k)const{return mix(k.x)^mix(k.y+0x9e3779b97f4a7c15ULL)^mix(k.z+0x243f6a8885a308d3ULL);}};
bool lesskey(const Key&a,const Key&b){return a.x!=b.x?a.x<b.x:(a.y!=b.y?a.y<b.y:a.z<b.z);}
struct Sum {
 double x=0,y=0,z=0,c=0,lo=INFINITY,hi=-INFINITY;uint64_t n=0;
 void add(double a,double b,double d,double v){x+=a;y+=b;z+=d;c+=v;++n;lo=std::min(lo,d);hi=std::max(hi,d);}
 void add(const Sum&o){x+=o.x;y+=o.y;z+=o.z;c+=o.c;n+=o.n;lo=std::min(lo,o.lo);hi=std::max(hi,o.hi);}
};
struct Voxel {Key key;Sum all,low;};
struct Patch {Key key;Sum sum;double x,y,z,c;uint32_t node;};
struct Column {std::vector<uint32_t> patches;};
struct Data {
 double xy,zres,threshold,span,slice_dh;
 std::vector<Voxel> voxels;std::vector<Patch> patches;
 std::unordered_map<Key,Column,Hash> columns;
 std::vector<std::pair<uint32_t,uint32_t>> edges;
 std::vector<uint64_t> offsets;std::vector<uint32_t> neighbors;
 std::array<uint64_t,12> stats{};std::array<double,5> times{};
 Column* col(int64_t x,int64_t y){auto it=columns.find({x,y,0});return it==columns.end()?nullptr:&it->second;}
 uint32_t match_best_layer(Column*column,double target_height){
  if(!column||!std::isfinite(target_height))return UINT32_MAX;
  uint32_t best=UINT32_MAX;double best_cost=INFINITY,best_diff=INFINITY;
  for(auto i:column->patches){auto&p=patches[i];double diff=std::abs(p.z-target_height);
   if(!std::isfinite(p.z)||!std::isfinite(p.c)||p.c<0||p.c>=threshold||diff>slice_dh+1e-6)continue;
   if(p.c<best_cost||(p.c==best_cost&&diff<best_diff)){best=i;best_cost=p.c;best_diff=diff;}
  }
  return best;
 }
 bool paired(const Patch&a,const Patch&b){
  return match_best_layer(col(b.key.x,b.key.y),a.z)==b.node&&match_best_layer(col(a.key.x,a.key.y),b.z)==a.node;
 }
 bool corner(const Patch&a,const Patch&b,int64_t x,int64_t y){
  auto column=col(x,y);if(!column)return false;
  for(auto id:column->patches){auto&p=patches[id];if(paired(a,p)&&paired(p,b))return true;}
  return false;
 }
 bool connected(uint32_t a,uint32_t b){
  return a==b||std::binary_search(neighbors.begin()+offsets[a],neighbors.begin()+offsets[a+1],b);
 }
};
thread_local std::string error;
void build(Data&d,const float*xyz,const float*costs,uint64_t n){
 auto started=Clock::now();auto tick=started;
 std::unordered_map<Key,uint32_t,Hash> index;index.reserve(std::min<uint64_t>(n,1000000));
 d.voxels.reserve(std::min<uint64_t>(n,1000000));d.stats[0]=n;
 const double bound=double(INT64_MAX)/4;
 for(uint64_t i=0;i<n;++i){
  double x=xyz[3*i],y=xyz[3*i+1],z=xyz[3*i+2],c=costs[i];
  if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||!std::isfinite(c)||c<0){++d.stats[1];continue;}
  double ix=std::floor(x/d.xy),iy=std::floor(y/d.xy),iz=std::floor(z/d.zres);
  if(std::abs(ix)>bound||std::abs(iy)>bound||std::abs(iz)>bound)throw std::runtime_error("Voxel coordinate exceeds int64 range");
  Key key{int64_t(ix),int64_t(iy),int64_t(iz)};auto inserted=index.emplace(key,uint32_t(d.voxels.size()));
  if(inserted.second)d.voxels.push_back({key,{},{}});
  auto&v=d.voxels[inserted.first->second];v.all.add(x,y,z,c);
  if(c<d.threshold){v.low.add(x,y,z,c);++d.stats[2];}
 }
 index.clear();index.rehash(0);
 std::sort(d.voxels.begin(),d.voxels.end(),[](const Voxel&a,const Voxel&b){return lesskey(a.key,b.key);});
 d.stats[3]=d.voxels.size();d.times[0]=elapsed(tick);tick=Clock::now();
 d.columns.reserve(d.voxels.size());
 for(size_t first=0;first<d.voxels.size();){
  auto key=d.voxels[first].key;size_t end=first;
  while(end<d.voxels.size()&&d.voxels[end].key.x==key.x&&d.voxels[end].key.y==key.y)++end;
  Sum group;int64_t previous=INT64_MIN;Key bottom{};
  auto flush=[&](){if(!group.n)return;++d.stats[4];uint32_t id=d.patches.size();
   d.patches.push_back({bottom,group,group.x/group.n,group.y/group.n,group.z/group.n,group.c/group.n,id});
   d.columns[{key.x,key.y,0}].patches.push_back(id);
  };
  for(size_t i=first;i<end;++i){auto&v=d.voxels[i];if(!v.low.n)continue;
   // Splitting a long run retains every cost-eligible voxel. It never rejects
   // vertical structures or adds missing observations between separate runs.
   if(group.n&&(v.key.z>previous+1||std::max(group.hi,v.low.hi)-std::min(group.lo,v.low.lo)>d.span)){flush();group=Sum{};}
   if(!group.n)bottom=v.key;group.add(v.low);previous=v.key.z;
  }
  flush();first=end;
 }
 d.times[1]=elapsed(tick);tick=Clock::now();
 const int directions[4][2]={{1,0},{0,1},{1,1},{1,-1}};
 for(auto&a:d.patches)for(auto&dir:directions){auto column=d.col(a.key.x+dir[0],a.key.y+dir[1]);
  auto j=d.match_best_layer(column,a.z);if(j==UINT32_MAX){if(column)++d.stats[5];continue;}auto&b=d.patches[j];
  if(!d.paired(a,b)){++d.stats[9];continue;}
  if(dir[0]&&dir[1]&&(!d.corner(a,b,b.key.x,a.key.y)||!d.corner(a,b,a.key.x,b.key.y))){++d.stats[10];continue;}
  d.edges.emplace_back(a.node,b.node);
 }
 d.offsets.assign(d.patches.size()+1,0);
 for(auto&e:d.edges){++d.offsets[e.first+1];++d.offsets[e.second+1];}
 for(size_t i=1;i<d.offsets.size();++i)d.offsets[i]+=d.offsets[i-1];
 auto cursor=d.offsets;d.neighbors.resize(d.edges.size()*2);
 for(auto&e:d.edges){d.neighbors[cursor[e.first]++]=e.second;d.neighbors[cursor[e.second]++]=e.first;}
 for(size_t i=0;i<d.patches.size();++i)std::sort(d.neighbors.begin()+d.offsets[i],d.neighbors.begin()+d.offsets[i+1]);
 d.times[2]=elapsed(tick);d.times[3]=elapsed(started);
}
}
extern "C" {
const char*surface_error(){return error.c_str();}
void*surface_build(const float*xyz,const float*cost,uint64_t n,const double*p){
 try{auto d=std::make_unique<Data>();d->xy=p[0];d->zres=p[1];d->threshold=p[2];d->span=p[3];d->slice_dh=p[4];build(*d,xyz,cost,n);return d.release();}
 catch(const std::exception&e){error=e.what();return nullptr;}
}
void surface_free(void*ptr){delete static_cast<Data*>(ptr);}
void surface_sizes(void*ptr,uint64_t*out,double*times){auto&d=*static_cast<Data*>(ptr);
 std::copy(d.stats.begin(),d.stats.end(),out);out[12]=d.patches.size();out[13]=d.neighbors.size();out[14]=d.patches.size();std::copy(d.times.begin(),d.times.end(),times);}
void surface_copy_full(void*ptr,double*out){auto&d=*static_cast<Data*>(ptr);for(size_t i=0;i<d.voxels.size();++i){auto&s=d.voxels[i].all;
 double v[7]={s.x/s.n,s.y/s.n,s.z/s.n,s.c/s.n,double(s.n),s.lo,s.hi};std::copy(v,v+7,out+7*i);}}
void surface_copy_patches(void*ptr,double*out){auto&d=*static_cast<Data*>(ptr);for(size_t i=0;i<d.patches.size();++i){auto&p=d.patches[i];
 double v[9]={p.x,p.y,p.z,p.c,0.,0.,p.sum.lo,p.sum.hi,double(p.sum.n)};std::copy(v,v+9,out+9*i);}}
void surface_copy_graph(void*ptr,double*xyz,double*cost,uint64_t*offsets,uint32_t*neighbors,double*detail){auto&d=*static_cast<Data*>(ptr);
 for(size_t i=0;i<d.patches.size();++i){auto&p=d.patches[i];xyz[3*i]=p.x;xyz[3*i+1]=p.y;xyz[3*i+2]=p.z;cost[i]=p.c;
  double v[6]={double(p.key.x),double(p.key.y),double(p.key.z),p.sum.lo,p.sum.hi,0.};std::copy(v,v+6,detail+6*i);}
 std::copy(d.offsets.begin(),d.offsets.end(),offsets);std::copy(d.neighbors.begin(),d.neighbors.end(),neighbors);
}
// Only observed cost-cell membership and graph continuity are checked here.
// Height matching is a local adjacency window, not slope/step/clearance inference.
int64_t surface_match_layer(void*ptr,double x,double y,double target_height){auto&d=*static_cast<Data*>(ptr);
 if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(target_height)||std::abs(x/d.xy)>double(INT64_MAX)/4||std::abs(y/d.xy)>double(INT64_MAX)/4)return -1;
 auto id=d.match_best_layer(d.col(std::floor(x/d.xy),std::floor(y/d.xy)),target_height);
 return id==UINT32_MAX?-1:int64_t(id);
}
int surface_validate(void*ptr,const double*xyz,uint64_t n,double,uint64_t*out){auto&d=*static_cast<Data*>(ptr);uint32_t previous=UINT32_MAX;
 for(uint64_t i=0;i<n;++i){double x=xyz[3*i],y=xyz[3*i+1],z=xyz[3*i+2];
  if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)){out[0]=i;out[1]=5;return 0;}
  auto column=d.col(std::floor(x/d.xy),std::floor(y/d.xy));auto best=d.match_best_layer(column,z);
  if(best==UINT32_MAX){out[0]=i;out[1]=column?2:1;return 0;}
  if(previous!=UINT32_MAX&&!d.connected(previous,best)){out[0]=i;out[1]=3;return 0;}
  previous=best;
 }out[0]=n;out[1]=0;return 1;
}
}
