// Exact stable core grouping and max-cost display representatives in one scan.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace {
struct Key {int64_t x,y,z;bool operator==(const Key&b)const{return x==b.x&&y==b.y&&z==b.z;}};
uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
struct Hash {size_t operator()(const Key&k)const{return mix(k.x)^mix(k.y+0x9e3779b97f4a7c15ULL)^mix(k.z+0x243f6a8885a308d3ULL);}};
struct Group {Key key;uint64_t count,row;uint32_t original;double cost;};
struct Result {std::vector<Group> groups;std::vector<int64_t> rows;std::vector<uint64_t> offsets;};
thread_local std::string error;
double value(const void*p,int dtype,uint64_t i){return dtype==4?double(static_cast<const float*>(p)[i]):static_cast<const double*>(p)[i];}
template<bool Partition>std::unique_ptr<Result> scan(const void*xyz,int dtype,uint64_t xyz_stride,const void*cost,int cost_dtype,uint64_t cost_stride,uint64_t n,const double*size){
 auto result=std::make_unique<Result>();auto&groups=result->groups;
 std::unordered_map<Key,uint32_t,Hash> table;table.reserve(std::min<uint64_t>(n,Partition?4096:1000000));
 std::vector<uint32_t> membership;if(Partition)membership.resize(n);
 for(uint64_t i=0;i<n;++i){
  double x=value(xyz,dtype,xyz_stride*i),y=value(xyz,dtype,xyz_stride*i+1),z=value(xyz,dtype,xyz_stride*i+2);
  if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z))throw std::runtime_error("Nonfinite XYZ in array operation");
  double a=std::floor(x/size[0]),b=std::floor(y/size[1]),c=std::floor(z/size[2]);
  if(std::abs(a)>double(INT64_MAX)/4||std::abs(b)>double(INT64_MAX)/4||std::abs(c)>double(INT64_MAX)/4)throw std::runtime_error("Grid coordinate exceeds int64 range");
  Key key{int64_t(a),int64_t(b),int64_t(c)};double score=cost?value(cost,cost_dtype,cost_stride*i):0.;
  if(!std::isfinite(score))throw std::runtime_error("Nonfinite display cost");
  auto found=table.emplace(key,uint32_t(groups.size()));auto id=found.first->second;
  if(found.second)groups.push_back({key,0,i,id,score});
  auto&g=groups[id];++g.count;
  if(score>g.cost){g.cost=score;g.row=i;}
  if(Partition)membership[i]=id;
 }
 std::sort(groups.begin(),groups.end(),[](const Group&a,const Group&b){return a.key.x!=b.key.x?a.key.x<b.key.x:(a.key.y!=b.key.y?a.key.y<b.key.y:a.key.z<b.key.z);});
 result->offsets.resize(groups.size()+1);result->offsets[0]=0;
 for(size_t i=0;i<groups.size();++i)result->offsets[i+1]=result->offsets[i]+groups[i].count;
 if(Partition){
  result->rows.resize(n);std::vector<uint64_t>cursor(groups.size());
  for(size_t i=0;i<groups.size();++i)cursor[groups[i].original]=result->offsets[i];
  for(uint64_t i=0;i<n;++i)result->rows[cursor[membership[i]]++]=i;
 }else{
  result->rows.resize(groups.size());for(size_t i=0;i<groups.size();++i)result->rows[i]=groups[i].row;
 }
 return result;
}
}
extern "C" {
void*fast_arrays(const void*xyz,int dtype,uint64_t xyz_stride,const void*cost,int cost_dtype,uint64_t cost_stride,uint64_t n,const double*size,int partition){
 try{return partition?scan<true>(xyz,dtype,xyz_stride,cost,cost_dtype,cost_stride,n,size).release():scan<false>(xyz,dtype,xyz_stride,cost,cost_dtype,cost_stride,n,size).release();}
 catch(const std::exception&e){error=e.what();return nullptr;}
}
const char*fast_arrays_error(){return error.c_str();}
void fast_arrays_sizes(void*ptr,uint64_t*out){auto&r=*static_cast<Result*>(ptr);out[0]=r.groups.size();out[1]=r.rows.size();}
void fast_arrays_copy(void*ptr,int64_t*rows,int64_t*keys,uint64_t*offsets){auto&r=*static_cast<Result*>(ptr);
 std::copy(r.rows.begin(),r.rows.end(),rows);
 if(keys)for(size_t i=0;i<r.groups.size();++i){keys[3*i]=r.groups[i].key.x;keys[3*i+1]=r.groups[i].key.y;keys[3*i+2]=r.groups[i].key.z;}
 if(offsets)std::copy(r.offsets.begin(),r.offsets.end(),offsets);
}
void fast_arrays_free(void*ptr){delete static_cast<Result*>(ptr);}
}
