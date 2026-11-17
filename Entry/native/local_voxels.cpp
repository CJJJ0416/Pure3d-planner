// Bounded, world-aligned local LIO representatives. No ROS or CUDA dependency.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Key = std::array<int64_t, 3>;
using XYZ = std::array<float, 3>;
struct Hash {
  size_t operator()(const Key& k) const {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (auto x : k) h ^= uint64_t(x) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
    return h;
  }
};
struct Map {
  double voxel;
  uint64_t limit, revision=0, rejected=0, invalid=0, observations=0;
  bool bounded=false;
  std::array<double, 3> lower{}, upper{};
  std::unordered_map<Key, XYZ, Hash> points;
  bool query_bounded=false;
  std::array<double, 3> query_lower{}, query_upper{};
  uint64_t query_points=0, query_observations=0;
  bool in_query(const XYZ& p) const {
    if(!query_bounded) return false;
    for(int i=0;i<3;++i) if(p[i]<query_lower[i] || p[i]>=query_upper[i]) return false;
    return true;
  }
  bool inside(const XYZ& p) const {
    for (int i=0;i<3;++i) if (p[i]<lower[i] || p[i]>=upper[i]) return false;
    return true;
  }
};
static thread_local std::string error;
extern "C" {
const char* local_voxels_error() { return error.c_str(); }
void* local_voxels_new(double voxel, uint64_t limit) {
  try {
    if (!std::isfinite(voxel) || voxel<=0 || !limit) throw std::runtime_error("Invalid voxel size/capacity");
    auto* m=new Map; m->voxel=voxel;m->limit=limit;return m;
  } catch(const std::exception& e) { error=e.what();return nullptr; }
}
void local_voxels_free(void* handle) { delete static_cast<Map*>(handle); }
int local_voxels_bounds(void* handle, const double* lower, const double* upper) {
  try {
    auto& m=*static_cast<Map*>(handle);bool changed=!m.bounded;
    for(int i=0;i<3;++i) {
      if(!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>=upper[i]) throw std::runtime_error("Invalid local box");
      changed |= lower[i]!=m.lower[i] || upper[i]!=m.upper[i];
    }
    if(!changed) return 0;
    std::copy(lower,lower+3,m.lower.begin());std::copy(upper,upper+3,m.upper.begin());m.bounded=true;
    for(auto it=m.points.begin();it!=m.points.end();) {
      if(!m.inside(it->second)) {
        if(m.in_query(it->second)) --m.query_points;
        it=m.points.erase(it);
      } else ++it;
    }
    ++m.revision;return 1;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
int local_voxels_add(void* handle, const float* xyz, uint64_t count) {
  try {
    auto& m=*static_cast<Map*>(handle);
    if(!m.bounded) throw std::runtime_error("Local box is not available");
    bool changed=false;
    for(uint64_t row=0;row<count;++row) {
      XYZ p{xyz[row*3],xyz[row*3+1],xyz[row*3+2]};
      if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {++m.invalid;continue;}
      if(!m.inside(p)) continue;
      ++m.observations;Key key;
      if(m.in_query(p)) ++m.query_observations;
      for(int i=0;i<3;++i) {
        double cell=std::floor(double(p[i])/m.voxel);
        if(std::abs(cell)>=double(INT64_MAX)/2) throw std::runtime_error("Voxel coordinate overflow");
        key[i]=int64_t(cell);
      }
      if(m.points.find(key)!=m.points.end()) continue;
      if(m.points.size()>=m.limit) {++m.rejected;continue;}
      m.points.emplace(key,p);changed=true;
      if(m.in_query(p)) ++m.query_points;
    }
    if(changed) ++m.revision;
    return changed?1:0;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
void local_voxels_stats(void* handle, uint64_t* out) {
  auto& m=*static_cast<Map*>(handle);
  out[0]=m.points.size();out[1]=m.revision;out[2]=m.rejected;out[3]=m.invalid;out[4]=m.observations;
}
int local_voxels_query_bounds(void* handle, const double* lower, const double* upper) {
  try {
    auto& m=*static_cast<Map*>(handle);
    for(int i=0;i<3;++i)
      if(!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>=upper[i])
        throw std::runtime_error("Invalid query bounds");
    std::copy(lower,lower+3,m.query_lower.begin());
    std::copy(upper,upper+3,m.query_upper.begin());m.query_bounded=true;
    m.query_points=0;
    for(const auto& row:m.points) if(m.in_query(row.second)) ++m.query_points;
    return 1;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
void local_voxels_query_stats(void* handle, uint64_t* out) {
  auto& m=*static_cast<Map*>(handle);
  out[0]=m.query_points;out[1]=m.query_observations;
}
int local_voxels_copy(void* handle, float* xyz) {
  try {
    auto& m=*static_cast<Map*>(handle);
    std::vector<std::pair<Key,XYZ>> rows(m.points.begin(),m.points.end());
    std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    for(size_t row=0;row<rows.size();++row) for(int j=0;j<3;++j) xyz[row*3+j]=rows[row].second[j];
    return 0;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
}
