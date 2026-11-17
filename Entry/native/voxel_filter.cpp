// World-anchored 27-cell support counts, without KNN or moving input points.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Key=std::array<int64_t,3>;
struct Hash {
  size_t operator()(const Key& k) const {
    uint64_t h=1469598103934665603ULL;
    for(auto v:k) { h^=uint64_t(v)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); }
    return size_t(h);
  }
};
struct Cell { uint64_t count=0,support=0; };
static thread_local std::string error;
extern "C" {
const char* entry_voxel_filter_error() { return error.c_str(); }
int entry_voxel_filter(const float* xyz,uint64_t n,double size,uint64_t minimum,
                       uint8_t* keep,uint64_t* support,uint64_t* cells,double* seconds) {
  try {
    if(!std::isfinite(size)||size<=0||!minimum) throw std::runtime_error("Invalid filter size/threshold");
    auto start=std::chrono::steady_clock::now();
    std::unordered_map<Key,Cell,Hash> map;map.reserve(n);
    std::vector<Key> keys(n);
    for(uint64_t i=0;i<n;++i) {
      for(int axis=0;axis<3;++axis) {
        double v=std::floor(double(xyz[3*i+axis])/size);
        if(!std::isfinite(v)||std::abs(v)>double(INT64_MAX)/2) throw std::runtime_error("Invalid/overflowing XYZ");
        keys[i][axis]=int64_t(v);
      }
      ++map[keys[i]].count;
    }
    auto built=std::chrono::steady_clock::now();
    for(auto& row:map) {
      for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz) {
        Key key{row.first[0]+dx,row.first[1]+dy,row.first[2]+dz};
        auto found=map.find(key);
        if(found!=map.end()) row.second.support+=found->second.count;
      }
    }
    for(uint64_t i=0;i<n;++i) {support[i]=map.at(keys[i]).support;keep[i]=support[i]>=minimum;}
    cells[0]=map.size();
    seconds[0]=std::chrono::duration<double>(built-start).count();
    seconds[1]=std::chrono::duration<double>(std::chrono::steady_clock::now()-built).count();
    return 0;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
}
