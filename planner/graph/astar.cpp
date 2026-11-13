// DC CSR图上的单线程单向A*。边权=L+alpha*cost积分，h=三维欧氏距离。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

static int dc_astar_impl(uint32_t n, const uint64_t* offsets,
    const uint32_t* neighbors, const double* xyz, const double* cost,
    uint32_t start, uint32_t goal, double alpha,
    uint32_t* path, uint32_t* path_size, double* stats,
    uint32_t* expansion_order, uint32_t* saved_parent, double* saved_g,
    double* expansion_priority, uint32_t* decisions) {
  const auto began=std::chrono::steady_clock::now();
  const double inf=std::numeric_limits<double>::infinity();
  const uint32_t none=std::numeric_limits<uint32_t>::max();
  auto distance=[&](uint32_t a,uint32_t b) {
    const double x=xyz[3*a]-xyz[3*b], y=xyz[3*a+1]-xyz[3*b+1], z=xyz[3*a+2]-xyz[3*b+2];
    return std::sqrt(x*x+y*y+z*z);
  };
  std::vector<double> h(n),g(n,inf);
  std::vector<uint32_t> parent(n,none);
  std::vector<uint8_t> closed(n,0);
  for(uint32_t i=0;i<n;++i) h[i]=distance(i,goal);
  using Entry=std::pair<double,uint32_t>;
  std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> open;
  g[start]=0.;open.push({h[start],start});
  uint64_t expanded=0,examined=0;
  bool found=false;
  while(true) {
    while(!open.empty() && closed[open.top().second]) open.pop();
    if(open.empty()) break;
    const double popped_priority=open.top().first;
    const uint32_t u=open.top().second;open.pop();closed[u]=1;
    if(expansion_order) {
      expansion_order[expanded]=u;
      expansion_priority[expanded]=popped_priority;
      decisions[3*expanded]=decisions[3*expanded+1]=decisions[3*expanded+2]=0;
    }
    ++expanded;
    if(u==goal) { found=true;break; }
    for(uint64_t e=offsets[u];e<offsets[u+1];++e) {
      const uint32_t v=neighbors[e];
      if(closed[v]) {
        if(decisions) ++decisions[3*(expanded-1)];
        continue;
      }
      ++examined;
      const double edge=distance(u,v)*(1.+alpha*(cost[u]+cost[v])/2.);
      const double proposed=g[u]+edge;
      if(proposed<g[v]) {
        g[v]=proposed;parent[v]=u;open.push({proposed+h[v],v});
        if(decisions) ++decisions[3*(expanded-1)+1];
      } else if(decisions) ++decisions[3*(expanded-1)+2];
    }
  }
  stats[0]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
  stats[1]=g[goal];stats[2]=expanded;stats[3]=examined;stats[4]=0.;
  *path_size=0;
  if(found) {
    std::vector<uint32_t> chain;
    for(uint32_t v=goal;v!=none;v=parent[v]) chain.push_back(v);
    std::reverse(chain.begin(),chain.end());
    std::copy(chain.begin(),chain.end(),path);*path_size=chain.size();
    for(size_t i=1;i<chain.size();++i) stats[4]+=distance(chain[i-1],chain[i]);
  }
  stats[5]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
  if(saved_parent) std::copy(parent.begin(),parent.end(),saved_parent);
  if(saved_g) std::copy(g.begin(),g.end(),saved_g);
  return found ? 1 : 0;
}

extern "C" int dc_astar(uint32_t n, const uint64_t* offsets,
    const uint32_t* neighbors, const double* xyz, const double* cost,
    uint32_t start, uint32_t goal, double alpha,
    uint32_t* path, uint32_t* path_size, double* stats) {
  return dc_astar_impl(n,offsets,neighbors,xyz,cost,start,goal,alpha,path,path_size,
      stats,nullptr,nullptr,nullptr,nullptr,nullptr);
}

// 可选追踪与正式入口调用同一个循环，不改变边权、优先队列或终止条件。
extern "C" int dc_astar_trace(uint32_t n, const uint64_t* offsets,
    const uint32_t* neighbors, const double* xyz, const double* cost,
    uint32_t start, uint32_t goal, double alpha,
    uint32_t* path, uint32_t* path_size, double* stats,
    uint32_t* expansion_order, uint32_t* parent, double* g,
    double* expansion_priority, uint32_t* decisions) {
  return dc_astar_impl(n,offsets,neighbors,xyz,cost,start,goal,alpha,path,path_size,
      stats,expansion_order,parent,g,expansion_priority,decisions);
}
