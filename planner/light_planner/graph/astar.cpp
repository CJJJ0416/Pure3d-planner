// CSR 图上的单线程单向 A*（正式 DC 规划的唯一搜索实现）。
//
// 节点：planner/light_planner/surface.py 生成的高度层片；边：相邻层片之间的双向连接。
// 边权：w(u, v) = ||p_u - p_v||_2 × (1 + cost_weight × (c_u + c_v) / 2)
// 启发：h(u)   = ||p_u - p_goal||_2
// 目标：g(path) = 路径长度 + cost_weight × 沿线 cost 积分
//
// cost 非负且 cost_weight 非负时，h 是剩余代价的一致下界；h 不含预测 cost，
// 路线偏好完全由 g 的边权决定。
//
// 两个导出入口共用同一搜索循环：
//   dc_astar       正式搜索，只输出路径与统计量；
//   dc_astar_trace 额外输出展开顺序、父节点、g 值和决策计数，供离线分析；
//                  不改变边权、堆序或终止条件。
//
// 调用方：planner/light_planner/graph/astar.py（ctypes）。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// 父节点数组中的"无父节点"标记。
constexpr uint32_t kNoParent = std::numeric_limits<uint32_t>::max();
// 尚未到达的节点，其 g 值保持为正无穷。
constexpr double kInfinity = std::numeric_limits<double>::infinity();

double elapsed_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// 三维欧氏距离。边权和启发项使用同一度量。
double point_distance(const double* xyz, uint32_t a, uint32_t b) {
  const double dx = xyz[3 * a] - xyz[3 * b];
  const double dy = xyz[3 * a + 1] - xyz[3 * b + 1];
  const double dz = xyz[3 * a + 2] - xyz[3 * b + 2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 可选的追踪输出。正式入口全部为空指针，不产生额外开销。
// expansion_order/expansion_priority 只写入前 stats[2] 项。
// decisions 是 (node_count, 3) 展平数组，每行对应一次节点展开：
//   [0] 邻居已关闭而跳过    [1] g 值被改进    [2] 检查但未改进
// parent/g 保存全部节点的最终值（长度 node_count）。
struct TraceOutputs {
  uint32_t* expansion_order = nullptr;
  uint32_t* parent = nullptr;
  double* g = nullptr;
  double* expansion_priority = nullptr;
  uint32_t* decisions = nullptr;
};

// 搜索主体。两个导出入口只在 trace 是否为空上有区别。
int run_astar(uint32_t node_count, const uint64_t* offsets,
              const uint32_t* neighbors, const double* xyz,
              const double* cost, uint32_t start, uint32_t goal,
              double cost_weight, uint32_t* path, uint32_t* path_size,
              double* stats, const TraceOutputs& trace) {
  const Clock::time_point began = Clock::now();

  // h：到目标的欧氏距离（可采纳启发）。
  // g：从起点到各节点的当前最优代价。
  // parent：路径回溯用的前驱。
  // closed：已展开集合；优先队列采用惰性删除。
  std::vector<double> h(node_count);
  std::vector<double> g(node_count, kInfinity);
  std::vector<uint32_t> parent(node_count, kNoParent);
  std::vector<uint8_t> closed(node_count, 0);
  for (uint32_t node = 0; node < node_count; ++node) {
    h[node] = point_distance(xyz, node, goal);
  }

  // 小顶堆按 (f, node) 排序；f 相同时节点编号小者优先。
  using QueueEntry = std::pair<double, uint32_t>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                      std::greater<QueueEntry>> open;
  g[start] = 0.;
  open.push({h[start], start});

  uint64_t expanded = 0;  // 已展开节点数
  uint64_t examined = 0;  // 已检查的边数
  bool found = false;

  while (true) {
    // 惰性删除：堆中可能残留同一节点的旧条目。
    while (!open.empty() && closed[open.top().second]) open.pop();
    if (open.empty()) break;

    const double popped_priority = open.top().first;
    const uint32_t current = open.top().second;
    open.pop();
    closed[current] = 1;

    if (trace.expansion_order) {
      // expanded 在自增前就是本次展开在追踪数组中的下标。
      trace.expansion_order[expanded] = current;
      trace.expansion_priority[expanded] = popped_priority;
      trace.decisions[3 * expanded] = 0;
      trace.decisions[3 * expanded + 1] = 0;
      trace.decisions[3 * expanded + 2] = 0;
    }
    ++expanded;

    if (current == goal) {
      found = true;
      break;
    }

    for (uint64_t edge = offsets[current]; edge < offsets[current + 1];
         ++edge) {
      const uint32_t next = neighbors[edge];
      if (closed[next]) {
        if (trace.decisions) ++trace.decisions[3 * (expanded - 1)];
        continue;
      }

      ++examined;
      // 边权 = 三维距离 × (1 + cost_weight × 两端平均 cost)。
      const double edge_cost =
          point_distance(xyz, current, next) *
          (1. + cost_weight * (cost[current] + cost[next]) / 2.);
      const double proposed = g[current] + edge_cost;

      if (proposed < g[next]) {
        g[next] = proposed;
        parent[next] = current;
        open.push({proposed + h[next], next});
        if (trace.decisions) {
          ++trace.decisions[3 * (expanded - 1) + 1];
        }
      } else if (trace.decisions) {
        ++trace.decisions[3 * (expanded - 1) + 2];
      }
    }
  }

  // stats 布局与 planner/light_planner/graph/astar.py 对应：
  //   [0] 搜索阶段耗时（秒，从入口到循环结束）
  //   [1] 目标节点的 g 值（objective），未找到时为 inf
  //   [2] 展开节点数
  //   [3] 检查边数（仅统计未关闭的邻居）
  //   [4] 路径三维折线长度（米），未找到时为 0
  //   [5] 总耗时（秒，含路径回溯；不含 trace 拷贝）
  stats[0] = elapsed_since(began);
  stats[1] = g[goal];
  stats[2] = static_cast<double>(expanded);
  stats[3] = static_cast<double>(examined);
  stats[4] = 0.;

  // 从目标沿 parent 回溯到起点，再反转为正向路径。
  *path_size = 0;
  if (found) {
    std::vector<uint32_t> chain;
    for (uint32_t node = goal; node != kNoParent; node = parent[node]) {
      chain.push_back(node);
    }
    std::reverse(chain.begin(), chain.end());
    std::copy(chain.begin(), chain.end(), path);
    *path_size = static_cast<uint32_t>(chain.size());

    for (size_t i = 1; i < chain.size(); ++i) {
      stats[4] += point_distance(xyz, chain[i - 1], chain[i]);
    }
  }

  stats[5] = elapsed_since(began);

  // 追踪数组使用调用方提供的缓冲区；正式入口不会执行这些拷贝。
  if (trace.parent) std::copy(parent.begin(), parent.end(), trace.parent);
  if (trace.g) std::copy(g.begin(), g.end(), trace.g);

  return found ? 1 : 0;
}

}  // namespace

// 正式搜索入口：只返回路径和 stats。
extern "C" int dc_astar(uint32_t node_count, const uint64_t* offsets,
                        const uint32_t* neighbors, const double* xyz,
                        const double* cost, uint32_t start, uint32_t goal,
                        double cost_weight, uint32_t* path,
                        uint32_t* path_size, double* stats) {
  return run_astar(node_count, offsets, neighbors, xyz, cost, start, goal,
                   cost_weight, path, path_size, stats, TraceOutputs{});
}

// 追踪入口：与 dc_astar 共用搜索循环，额外输出：
//   expansion_order/expansion_priority：前 stats[2] 项展开顺序与弹出 f 值
//   parent/g：长度 node_count 的最终父节点与 g 值
//   decisions：形状 (node_count, 3) 的决策计数
extern "C" int dc_astar_trace(uint32_t node_count, const uint64_t* offsets,
                              const uint32_t* neighbors, const double* xyz,
                              const double* cost, uint32_t start,
                              uint32_t goal, double cost_weight,
                              uint32_t* path, uint32_t* path_size,
                              double* stats, uint32_t* expansion_order,
                              uint32_t* parent, double* g,
                              double* expansion_priority,
                              uint32_t* decisions) {
  const TraceOutputs trace{expansion_order, parent, g,
                           expansion_priority, decisions};
  return run_astar(node_count, offsets, neighbors, xyz, cost, start, goal,
                   cost_weight, path, path_size, stats, trace);
}
