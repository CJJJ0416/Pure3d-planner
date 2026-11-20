// 仅基于 cost 的稀疏支撑面图（cost-only sparse surface graph）。
//
// 输入为重力对齐坐标系下的点云 XYZ 和逐点通行代价。构建流程：
//   1. 按 (xy_resolution, xy_resolution, z_resolution) 将点量化到体素，并分别
//      累计全部有效点与低代价点（cost < cost_threshold）；
//   2. 按 XY 列分组，在每列内沿 Z 轴把连续的低代价体素切成高度层片（patch）；
//   3. 在相邻 XY 列之间按高度窗口 slice_dh 寻找匹配层片，双向确认后连边；
//   4. 将无向边展开为双向 CSR 邻接表，供 A* 搜索和轨迹校验使用。
//
// XYZ 只用于分层和建立索引，不参与可通行性判断：
//   * 不计算坡度、台阶高度、离地间隙或机体碰撞；
//   * 高代价点只在建图时被过滤，不会被当作障碍再做几何绕行；
//   * 缺失观测的 XY 列不会生成层片，相关方向的边自然断开。
//
// 对外接口为 C ABI，由 planner/light_planner/surface.py 通过 ctypes 调用。
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// surface_build 的 parameters 数组下标。
constexpr int kParamXyResolution = 0;   // 水平体素边长（米）
constexpr int kParamZResolution = 1;    // 垂直体素边长（米）
constexpr int kParamCostThreshold = 2;  // 严格小于该值的点才参与建图
constexpr int kParamMaxPatchSpan = 3;   // 单个层片允许的最大高度跨度（米）
constexpr int kParamSliceDh = 4;        // 相邻层片匹配的高度窗口（米）

// 用 uint32_t 最大值表示"没有匹配的层片/节点"。
constexpr uint32_t kNoPatch = std::numeric_limits<uint32_t>::max();
// 量化后的坐标必须能安全放入 int64_t，这里留出 4 倍余量。
constexpr double kMaxGridCoordinate =
    static_cast<double>(std::numeric_limits<int64_t>::max()) / 4.;
// 比较高度窗口时允许的浮点误差。
constexpr double kHeightEpsilon = 1e-6;

double elapsed_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// 三维整数体素索引（非米制坐标）。x/y 决定所属列，z 决定垂直层。
struct VoxelKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

// splitmix64：对相邻的整数坐标也能给出分布均匀的哈希值。
uint64_t mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    // 三个坐标分别混合后异或，避免简单相加带来的碰撞模式。
    return static_cast<size_t>(
        mix64(static_cast<uint64_t>(key.x)) ^
        mix64(static_cast<uint64_t>(key.y) + 0x9e3779b97f4a7c15ULL) ^
        mix64(static_cast<uint64_t>(key.z) + 0x243f6a8885a308d3ULL));
  }
};

// 按 x、y、z 的字典序比较，用于把体素排成稳定的列/层顺序。
bool less_key(const VoxelKey& a, const VoxelKey& b) {
  if (a.x != b.x) return a.x < b.x;
  if (a.y != b.y) return a.y < b.y;
  return a.z < b.z;
}

// 一组点的增量统计量。均值由 sum_* / count 得到；
// min_z/max_z 用于判断层片的高度跨度，不参与坐标系变换。
struct Accumulator {
  double sum_x = 0.;
  double sum_y = 0.;
  double sum_z = 0.;
  double sum_cost = 0.;
  double min_z = std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  uint64_t count = 0;

  void add_point(double x, double y, double z, double cost) {
    sum_x += x;
    sum_y += y;
    sum_z += z;
    sum_cost += cost;
    ++count;
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }

  void merge(const Accumulator& other) {
    sum_x += other.sum_x;
    sum_y += other.sum_y;
    sum_z += other.sum_z;
    sum_cost += other.sum_cost;
    count += other.count;
    min_z = std::min(min_z, other.min_z);
    max_z = std::max(max_z, other.max_z);
  }
};

// 一个 (x, y, z) 体素。
struct Voxel {
  VoxelKey key;
  Accumulator all;       // 全部有效点，供 GPMP 查询完整 cost 场
  Accumulator low_cost;  // 仅 cost < threshold 的点，用于建图
};

// 一个高度层片：同一 XY 列内、Z 方向连续且高度跨度受限的一组低代价体素。
// 每个层片对应图中的一个节点。
struct Patch {
  VoxelKey key;       // x/y 为列索引，z 为该层片最低体素的 Z 索引
  Accumulator sum;    // 层片内低代价点的统计量
  double x = 0.;      // 低代价点坐标均值
  double y = 0.;
  double z = 0.;
  double cost = 0.;   // 低代价点 cost 均值
  uint32_t node = 0;  // 图节点编号，等于本层片在 patches 中的下标
};

// 一个 XY 列上由低到高排列的所有层片编号。
struct Column {
  std::vector<uint32_t> patches;
};

// 一次构建的全部状态。句柄指向该结构，构建完成后只读。
struct Data {
  // ---- 构建参数（来自 surface_build 的 parameters）----
  double xy_resolution = 0.;
  double z_resolution = 0.;
  double cost_threshold = 0.;
  double max_patch_span = 0.;
  double slice_dh = 0.;

  // ---- 构建结果 ----
  std::vector<Voxel> voxels;   // 按 key 升序排列
  std::vector<Patch> patches;  // 节点编号 = 下标
  std::unordered_map<VoxelKey, Column, VoxelKeyHash> columns;

  std::vector<std::pair<uint32_t, uint32_t>> edges;  // 无向边，只存一次
  std::vector<uint64_t> offsets;                     // CSR 行偏移，长度 n+1
  std::vector<uint32_t> neighbors;                   // 双向邻接表，长度 2|E|
  std::unordered_map<uint64_t, bool> step_edges;

  // 构建统计量，索引与 planner/light_planner/surface.py 的 report 字段对应：
  //   [0] 输入点数                [1] 无效点数
  //   [2] 保留的低代价点数        [3] 非空体素数
  //   [4] 层片数（图节点数）      [5] 有列但高度窗口内无匹配的次数
  //   [6..8] 保留                 [9] 双向匹配失败的次数
  //   [10] 对角方向缺少公共角点的次数  [11] 保留
  std::array<uint64_t, 12> stats{};

  // 构建耗时（秒）：[0] 聚合+排序  [1] 分列+分层片  [2] 建边+CSR
  // [3] 总耗时  [4] 保留
  std::array<double, 5> times{};

  // 查找 XY 列；列键的 z 固定为 0，不参与查找。不存在时返回 nullptr。
  const Column* find_column(int64_t x, int64_t y) const {
    const auto found = columns.find(VoxelKey{x, y, 0});
    return found == columns.end() ? nullptr : &found->second;
  }

  // 在列中挑选与 target_z 匹配的最佳层片；没有可用层片时返回 kNoPatch。
  // 选择顺序：cost 更低优先；cost 相同时高度差更小优先；仍相同则保留先
  // 遇到的层片（层片按 Z 升序构建，相当于取较低层）。
  // 候选必须满足：cost 有限且非负、cost < cost_threshold、高度差 <= slice_dh。
  uint32_t match_best_layer(const Column* column, double target_z) const {
    if (!column || !std::isfinite(target_z)) return kNoPatch;

    uint32_t best = kNoPatch;
    double best_cost = std::numeric_limits<double>::infinity();
    double best_difference = std::numeric_limits<double>::infinity();

    for (const uint32_t id : column->patches) {
      const Patch& patch = patches[id];
      if (!std::isfinite(patch.z) || !std::isfinite(patch.cost) ||
          patch.cost < 0. || patch.cost >= cost_threshold) {
        continue;
      }
      const double difference = std::abs(patch.z - target_z);
      if (difference > slice_dh + kHeightEpsilon) continue;

      if (patch.cost < best_cost ||
          (patch.cost == best_cost && difference < best_difference)) {
        best = id;
        best_cost = patch.cost;
        best_difference = difference;
      }
    }
    return best;
  }

  // 判断两个层片是否"互为最佳匹配"：a 所在列选中 b，b 所在列也选中 a。
  // 单侧匹配可能跨到另一条互不相连的层，因此不作为连边依据。
  bool mutually_paired(const Patch& a, const Patch& b) const {
    const Column* a_column = find_column(a.key.x, a.key.y);
    const Column* b_column = find_column(b.key.x, b.key.y);
    return match_best_layer(b_column, a.z) == b.node &&
           match_best_layer(a_column, b.z) == a.node;
  }

  // 对角边必须经过一个公共角点：在 (x, y) 列中存在层片 p，
  // 使得 a-p、p-b 都互为最佳匹配，从而排除斜穿缺失栅格的情况。
  bool has_shared_corner(const Patch& a, const Patch& b,
                         int64_t x, int64_t y) const {
    const Column* column = find_column(x, y);
    if (!column) return false;
    for (const uint32_t id : column->patches) {
      const Patch& corner = patches[id];
      if (mutually_paired(a, corner) && mutually_paired(corner, b)) {
        return true;
      }
    }
    return false;
  }

  // 检查两个节点在 CSR 邻接表中是否相邻（相同节点视为连通）。
  bool is_connected(uint32_t a, uint32_t b) const {
    if (a == b) return true;
    const auto begin =
        neighbors.begin() + static_cast<std::ptrdiff_t>(offsets[a]);
    const auto end =
        neighbors.begin() + static_cast<std::ptrdiff_t>(offsets[a + 1]);
    return std::binary_search(begin, end, b);
  }

  uint32_t match_step(uint32_t previous, double x, double y, double z) const {
    if (step_edges.empty()) return kNoPatch;
    uint32_t best=kNoPatch;
    double best_distance=0.035*0.035+1e-12;
    const auto inspect=[&](uint32_t a_id,uint32_t b_id) {
      const Patch& a=patches[a_id];
      const Patch& b=patches[b_id];
      const double dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z;
      const double t=((x-a.x)*dx+(y-a.y)*dy+(z-a.z)*dz)/(dx*dx+dy*dy+dz*dz);
      if (t<0. || t>1.) return;
      const double ex=x-a.x-t*dx,ey=y-a.y-t*dy,ez=z-a.z-t*dz;
      const double distance=ex*ex+ey*ey+ez*ez;
      const uint32_t endpoint=t<.5?a_id:b_id;
      if (distance<best_distance || (distance==best_distance && endpoint<best)) {
        best_distance=distance;
        best=endpoint;
      }
    };
    if (previous==kNoPatch) {
      // A rolling prefix can begin in the interior of a checked step edge.
      // Initialise from the closest actual edge, not an unrelated column node.
      for (const auto& edge:step_edges)
        inspect(static_cast<uint32_t>(edge.first>>32),static_cast<uint32_t>(edge.first));
    } else {
      for (uint64_t index=offsets[previous];index<offsets[previous+1];++index) {
        const uint32_t id=neighbors[index];
        const uint64_t key=(static_cast<uint64_t>(std::min(previous,id))<<32)|std::max(previous,id);
        if (step_edges.count(key)) inspect(previous,id);
      }
    }
    return best;
  }

};

// 最近一次构建失败的原因，每个线程独立保存。
thread_local std::string error;

// 将米制坐标量化为整数体素索引；超出安全范围时抛出异常，
// 由 surface_build 转成 Python 侧的 ValueError。
VoxelKey quantize(const Data& data, double x, double y, double z) {
  const double ix = std::floor(x / data.xy_resolution);
  const double iy = std::floor(y / data.xy_resolution);
  const double iz = std::floor(z / data.z_resolution);
  if (std::abs(ix) > kMaxGridCoordinate ||
      std::abs(iy) > kMaxGridCoordinate ||
      std::abs(iz) > kMaxGridCoordinate) {
    throw std::runtime_error("Voxel coordinate exceeds int64 range");
  }
  return VoxelKey{static_cast<int64_t>(ix), static_cast<int64_t>(iy),
                  static_cast<int64_t>(iz)};
}

// 第一遍：过滤无效点、量化到体素，并累计全部点与低代价点。
void aggregate_points(Data& data, const float* xyz, const float* costs,
                      uint64_t count) {
  std::unordered_map<VoxelKey, uint32_t, VoxelKeyHash> index;
  index.reserve(std::min<uint64_t>(count, 1000000));
  data.voxels.reserve(std::min<uint64_t>(count, 1000000));
  data.stats[0] = count;

  for (uint64_t i = 0; i < count; ++i) {
    const double x = xyz[3 * i];
    const double y = xyz[3 * i + 1];
    const double z = xyz[3 * i + 2];
    const double cost = costs[i];

    // 非有限坐标/代价、负代价都无法参与建图，只计数后跳过。
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(cost) || cost < 0.) {
      ++data.stats[1];
      continue;
    }

    const VoxelKey key = quantize(data, x, y, z);
    const auto inserted =
        index.emplace(key, static_cast<uint32_t>(data.voxels.size()));
    if (inserted.second) data.voxels.push_back(Voxel{key, {}, {}});

    Voxel& voxel = data.voxels[inserted.first->second];
    voxel.all.add_point(x, y, z, cost);
    if (cost < data.cost_threshold) {
      voxel.low_cost.add_point(x, y, z, cost);
      ++data.stats[2];
    }
  }

  // 后续改为按 key 排序处理，哈希表可以立即释放。
  index.clear();
  index.rehash(0);
}

// 第二遍：按 XY 列分组，并在列内沿 Z 轴切分高度层片。
// 只处理低代价体素；长竖直结构会被切成多个层片，但不会被删除。
void build_columns(Data& data) {
  data.columns.reserve(data.voxels.size());

  size_t first = 0;
  while (first < data.voxels.size()) {
    const VoxelKey column_key = data.voxels[first].key;

    // 体素已按 (x, y, z) 排序，同一 XY 列是连续的一段。
    size_t end = first;
    while (end < data.voxels.size() &&
           data.voxels[end].key.x == column_key.x &&
           data.voxels[end].key.y == column_key.y) {
      ++end;
    }

    Accumulator group;
    int64_t previous_z = std::numeric_limits<int64_t>::min();
    VoxelKey bottom_key{};

    // 结束当前层片，写入均值、最低体素索引和所属列。
    const auto flush = [&]() {
      if (!group.count) return;
      ++data.stats[4];
      const uint32_t id = static_cast<uint32_t>(data.patches.size());

      Patch patch;
      patch.key = bottom_key;
      patch.sum = group;
      patch.x = group.sum_x / static_cast<double>(group.count);
      patch.y = group.sum_y / static_cast<double>(group.count);
      patch.z = group.sum_z / static_cast<double>(group.count);
      patch.cost = group.sum_cost / static_cast<double>(group.count);
      patch.node = id;
      data.patches.push_back(patch);
      Column& column = data.columns[VoxelKey{column_key.x, column_key.y, 0}];
      column.patches.push_back(id);
    };

    for (size_t i = first; i < end; ++i) {
      const Voxel& voxel = data.voxels[i];
      if (!voxel.low_cost.count) continue;

      // 两种情况下切开层片：
      //   1) Z 方向不连续（中间至少隔了一个空体素）；
      //   2) 低代价点的高度跨度超过 max_patch_span。
      // 切分只影响分层表示，不会拒绝竖直结构，也不会在断开的层之间补点。
      if (group.count) {
        const double merged_span =
            std::max(group.max_z, voxel.low_cost.max_z) -
            std::min(group.min_z, voxel.low_cost.min_z);
        if (voxel.key.z > previous_z + 1 ||
            merged_span > data.max_patch_span) {
          flush();
          group = Accumulator{};
        }
      }

      if (!group.count) bottom_key = voxel.key;
      group.merge(voxel.low_cost);
      previous_z = voxel.key.z;
    }
    flush();

    first = end;
  }
}

// 第三遍：在四个"前向"方向上连接相邻层片。
// 每个无序对只会被检查一次，因此 edges 中不会出现同一条无向边的重复项。
void build_edges(Data& data) {
  constexpr int kDirections[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

  for (const Patch& a : data.patches) {
    for (const auto& direction : kDirections) {
      const Column* target =
          data.find_column(a.key.x + direction[0], a.key.y + direction[1]);
      const uint32_t b_id = data.match_best_layer(target, a.z);

      if (b_id == kNoPatch) {
        // 目标列存在，但没有落在高度窗口内的低代价层片：此处断开。
        if (target) ++data.stats[5];
        continue;
      }

      const Patch& b = data.patches[b_id];
      if (!data.mutually_paired(a, b)) {
        ++data.stats[9];
        continue;
      }

      // 对角方向额外要求两个正交相邻列都能提供公共角点，防止斜穿缺失栅格。
      if (direction[0] != 0 && direction[1] != 0 &&
          (!data.has_shared_corner(a, b, b.key.x, a.key.y) ||
           !data.has_shared_corner(a, b, a.key.x, b.key.y))) {
        ++data.stats[10];
        continue;
      }

      data.edges.emplace_back(a.node, b.node);
    }
  }
}

// 第四遍：把无向边展开为双向 CSR 邻接表。
void build_csr(Data& data) {
  const size_t node_count = data.patches.size();
  data.offsets.assign(node_count + 1, 0);

  // 先统计每个节点的度（每条无向边同时贡献两个端点）。
  for (const auto& edge : data.edges) {
    ++data.offsets[edge.first + 1];
    ++data.offsets[edge.second + 1];
  }
  for (size_t i = 1; i < data.offsets.size(); ++i) {
    data.offsets[i] += data.offsets[i - 1];
  }

  // 再按行偏移写入邻居；cursor 记录每个节点下一个空位。
  std::vector<uint64_t> cursor = data.offsets;
  data.neighbors.resize(data.edges.size() * 2);
  for (const auto& edge : data.edges) {
    data.neighbors[cursor[edge.first]++] = edge.second;
    data.neighbors[cursor[edge.second]++] = edge.first;
  }

  // surface_validate 用二分查找判断连通性，邻接段必须有序。
  for (size_t i = 0; i < node_count; ++i) {
    const auto begin =
        data.neighbors.begin() + static_cast<std::ptrdiff_t>(data.offsets[i]);
    const auto end =
        data.neighbors.begin() + static_cast<std::ptrdiff_t>(data.offsets[i + 1]);
    std::sort(begin, end);
  }
}

// 完整构建流程，并记录三个阶段的耗时。
void build(Data& data, const float* xyz, const float* costs, uint64_t count) {
  const Clock::time_point started = Clock::now();
  Clock::time_point tick = started;

  aggregate_points(data, xyz, costs, count);
  std::sort(data.voxels.begin(), data.voxels.end(),
            [](const Voxel& a, const Voxel& b) {
              return less_key(a.key, b.key);
            });
  data.stats[3] = data.voxels.size();
  data.times[0] = elapsed_since(tick);
  tick = Clock::now();

  build_columns(data);
  data.times[1] = elapsed_since(tick);
  tick = Clock::now();

  build_edges(data);
  build_csr(data);
  data.times[2] = elapsed_since(tick);

  data.times[3] = elapsed_since(started);
}

}  // namespace

extern "C" {

// 最近一次 surface_build 失败的原因；调用方在构建失败后读取。
const char* surface_error() { return error.c_str(); }

// 由点云和 cost 构建稀疏支撑面图。
// parameters 依次为：xy_resolution、z_resolution、cost_threshold、
// max_patch_span、slice_dh。成功返回句柄，失败返回 nullptr。
void* surface_build(const float* xyz, const float* cost, uint64_t count,
                    const double* parameters) {
  try {
    auto data = std::make_unique<Data>();
    data->xy_resolution = parameters[kParamXyResolution];
    data->z_resolution = parameters[kParamZResolution];
    data->cost_threshold = parameters[kParamCostThreshold];
    data->max_patch_span = parameters[kParamMaxPatchSpan];
    data->slice_dh = parameters[kParamSliceDh];
    build(*data, xyz, cost, count);
    return data.release();
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
}

void surface_free(void* handle) { delete static_cast<Data*>(handle); }

// 输出各数组长度与构建耗时：
//   counts[0..11] = stats，counts[12] = 节点数，
//   counts[13] = CSR 邻接表长度（2×边数），counts[14] = 层片数。
//   times[0..4] = 各阶段耗时。
void surface_sizes(void* handle, uint64_t* counts, double* times) {
  const Data& data = *static_cast<const Data*>(handle);
  std::copy(data.stats.begin(), data.stats.end(), counts);
  counts[12] = data.patches.size();
  counts[13] = data.neighbors.size();
  counts[14] = data.patches.size();
  std::copy(data.times.begin(), data.times.end(), times);
}

// 每个非空体素输出 7 个 double：
// x、y、z、cost 的均值，点数，以及 z 的最小/最大值。
// 这里包含全部有效点（不按 cost 过滤），供 GPMP 查询完整 cost 场。
void surface_copy_full(void* handle, double* out) {
  const Data& data = *static_cast<const Data*>(handle);
  for (size_t i = 0; i < data.voxels.size(); ++i) {
    const Accumulator& sum = data.voxels[i].all;
    const double values[7] = {
        sum.sum_x / static_cast<double>(sum.count),
        sum.sum_y / static_cast<double>(sum.count),
        sum.sum_z / static_cast<double>(sum.count),
        sum.sum_cost / static_cast<double>(sum.count),
        static_cast<double>(sum.count), sum.min_z, sum.max_z};
    std::copy(values, values + 7, out + 7 * i);
  }
}

// 每个层片输出 9 个 double：
// x、y、z、cost 均值，两个保留字段 0，z 的最小/最大值，点数。
// 保留字段用于维持既有数组布局，调用方只读取 [0:4] 与 [8]。
void surface_copy_patches(void* handle, double* out) {
  const Data& data = *static_cast<const Data*>(handle);
  for (size_t i = 0; i < data.patches.size(); ++i) {
    const Patch& patch = data.patches[i];
    const double values[9] = {patch.x, patch.y, patch.z, patch.cost, 0., 0.,
                              patch.sum.min_z, patch.sum.max_z,
                              static_cast<double>(patch.sum.count)};
    std::copy(values, values + 9, out + 9 * i);
  }
}

// 输出 A* 使用的图：节点坐标、节点 cost、CSR offsets/neighbors，
// 以及每个节点的诊断信息：体素键 x/y/z（x/y 为列索引，z 为最低体素层）、
// 层片 min_z/max_z、保留位。
void surface_copy_graph(void* handle, double* xyz, double* cost,
                        uint64_t* offsets, uint32_t* neighbors,
                        double* detail) {
  const Data& data = *static_cast<const Data*>(handle);
  for (size_t i = 0; i < data.patches.size(); ++i) {
    const Patch& patch = data.patches[i];
    xyz[3 * i] = patch.x;
    xyz[3 * i + 1] = patch.y;
    xyz[3 * i + 2] = patch.z;
    cost[i] = patch.cost;

    const double values[6] = {static_cast<double>(patch.key.x),
                              static_cast<double>(patch.key.y),
                              static_cast<double>(patch.key.z),
                              patch.sum.min_z, patch.sum.max_z, 0.};
    std::copy(values, values + 6, detail + 6 * i);
  }
  std::copy(data.offsets.begin(), data.offsets.end(), offsets);
  std::copy(data.neighbors.begin(), data.neighbors.end(), neighbors);
}

// 查询 (x, y) 所在列中与 target_height 匹配的最佳层片。
// Install the checked sparse graph and retain explicit cross-step segments.
void surface_set_graph(void* handle, const uint64_t* offsets, const uint32_t* neighbors,
                       uint64_t edge_count, double minimum_step_height) {
  Data& data=*static_cast<Data*>(handle);
  data.offsets.assign(offsets,offsets+data.patches.size()+1);
  data.neighbors.assign(neighbors,neighbors+edge_count);
  data.step_edges.clear();
  for (uint32_t a=0;a<data.patches.size();++a) {
    for (uint64_t index=data.offsets[a];index<data.offsets[a+1];++index) {
      const uint32_t b=data.neighbors[index];
      if (a<b && std::abs(data.patches[a].z-data.patches[b].z)>minimum_step_height) {
        data.step_edges[(static_cast<uint64_t>(a)<<32)|b]=true;
      }
    }
  }
}

// 查询 (x, y) 所在列中与 target_height 匹配的最佳层片。
// 返回节点编号；坐标非法、列缺失或没有匹配层片时返回 -1。
// 这里只检查已观测 cost 单元和高度窗口，不推断坡度/台阶/净空。
int64_t surface_match_layer(void* handle, double x, double y,
                            double target_height) {
  const Data& data = *static_cast<const Data*>(handle);
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(target_height) ||
      std::abs(x / data.xy_resolution) > kMaxGridCoordinate ||
      std::abs(y / data.xy_resolution) > kMaxGridCoordinate) {
    return -1;
  }

  const auto cell_x = static_cast<int64_t>(std::floor(x / data.xy_resolution));
  const auto cell_y = static_cast<int64_t>(std::floor(y / data.xy_resolution));
  const uint32_t id =
      data.match_best_layer(data.find_column(cell_x, cell_y), target_height);
  return id == kNoPatch ? -1 : static_cast<int64_t>(id);
}

// 沿轨迹采样检查支撑面几何：
//   out[0] = 第一个失败样本的下标；全部通过时等于样本数。
//   out[1] = 0 通过 / 1 缺失 cost 列 / 2 高度不匹配 /
//            3 相邻样本不连通 / 5 非有限坐标。
// 这里只检查观测 cost 单元是否存在以及图中是否连通，不做坡度/台阶/
// 净空/碰撞推断；第 4 个参数为兼容旧调用保留，当前未使用。
int surface_validate(void* handle, const double* xyz, uint64_t count,
                     double /*unused*/, uint64_t* out) {
  const Data& data = *static_cast<const Data*>(handle);
  uint32_t previous = kNoPatch;

  for (uint64_t i = 0; i < count; ++i) {
    const double x = xyz[3 * i];
    const double y = xyz[3 * i + 1];
    const double z = xyz[3 * i + 2];

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      out[0] = i;
      out[1] = 5;
      return 0;
    }

    const auto cell_x =
        static_cast<int64_t>(std::floor(x / data.xy_resolution));
    const auto cell_y =
        static_cast<int64_t>(std::floor(y / data.xy_resolution));
    const Column* column = data.find_column(cell_x, cell_y);
    uint32_t best = data.match_best_layer(column, z);
    if (!data.step_edges.empty() && column) {
      // A route sample follows its actual support height, including when a
      // lower-cost but different floor also lies inside the endpoint window.
      double difference=std::numeric_limits<double>::infinity();
      for (const uint32_t id:column->patches) {
        const double height=std::abs(data.patches[id].z-z);
        if (height<=data.slice_dh+kHeightEpsilon && height<difference) {
          best=id;difference=height;
        }
      }
    }

    if (i==0 || previous!=kNoPatch || best==kNoPatch) {
      const uint32_t step=data.match_step(previous,x,y,z);
      if (step!=kNoPatch) { previous=step; continue; }
    }

    if (best == kNoPatch) {
      out[0] = i;
      out[1] = column ? 2 : 1;
      return 0;
    }
    if (previous != kNoPatch && !data.is_connected(previous, best)) {
      out[0] = i;
      out[1] = 3;
      return 0;
    }
    previous = best;
  }

  out[0] = count;
  out[1] = 0;
  return 1;
}

}  // extern "C"
