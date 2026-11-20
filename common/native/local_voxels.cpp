// 有界、世界坐标对齐的局部 LIO 体素代表点集合（无 ROS/CUDA 依赖）。
//
// 每个体素只保存第一次观测到的真实 XYZ：不做质心、占据膨胀、地形过滤
// 或自由空间填充。主边界是半开区间 [lower, upper)，越界点直接丢弃；
// 查询 ROI 独立于主边界，用于统计当前上下文窗口内的代表点与观测数。
//
// 对外接口为 C ABI，由 common/mapping/local_map.py 通过 ctypes 调用。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using VoxelKey = std::array<int64_t, 3>;
using Position = std::array<float, 3>;

// 三维体素键的哈希（FNV-1a 变体，逐坐标混入固定偏移）。
struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (const int64_t axis : key) {
      hash ^= static_cast<uint64_t>(axis) + 0x9e3779b97f4a7c15ULL +
              (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

// 一次局部地图会话的全部状态；句柄指向该结构。
struct Map {
  // ---- 配置 ----
  double voxel_size = 0.;
  uint64_t capacity = 0;

  // ---- 主边界与累计统计 ----
  bool bounded = false;
  std::array<double, 3> lower{};
  std::array<double, 3> upper{};
  uint64_t revision = 0;
  uint64_t capacity_rejections = 0;
  uint64_t invalid_points = 0;
  uint64_t observations = 0;
  std::unordered_map<VoxelKey, Position, VoxelKeyHash> points;

  // ---- 独立查询 ROI ----
  bool query_bounded = false;
  std::array<double, 3> query_lower{};
  std::array<double, 3> query_upper{};
  uint64_t query_points = 0;
  uint64_t query_observations = 0;

  // 半开区间 [lower, upper)。
  static bool in_box(const Position& point,
                     const std::array<double, 3>& box_lower,
                     const std::array<double, 3>& box_upper) {
    for (int axis = 0; axis < 3; ++axis) {
      if (point[axis] < box_lower[axis] || point[axis] >= box_upper[axis]) {
        return false;
      }
    }
    return true;
  }

  bool inside(const Position& point) const {
    return in_box(point, lower, upper);
  }

  bool in_query(const Position& point) const {
    return query_bounded && in_box(point, query_lower, query_upper);
  }
};

// 最近一次失败的原因，每个线程独立保存。
thread_local std::string error;

}  // namespace

extern "C" {

const char* local_voxels_error() { return error.c_str(); }

// 创建地图会话；体素尺寸或容量非法时返回 nullptr。
void* local_voxels_new(double voxel_size, uint64_t capacity) {
  try {
    if (!std::isfinite(voxel_size) || voxel_size <= 0. || capacity == 0) {
      throw std::runtime_error("Invalid voxel size/capacity");
    }
    auto* map = new Map;
    map->voxel_size = voxel_size;
    map->capacity = capacity;
    return map;
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
}

void local_voxels_free(void* handle) {
  delete static_cast<Map*>(handle);
}

// 设置主边界；返回 1 表示有变化，0 表示边界未变，-1 表示参数非法。
int local_voxels_bounds(void* handle, const double* lower,
                        const double* upper) {
  try {
    Map& map = *static_cast<Map*>(handle);
    bool changed = !map.bounded;
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) ||
          lower[axis] >= upper[axis]) {
        throw std::runtime_error("Invalid local box");
      }
      changed |= lower[axis] != map.lower[axis] ||
                 upper[axis] != map.upper[axis];
    }
    if (!changed) return 0;

    std::copy(lower, lower + 3, map.lower.begin());
    std::copy(upper, upper + 3, map.upper.begin());
    map.bounded = true;

    // 主边界变化后丢弃越界点；若该点同时在查询 ROI 内，同步维护查询计数。
    for (auto it = map.points.begin(); it != map.points.end();) {
      if (!map.inside(it->second)) {
        if (map.in_query(it->second)) --map.query_points;
        it = map.points.erase(it);
      } else {
        ++it;
      }
    }
    ++map.revision;
    return 1;
  } catch (const std::exception& exception) {
    error = exception.what();
    return -1;
  }
}

// 追加一帧点云；返回 1 表示地图有变化，0 表示无变化，-1 表示失败。
// 同一体素只保留第一次出现的真实坐标；容量满后新体素计入 capacity_rejections。
int local_voxels_add(void* handle, const float* xyz, uint64_t count) {
  try {
    Map& map = *static_cast<Map*>(handle);
    if (!map.bounded) throw std::runtime_error("Local box is not available");

    constexpr double kMaxGridCoordinate =
        static_cast<double>(std::numeric_limits<int64_t>::max()) / 2.;

    bool changed = false;
    for (uint64_t row = 0; row < count; ++row) {
      const Position point{xyz[3 * row], xyz[3 * row + 1], xyz[3 * row + 2]};
      if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
          !std::isfinite(point[2])) {
        ++map.invalid_points;
        continue;
      }
      if (!map.inside(point)) continue;

      // 进入主边界的有限点都算一次观测，包括重复体素与容量拒绝。
      ++map.observations;
      if (map.in_query(point)) ++map.query_observations;

      VoxelKey key{};
      for (int axis = 0; axis < 3; ++axis) {
        const double cell =
            std::floor(static_cast<double>(point[axis]) / map.voxel_size);
        if (std::abs(cell) >= kMaxGridCoordinate) {
          throw std::runtime_error("Voxel coordinate overflow");
        }
        key[axis] = static_cast<int64_t>(cell);
      }

      if (map.points.find(key) != map.points.end()) continue;
      if (map.points.size() >= map.capacity) {
        ++map.capacity_rejections;
        continue;
      }

      map.points.emplace(key, point);
      changed = true;
      if (map.in_query(point)) ++map.query_points;
    }

    if (changed) ++map.revision;
    return changed ? 1 : 0;
  } catch (const std::exception& exception) {
    error = exception.what();
    return -1;
  }
}

// out = [points, revision, capacity_rejections, invalid_points, observations]
void local_voxels_stats(void* handle, uint64_t* out) {
  const Map& map = *static_cast<const Map*>(handle);
  out[0] = map.points.size();
  out[1] = map.revision;
  out[2] = map.capacity_rejections;
  out[3] = map.invalid_points;
  out[4] = map.observations;
}

// 设置独立的查询 ROI（半开区间），返回 1 成功、-1 参数非法。
// query_observations 是累计值，不随查询框变化而清零。
int local_voxels_query_bounds(void* handle, const double* lower,
                              const double* upper) {
  try {
    Map& map = *static_cast<Map*>(handle);
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) ||
          lower[axis] >= upper[axis]) {
        throw std::runtime_error("Invalid query bounds");
      }
    }

    std::copy(lower, lower + 3, map.query_lower.begin());
    std::copy(upper, upper + 3, map.query_upper.begin());
    map.query_bounded = true;

    // 重新统计当前查询框内的已存代表点。
    map.query_points = 0;
    for (const auto& entry : map.points) {
      if (map.in_query(entry.second)) ++map.query_points;
    }
    return 1;
  } catch (const std::exception& exception) {
    error = exception.what();
    return -1;
  }
}

// out = [query_points, query_observations]
void local_voxels_query_stats(void* handle, uint64_t* out) {
  const Map& map = *static_cast<const Map*>(handle);
  out[0] = map.query_points;
  out[1] = map.query_observations;
}

// 按体素键升序把所有代表点写入 xyz（行优先 float32）。
int local_voxels_copy(void* handle, float* xyz) {
  try {
    const Map& map = *static_cast<const Map*>(handle);
    std::vector<std::pair<VoxelKey, Position>> rows(map.points.begin(),
                                                    map.points.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (size_t row = 0; row < rows.size(); ++row) {
      for (int axis = 0; axis < 3; ++axis) {
        xyz[3 * row + axis] = rows[row].second[axis];
      }
    }
    return 0;
  } catch (const std::exception& exception) {
    error = exception.what();
    return -1;
  }
}

}  // extern "C"
