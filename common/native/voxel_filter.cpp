// 基于世界体素 3×3×3 邻域点数的保守滤波器（无 KNN，不移动输入点）。
//
// 每个点量化到世界对齐体素；每个体素的支持度定义为自身及周围 26 个
// 邻居（共 27 格，包含自身）内的点数之和。支持度达到 minimum 的点保留。
// 该过程只做计数，不生成几何、不插值、不修改原始坐标。
//
// 对外接口为 C ABI，由 common/mapping/voxel_filter.py 通过 ctypes 调用。
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using VoxelKey = std::array<int64_t, 3>;

// 三维体素键的哈希（FNV-1a 变体，逐坐标混入固定偏移）。
struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    uint64_t hash = 1469598103934665603ULL;
    for (const int64_t axis : key) {
      hash ^= static_cast<uint64_t>(axis) + 0x9e3779b97f4a7c15ULL +
              (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

// 一个体素的原始点数与 3×3×3 邻域支持度。
struct VoxelCell {
  uint64_t count = 0;
  uint64_t support = 0;
};

// 最近一次失败的原因，每个线程独立保存。
thread_local std::string error;

}  // namespace

extern "C" {

const char* entry_voxel_filter_error() { return error.c_str(); }

// xyz：N×3 的 float32 输入；keep：长度 N 的保留掩码；
// support：长度 N 的邻域点数；cells：输出占据体素数；
// seconds：[0] 哈希聚合耗时，[1] 邻域计数耗时。
// 成功返回 0，失败返回 -1 并通过 entry_voxel_filter_error 提供原因。
int entry_voxel_filter(const float* xyz, uint64_t count, double voxel_size,
                       uint64_t minimum, uint8_t* keep, uint64_t* support,
                       uint64_t* cells, double* seconds) {
  try {
    if (!std::isfinite(voxel_size) || voxel_size <= 0. || minimum == 0) {
      throw std::runtime_error("Invalid filter size/threshold");
    }

    constexpr double kMaxGridCoordinate =
        static_cast<double>(std::numeric_limits<int64_t>::max()) / 2.;

    const auto started = std::chrono::steady_clock::now();

    // 第一遍：把每个输入点量化到体素，并统计每个体素的原始点数。
    std::unordered_map<VoxelKey, VoxelCell, VoxelKeyHash> voxels;
    voxels.reserve(count);
    std::vector<VoxelKey> keys(count);
    for (uint64_t row = 0; row < count; ++row) {
      for (int axis = 0; axis < 3; ++axis) {
        const double cell =
            std::floor(static_cast<double>(xyz[3 * row + axis]) / voxel_size);
        if (!std::isfinite(cell) ||
            std::abs(cell) > kMaxGridCoordinate) {
          throw std::runtime_error("Invalid/overflowing XYZ");
        }
        keys[row][axis] = static_cast<int64_t>(cell);
      }
      ++voxels[keys[row]].count;
    }
    const auto aggregated = std::chrono::steady_clock::now();

    // 第二遍：对每个体素累加 3×3×3 邻域（含自身）的点数。
    for (auto& entry : voxels) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const VoxelKey neighbor{entry.first[0] + dx,
                                    entry.first[1] + dy,
                                    entry.first[2] + dz};
            const auto found = voxels.find(neighbor);
            if (found != voxels.end()) {
              entry.second.support += found->second.count;
            }
          }
        }
      }
    }

    // 第三遍：把所属体素的支持度写回每个输入点。
    for (uint64_t row = 0; row < count; ++row) {
      const uint64_t value = voxels.at(keys[row]).support;
      support[row] = value;
      keep[row] = value >= minimum ? 1 : 0;
    }

    cells[0] = voxels.size();
    seconds[0] =
        std::chrono::duration<double>(aggregated - started).count();
    seconds[1] = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - aggregated)
                     .count();
    return 0;
  } catch (const std::exception& exception) {
    error = exception.what();
    return -1;
  }
}

}  // extern "C"