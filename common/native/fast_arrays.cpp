// 精确、稳定的体素分组与显示代表点选择（单次扫描完成）。
//
// 两种输出模式：
//   partition=false：每个体素输出一个"显示代表点"，取该体素内 cost 最大的
//                    输入行；cost 相同时保留最早出现的行。
//   partition=true ：除分组信息外，输出按体素键排序的全部行号，
//                    组内保持原始输入顺序。
//
// 输入数组可以是 float32 或 float64，并允许行/元素 stride 非连续；
// 本模块只按 stride 读取，不复制、不修改输入。
//
// 对外接口为 C ABI，由 common/native/arrays.py 通过 ctypes 调用。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// 量化后的三维体素索引。
struct VoxelKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

// splitmix64：让相邻的整数坐标也能得到均匀的哈希值。
uint64_t mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    return static_cast<size_t>(
        mix64(static_cast<uint64_t>(key.x)) ^
        mix64(static_cast<uint64_t>(key.y) + 0x9e3779b97f4a7c15ULL) ^
        mix64(static_cast<uint64_t>(key.z) + 0x243f6a8885a308d3ULL));
  }
};

// 按 x、y、z 字典序排序体素键。
bool less_key(const VoxelKey& a, const VoxelKey& b) {
  if (a.x != b.x) return a.x < b.x;
  if (a.y != b.y) return a.y < b.y;
  return a.z < b.z;
}

// 一个体素分组的累计信息。
// original 是分组首次出现时的编号，用于把排序后的分组映射回输入点。
struct Group {
  VoxelKey key;
  uint64_t count = 0;
  uint64_t representative_row = 0;  // cost 最大的输入行
  uint32_t original = 0;
  double max_cost = 0.;
};

// 一次扫描的全部输出；句柄指向该结构。
struct Result {
  std::vector<Group> groups;      // 按体素键升序排列
  std::vector<int64_t> rows;      // 代表行或重排后的全部行号
  std::vector<uint64_t> offsets;  // 分组在 rows 中的起止偏移
};

// 最近一次失败的原因，每个线程独立保存。
thread_local std::string error;

// 读取 float32/float64 数组的第 index 个元素；dtype 是元素字节宽度（4/8）。
double read_value(const void* data, int dtype, uint64_t index) {
  return dtype == 4
             ? static_cast<double>(static_cast<const float*>(data)[index])
             : static_cast<const double*>(data)[index];
}

// 按输入顺序分组；Partition=true 时同时保留每个点所属的原始分组。
template <bool Partition>
std::unique_ptr<Result> scan(const void* xyz, int xyz_dtype, uint64_t xyz_stride,
                             const void* cost, int cost_dtype,
                             uint64_t cost_stride, uint64_t point_count,
                             const double* voxel_size) {
  constexpr double kMaxGridCoordinate =
      static_cast<double>(std::numeric_limits<int64_t>::max()) / 4.;

  auto result = std::make_unique<Result>();
  auto& groups = result->groups;

  // partition 模式只需较少预留（结果会被重排），显示模式按输入规模预留。
  std::unordered_map<VoxelKey, uint32_t, VoxelKeyHash> index;
  index.reserve(std::min<uint64_t>(point_count, Partition ? 4096 : 1000000));

  std::vector<uint32_t> membership;
  if (Partition) membership.resize(point_count);

  for (uint64_t row = 0; row < point_count; ++row) {
    const double x = read_value(xyz, xyz_dtype, xyz_stride * row);
    const double y = read_value(xyz, xyz_dtype, xyz_stride * row + 1);
    const double z = read_value(xyz, xyz_dtype, xyz_stride * row + 2);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      throw std::runtime_error("Nonfinite XYZ in array operation");
    }

    // 米制坐标 → 整数体素索引，先检查可安全放入 int64_t。
    const double ix = std::floor(x / voxel_size[0]);
    const double iy = std::floor(y / voxel_size[1]);
    const double iz = std::floor(z / voxel_size[2]);
    if (std::abs(ix) > kMaxGridCoordinate ||
        std::abs(iy) > kMaxGridCoordinate ||
        std::abs(iz) > kMaxGridCoordinate) {
      throw std::runtime_error("Grid coordinate exceeds int64 range");
    }
    const VoxelKey key{static_cast<int64_t>(ix), static_cast<int64_t>(iy),
                       static_cast<int64_t>(iz)};

    // 未提供 cost 时所有点同分，代表点退化为该体素的首个输入行。
    const double score =
        cost ? read_value(cost, cost_dtype, cost_stride * row) : 0.;
    if (!std::isfinite(score)) {
      throw std::runtime_error("Nonfinite display cost");
    }

    const auto inserted =
        index.emplace(key, static_cast<uint32_t>(groups.size()));
    const uint32_t group_id = inserted.first->second;
    if (inserted.second) {
      groups.push_back(Group{key, 0, row, group_id, score});
    }

    Group& group = groups[group_id];
    ++group.count;
    // 严格大于：cost 相同时保留最早出现的行。
    if (score > group.max_cost) {
      group.max_cost = score;
      group.representative_row = row;
    }

    if (Partition) membership[row] = group_id;
  }

  // 统一按体素键排序，保证输出与输入顺序无关。
  std::sort(groups.begin(), groups.end(),
            [](const Group& a, const Group& b) {
              return less_key(a.key, b.key);
            });

  // 前缀和：offsets[i] 是第 i 组在 rows 中的起始位置。
  result->offsets.resize(groups.size() + 1);
  result->offsets[0] = 0;
  for (size_t i = 0; i < groups.size(); ++i) {
    result->offsets[i + 1] = result->offsets[i] + groups[i].count;
  }

  if (Partition) {
    // 把每个原始分组映射到排序后的槽位，再按输入顺序填充行号。
    result->rows.resize(point_count);
    std::vector<uint64_t> cursor(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
      cursor[groups[i].original] = result->offsets[i];
    }
    for (uint64_t row = 0; row < point_count; ++row) {
      result->rows[cursor[membership[row]]++] = static_cast<int64_t>(row);
    }
  } else {
    // 每个体素只输出一个代表行。
    result->rows.resize(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
      result->rows[i] = static_cast<int64_t>(groups[i].representative_row);
    }
  }

  return result;
}

}  // namespace

extern "C" {

// 扫描输入并返回结果句柄；失败返回 nullptr，原因由 fast_arrays_error 读取。
void* fast_arrays(const void* xyz, int xyz_dtype, uint64_t xyz_stride,
                  const void* cost, int cost_dtype, uint64_t cost_stride,
                  uint64_t point_count, const double* voxel_size,
                  int partition) {
  try {
    if (partition) {
      return scan<true>(xyz, xyz_dtype, xyz_stride, cost, cost_dtype,
                        cost_stride, point_count, voxel_size)
          .release();
    }
    return scan<false>(xyz, xyz_dtype, xyz_stride, cost, cost_dtype,
                       cost_stride, point_count, voxel_size)
        .release();
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
}

const char* fast_arrays_error() { return error.c_str(); }

// out[0] = 体素分组数，out[1] = rows 长度。
void fast_arrays_sizes(void* handle, uint64_t* out) {
  const Result& result = *static_cast<const Result*>(handle);
  out[0] = result.groups.size();
  out[1] = result.rows.size();
}

// 复制 rows；partition 模式下再复制分组键与 offsets。
void fast_arrays_copy(void* handle, int64_t* rows, int64_t* keys,
                      uint64_t* offsets) {
  const Result& result = *static_cast<const Result*>(handle);
  std::copy(result.rows.begin(), result.rows.end(), rows);
  if (keys) {
    for (size_t i = 0; i < result.groups.size(); ++i) {
      keys[3 * i] = result.groups[i].key.x;
      keys[3 * i + 1] = result.groups[i].key.y;
      keys[3 * i + 2] = result.groups[i].key.z;
    }
  }
  if (offsets) {
    std::copy(result.offsets.begin(), result.offsets.end(), offsets);
  }
}

void fast_arrays_free(void* handle) {
  delete static_cast<Result*>(handle);
}

}  // extern "C"
