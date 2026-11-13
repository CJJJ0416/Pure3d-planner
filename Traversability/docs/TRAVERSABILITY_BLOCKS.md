# 建筑输入与 Tomogram 查询分块

第 6 步的数据接口。使用已有 PCD 和原生 Tomogram，不重新生成教师，不启动训练。

## 生成

在工作区根目录运行：

```bash
/home/cjh/glocal2-nav/.venv/bin/python -m Traversability.data.blocks
```

参数在 `Traversability/configs/dataset.json` 的 `blocks` 中。`--batch batch_01` 只替换指定批次。完整 `Traversability.data.generate` 流程也会在教师生成后执行分块。

- 核心块为 `6×6×4 m`，采用 `[lower, upper)` 半开边界。
- 建筑输入外扩 `1.5 m` 上下文，完整保留盒内输入点。空间树每场景构建一次。
- 每个原生 Tomogram 查询点只归属一个核心块，没有重复或遗漏。一个查询点对应25列原生归一化 cost，其中24列用于学生训练。
- 保留全部零代价、障碍代价和边界点。没有高度连续、回波匹配或覆盖筛选。
- 另外保存同一建筑输入的 `0.5 m` 体素质心作为全楼粗几何，只包含 XYZ。第 7 步已接入共享编码器与查询聚合。
- 几何近邻预计算为每个查询的局部 K=8 和全楼 K=4，索引仅用于查询特征，不改变 cost 标签或监督范围。
- 划分继承场景 seed 的 train/val/test，块不会跨场景混入其他 split。

## 文件

```text
datasets/wfc_capability_dr_stand_v3/
├── manifest.json
├── blocks.json                         # 全部分块、场景、条件顺序与坐标约定
└── scene_<seed>/
    ├── terrain_downsampled.pcd          # 原始建筑输入，保持不变
    ├── tomogram_<profile>.pcd           # 原始教师监督，保持不变
    ├── vst_<profile>.pickle             # 原始中间地图，保持不变
    ├── blocks.report.json
    ├── blocks_overview.png
    └── blocks/
        ├── input_coord.npy             # [N,3] 建筑世界 XYZ
        ├── query_coord.npy             # [M,3] 原生 Tomogram 世界 XYZ
        ├── cost.npy                    # [M,K] 归一化标签
        ├── conditions.npy              # [K,5] 机器人条件
        ├── global_coord.npy            # [G,3] 全楼粗几何
        ├── global_neighbors.npy        # [M,4] 全楼粗点索引
        └── block_<index>.npz           # input_indices、input_core、query_indices、query_neighbors
```

NPY 是训练缓存，支持 mmap 按需读取，不替代原生教师 pickle。NPZ 只保存未压缩索引，不在每个块中重复保存坐标和全部条件标签。索引均对应原 PCD 行号，全部条件查询点的世界 XYZ 和顺序遵循原生导出契约。

`input_core` 只表示建筑输入点是否位于核心盒，用于几何采样和可视化。它不筛选 Tomogram 监督。查询点全部位于核心盒内。

## 读取与合批

```python
from torch.utils.data import DataLoader
from Traversability.data.blocks import TomogramBlocks, collate_tomogram_blocks

data = TomogramBlocks("datasets/wfc_capability_dr_v2", "train")
loader = DataLoader(data, batch_size=2, num_workers=2,
                    collate_fn=collate_tomogram_blocks)
batch = next(iter(loader))
```

| 字段 | 形状 | 用途 |
|---|---|---|
| `input_coord` | `[ΣN,3]` | 建筑几何，后续由 PTv3 编码 |
| `query_coord` | `[ΣM,3]` | 独立查询位置，后续聚合几何特征 |
| `global_coord` | `[ΣG,3]` | 全楼粗几何上下文 |
| `target` | `[ΣM,K]` | 只用于计算损失 |
| `conditions` | `[B,K,5]` | 教师顺序：clearance_min、slope_max、step_max、interval_free、stand_ratio |
| `input/query/global_offset` | `[B]` | 各点集每块的累计结束位置 |
| `input/query/global_batch` | `[ΣN/ΣM/ΣG]` | 各点所属的批次块编号 |
| `origin` | `[B,3]` | 核心块世界中心 |
| `input_index`、`query_index` | `[ΣN]`、`[ΣM]` | 原场景行号，配合 scene/batch 追溯来源 |
| `query_neighbors` | `[ΣM,8]` | 指向当前拼接后的原始输入点，尚非体素特征索引 |
| `global_neighbors` | `[ΣM,4]` | 指向当前拼接后的全楼粗点 |

缓存始终保留世界坐标。读取时三类坐标共用一个块中心进行平移，不缩放，不改变 Z 向上和米制距离：

`world_query = query_coord + origin[query_batch]`。

输入几何与查询点数量不同，不能用输入体素的 inverse 代替查询匹配。第 7 步已接入多尺度特征查询、FiLM 和 CostHead。模型使用 dataset.py 的 PointBlocks/collate_blocks，在体素化后通过 inverse[query_neighbors] 转为特征索引，保持输入与查询的独立性。训练增强会同时绕 Z 轴旋转三类坐标，近邻关系保持不变。实现及真实 GPU 前向见 [代价模型](TRAVERSABILITY_MODEL.md)，尚未训练。

PointBlocks选取interval_free=0.5的24组条件，学生顺序为clearance_min、slope_max、step_max、stand_ratio，collate_blocks输出条件 `[B,24,4]`、target `[ΣM,24]`。tall_robot仅作教师诊断。

上述 Tomogram 位置只用于训练或监督评估。部署由 inference.py 直接对建筑原始 PCD 全部 XYZ 分块，每个输入点恰好作为输出位置一次，不读取教师查询位置。输入和输出都基于地图原点，分块中心平移仅是计算过程中的坐标变换，导出直接使用原建筑 XYZ。

## 历史三参数数据统计

| 划分 | 场景 | 核心块 | 原生查询点 | 五条件 cost 标签 |
|---|---:|---:|---:|---:|
| train | 4 | 342 | 999,422 | 4,997,110 |
| val | 1 | 75 | 211,406 | 1,057,030 |
| test | 1 | 104 | 252,715 | 1,263,575 |
| 合计 | 6 | 521 | 1,463,543 | 7,317,715 |

## 当前四参数数据

新数据位于datasets/wfc_capability_dr_stand_v3。各场景有25组教师结果，其中24组用于学生，条件顺序为[clearance_min,slope_max,step_max,stand_ratio]。建筑输入、查询XYZ和已保存近邻复用既有几何，重新计算全部教师代价。

| 划分 | 场景 | 核心块 | 原生查询点 | 学生cost标签（24组） |
|---|---:|---:|---:|---:|
| train | 4 | 342 | 999,422 | 23,986,128 |
| val | 1 | 75 | 211,406 | 5,073,744 |
| test | 2 | 177 | 441,150 | 10,587,600 |
| 合计 | 7 | 594 | 1,651,978 | 39,647,472 |
