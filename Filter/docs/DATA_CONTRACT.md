# 全点评分数据契约

版本 `filter_scoring_mesh_full_20261112`，地图坐标系 `map`，单位米，Z 向上。模型只读取 XYZ，不使用 PCD VIEWPOINT 作为采集位姿。

## 参考与连续监督

完整静态 Mesh 包含楼梯、地面、墙、栏杆及静止物体。每场景建立一次 Open3D 三角面距离索引。

- 运动残影全部 `keep_target=0`，包括最后位置。
- 其他点 `keep_target=exp(-0.5*(surface_distance_m/0.03)^2)`。
- 准确静止结构按表面距离得高分，位置在块外、密度低或断连本身不降低标签。
- 配准副本落回正确表面附近时允许高分。

本轮按 3 cm 间距采样完整 Mesh，保留全部点。关闭人工缺口、条带删除、主动断连和输入体素降采样。Tiny 内部体素化仍通过 inverse 恢复到原始核心点。

## 每场景与逐点文件

| 文件 | 内容 |
|---|---|
| `clean.obj`、`face_metadata.npz` | 静态三角面及其身份 |
| `clean.pcd`、`clean_attributes.npz` | 完整表面点及对应来源 |
| `rough_000.pcd` | 最终 XYZ 输入，只有 XYZ |
| `gt_000.pcd` | 同点序 XYZ + keep_target，显示用途 |
| `rough_attributes_000.npz` | 来源、距离、标签、语义、实例、变换索引等逐点属性 |
| `augmentation_sources_000.npz` | 刚体变换表等监督记录 |
| `augmentation_000.json` | 固定变体、增强来源、标签分布与耗时 |
| `arrays_000/` | `coord.npy`、`keep_target.npy`、`loss_group.npy`、`semantic_id.npy` |
| `overview.png`、`preview/` | 几何总览和同点集 GT 对照 |

来源编号 0/1 为建筑及轻微测量噪声，2/6 为悬空或配准增强，5 为静止物体，7 为动态残影。来源、实例、语义与变换仅用于生成、采样、监督和评价。

## 分块与缓存

核心 4×4×3 m，上下文向外 1 m，map 网格 shift 固定为 0，与部署相同。每点只属于一个核心，可以进入多个上下文。

`windows_000_index.npy` 保存拼接后的完整点索引，`windows_000_core.npy` 保存同长度核心掩码，`windows_000_offsets.npy` 保存各块区间。`boxes/variant_000/box_00000/box.json` 保存边界和点数。当前不为每盒重复保存 PCD。

`point_index` 指向该变体的原始点序。监督只计算核心点，上下文不重复计分。相同 seed 的全部变体和盒保持同一 split。

缓存每盒保存 `features.npy [M,486]`、`inverse.npy [N_core]`、`relative.npy [N_core,3]`、`geometry.npy [N_core,6]`，以及目标、来源组、语义和原始索引。编码器权重身份及训练集统计位于缓存的 `encoding.json`。

## 独立预测与可选选择

`Filter.inference.predict` 输出 `prediction.pcd`，字段为 XYZ + keep_score，点数和点序与输入完全相同。`prediction.npz` 保存 coord、keep_score 和 point_index。`prediction.json` 保存权重路径、分块、坐标及分阶段耗时。

`Filter.inference.threshold` 接收已有评分文件，显式按阈值输出 `filtered.pcd`、`removed.pcd`、`filter_mapping.npz` 和 `selection.json`。映射中的 point_index 对应评分文件原始行号。0.2 是本次用户指定的操作阈值，不是测试集选优或校准概率阈值。
# Incremental Faster-LIO supervision

`common/scanning` adds a separate single-scene pilot. Raw rolling scans and IMU
are passed through the existing Faster-LIO process. Registered returns retain
their input frame/ray IDs through the dense, time-sorted row mapping. The local
LIO publisher drops curvature, so provenance is audited with exact dense counts
and the preserved reflectivity sequence, not inferred by spatial neighbours.

One fixed initial-frame rigid transform aligns the entire estimated map to the
static reference. Mesh surface distance supplies continuous keep_target, with
moving-instance returns fixed at zero. Mesh-sampled PCD nearest distances are
an additional alignment diagnostic, not the formal teacher. Repeated returns
may share a reference surface. Missing reference surfaces indicate coverage,
not low-quality observed points. Full original returns are retained. The model
receives only map XYZ. Core/context indices use the existing 4x4x3 m plus 1 m
interface. This pilot does not join an existing train/val/test split.
