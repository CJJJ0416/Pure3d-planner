# 当前局部块契约

## 全点块内外幽灵示例

`filter_block_ghost_full_points_v3_air_scatter` 使用 `Filter/configs/block_ghost_example.json`，seed 821010。干净组件包含 4.8×4.8 m 地面、原生 terrain-generator 楼梯、上层平台、栏杆、支柱和实体障碍物。表面采样间距 0.03 m，采样及增强后均不做体素降采样。三角面共享边界的重复坐标由表面采样器精确去重，不移动点。

幽灵按生成时的 points_per_position 控制为少量散点，不输出密集墙片、楼梯副本。每个拖影位置独立抽取少量源表面点，然后施加错位、漂移及扰动。近表面散点对照仍按实际表面距离连续评分。可视化扩大幽灵标记以便查看，没有增加实际点数。

核心区的 volume_scatter 在多个空气体积内独立生成 180 个位置，覆盖不同高度，不保留源墙片形状。源面片身份沿用随机选择的参考点，每个散点有独立平移矩阵，仍能逐点回放。核心区另有 12 个错位点和 24 个短拖影点。边界外点与核心内点使用相同的表面距离标签，不按区块直接判定。

先生成完整增强点集，再查询完整 `clean.obj` 的三角面距离，最后裁切核心和上下文。`rough.pcd` 只有 XYZ，`labels.pcd` 是相同点序的 XYZ+keep_target。`rough_attributes_000.npz` 保存完整点集的来源、面片、实例、变换、标签和原始索引。`ghost_placement_id` 为显式幽灵放置编号，-1 表示基础退化或原有随机配准增强。

`region_id` 为 0 核心、1 上下文外圈、2 上下文之外。它与 `core_mask`、`context_mask` 仅描述空间归属，不用于决定 GT，也不进入模型。`targets.npz` 是完整点集标签，单盒 targets.npz 的 point_index 指回完整 rough 点序。显式幽灵的预定区域见 augmentation_000.json，最终逐点区域以 region_id 为准。

`context.pcd`、`core.pcd` 及对应 labels PCD 分别保留上下文和核心的全部原始行。上下文外的点仍存在于完整 rough、labels 和 NPZ。按 GT≥0.5 导出的 retained_gt.pcd 仅用于观察。comparison.png、top.png、side.png 均绘制全部点，不对显示标记进行抽样。

以下是此前训练数据的契约。

filter_local_10cm_v3已替代大场景训练。见 [完整工作流](TRAINING_WORKFLOW.md)。以下v2字段解释继续适用于逐点来源与系数，v3不要求生成独立大地图或每盒重复PCD。

v3每案例直接生成局部组件，场景共享arrays_<variant>/*.npy保存coord、keep_target、loss_group、semantic_id。windows_<variant>_index/core/offsets.npy保存上下文索引和核心归属。每案例每变体一个集中目标的训练样本。训练特征缓存仅保存核心所需原486维特征与inverse。

同seed的所有案例保持同split。轨迹、语义和组别只参与监督、采样和评价。完整新PCD推理依然按各核心窗口恢复全部原点序。

---

# Filter 数据契约

版本：`filter_trails_v2`。更新日期：2026-11-11。

## 输入与坐标

部署输入只是一张 `FIELDS x y z` 的 PCD。所有点位于同一米制地图坐标系，Z 向上。分盒不改变地图原点。训练使用小盒上下文，输出根据原始 point_index 合并。语义、运动身份、时间步、GT 和 Mesh 均不属于模型输入。

整个 WFC seed 是划分单位。同场景的全部物体模板、增强变体与盒继承 geometry_group_id，不跨 train/val/test。

## 静态参考与来源

`scene.obj/clean.obj` 包含建筑和静止人物、椅子、可移动物体。`face_metadata.npz` 与其三角面行序一致，保存 face_id、semantic_id、instance_id、target_role。当前静态参考面 target_role 全为 1。`terrain.obj` 仅含建筑，`static_objects.obj` 单独保存静止物体。

`clean.pcd` 是 3 cm 表面样本经 5 cm 体素选择原始表面点的结果。clean_attributes 同点序保存面片、实例、语义、法向和原始表面样本编号。动态物体复用这些静止模板，另分配 actor instance_id。静态参考排除所有运动位置。

| source_role | 含义 |
|---|---|
| 0 | 正确建筑观测 |
| 1 | 轻微噪声建筑观测 |
| 2 | 建筑配准错位片段 |
| 5 | 静止物体，属于保留参考 |
| 6 | 建筑渐进配准重影 |
| 7 | 运动物体沿轨迹累积的残影 |

编号 3、4 为旧来源表的随机点/簇，本次不生成。配准类型不根据来源直接标零。运动类型按实例来源全部标零，包括最后位置。

## 每变体逐点数组

rough_*.pcd 只有 XYZ。rough_attributes_*.npz 与其 N 行严格对齐：

| 字段 | 形状 | 含义 |
|---|---|---|
| coord | [N,3] float32 | 最终地图 XYZ |
| keep_target | [N] float32 | 连续保留系数 |
| surface_distance_m | [N] float32 | 到静态三角面参考的距离 |
| source_point_id | [N] int64 | clean.pcd 源点行号 |
| source_face_id | [N] int64 | clean.obj 源三角面行号 |
| source_instance_id | [N] int32 | 原静止模板实例 |
| instance_id | [N] int32 | 当前观测实例，运动 actor 与静止模板分开 |
| semantic_id | [N] int32 | 仅供制标签和分项评价 |
| source_role | [N] uint8 | 来源类型 |
| target_role / target_relevance | [N] | 静态/配准为 1，运动为 0 |
| fragment_id | [N] int32 | 配准片段，无则 -1 |
| trail_id / trail_step | [N] int32 | 对应来源类型内的轨迹与位置编号，无则 -1 |
| transform_id | [N] int32 | 逐点刚体变换表索引 |
| clean_xyz | [N,3] float32 | 源点原始地图位置 |
| coordinate_residual | [N,3] float32 | 刚体变换后的测量噪声残差 |
| is_disconnected_control | [N] bool | 准确断连结构的保留对照标记 |

`augmentation_sources_*.npz` 中的 transforms 为 [K,4,4]。恢复关系为：

`coord = clean_xyz @ transforms[transform_id,:3,:3].T + transforms[transform_id,:3,3] + coordinate_residual`。

逐轨迹记录位于 augmentation_*.json，保留 actor、模板、走廊、路径类型、每步变换编号及抽取比例。动态和配准轨迹编号分别解释，不能混用。

## 连续目标

运动点 keep_target=0。其他点 keep_target=exp(-0.5*(surface_distance_m/0.03)^2)。准确但断连的结构和静止物体高分。位移不等于表面误差，副本落回正确表面也可高分。

每场景只构建一次 Open3D 三角面索引，分批查询四变体。targets_*.npz 单独保存 keep_target 和 surface_distance_m，targets_*.json 保存分类统计及耗时。它们不是概率校准结果。

## 分盒与未来编码接口

独立盒位于 boxes/variant_*/box_*/。rough.pcd 为上下文输入，clean.pcd 为相同范围静态参考，targets.npz 同点序保存目标、距离、point_index、core_mask。rough_attributes 也携带对应标签与原始身份。

核心尺寸 4×4×3 m，外扩 1 m。每点有唯一核心归属，可以出现在其他盒上下文。未来训练仅监督核心。距离查询引用完整静态 Mesh，避免盒边界截断参考。

boxes.json 是权威样本索引，windows_*.npz 保存拼接 point_index、core_mask 和 offsets。恢复整图时按核心点原始索引赋值，不平均改变 XYZ。

后续固定 Tiny 缓存保存 [M,486] voxel_features、inverse、原始点索引与 XYZ 派生的几何统计。活动投影不缓存。RECON 继续复用 F_voxel[inverse[keep_indices]]，不重新编码 filtered PCD。当前未实现编码缓存和模型训练。
