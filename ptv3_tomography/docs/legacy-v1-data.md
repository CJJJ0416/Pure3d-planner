# 点云与 Tomography 标签配对（D1）

> 历史记录。2026-11-06 已切换为原生 VST 标签可用性监督，文中的三重筛选、旧验证与旧 v1 入口已删除。当前代码见 [原生监督清理](TEACHER_SUPERVISION_CLEANUP.md)。

2026-11-04：单场景配对、连续窗口加载、独立几何核对和 Isaac Sim 原生叠加已完成。当前产物用于下一步小批量训练检查，尚无 PTv3 权重或导航训练结果。

后续接口澄清：每次网络输入为历史多帧雷达自身 odom、原始雷达点云及时间。用户指定以当前雷达为输出参考；下文记录的 D1 v1 标签则是机身原点重力/yaw 系。两者需要 D1-R0 迁移与复验后才能进入新参考契约下的 P1。已有数据、指纹和验收未被改写；本文件下文的 v1 字段仍描述当前磁盘产物。

## 已生成的数据

| 包 | 目录（仓库根目录下） | 原始帧 / 完整窗口 | 高度 / 代价监督单元 |
| --- | --- | --- | --- |
| 历史窗口 K=8 | `outputs/mid360_tomography_pairs_20261104` | 130 / 123 | 72,119 / 69,607 |
| 单帧 K=1 | `outputs/mid360_tomography_pairs_k1_20261104` | 130 / 130 | 20,604 / 16,346 |

监督单元数按帧累计，包含重叠区域，不是独立场景或独立样本数。K=8 的代价监督包含 34,458 个 cost≥10 单元、187 个 cost=60 单元；原始可用标签保存 53,508 个 cost=60 单元。K=1 的默认保守掩码内没有 cost=60 单元，该包适合检查单帧链路，不作为完整危险区域训练集。

两包共享同一坡道场景、同一条上坡参考轨迹和同一组实际 PhysX 扫描：

- 教师场景：`outputs/ramp_platform_side_20261103_cost60_threshold10`。
- 原始扫描：`outputs/isaac_mid360_native_viewer_20261104`，每帧 1000 射线、10 Hz。
- 轨迹为位姿驱动参考路径采样；不代表机器人自主上坡或完整四足动力学。
- 当前包不设 train/dev/test；多建筑谱系划分是 D3 的工作。

## 数据契约

局部系以当前机身原点为中心，只保留机身 yaw，Z 与世界重力轴一致；X 前、Y 左、Z 上。局部地图 64×64，分辨率 0.1 m，覆盖单元宽度 6.4 m，中心索引 `[32,32]` 对应 XY 零点，格心从 -3.2 到 3.1 m。

标签轴顺序为 `[height_bin, x, y]`，不是图片的 `[y,x]`。6 个高度带边界为 `[-2,-1.5,-1,-0.5,0,0.5,1]` m，相对当前机身 Z；左闭右开。保存实际地面高度 `ground_z - body_z`，不使用 slice 编号、slice_h0 或路径 Z 代替地面高度。

按世界 XY 在原生 VST 中做最近邻查询。各源层的实际高度落入同一高度带后，只有高度相同（容差 1 mm）的重复表示才合并，并记录选中的 `source_layer`；此时选择同高候选中的最小代价。不同高度落入同一带时标为 ambiguous 并排除，不能跨层选低代价。高度带外也不自动视为障碍或无支撑。

### 标签与独立掩码

- `cost_raw`：原始代价 0–60；`cost_normalized=cost_raw/60`，固定尺度，无逐帧 min-max 或规划阈值裁剪。
- `height_m`：实际地面相对当前机身高度；诊断分片的未知位置保存 NaN。
- `target_available`：有有限且无高度歧义的教师记录，仅表示教师标签存在。
- `mask_height`：教师可用、源网格 3×3 高度连续（邻格差≤4 cm），且当前/历史真实命中点与该高度吻合（≤4 cm）。排除重采样边界。
- `mask_cost`：高度掩码成立，且 0.6 m 邻域的观测覆盖率≥0.3。代价头、高度头分别使用自己的掩码。
- `support_target=-1`、`mask_support=0`：全部 unknown，支撑损失关闭。观测到一块表面并不证明机器人足迹范围内有支撑或净空。
- `surface_evidence_count`、`free_evidence_count`、`observed_band`、`cost_context_coverage`：保留观测依据。自由证据只采样在已确认命中的射线终点之前；无回波、近界遮挡和非法范围不能填成自由或安全。

**代价的局部覆盖判据是工程代理。** 已验证数值来自正确教师单元，但尚未证明骨架/膨胀等完整 VST 代价都能由局部窗口恢复。P1 可以检查前向、掩码损失与恢复；正式共同预训练前应另外验证代价可观测性并补充危险边缘、实体障碍与跨层诊断。标签存在、数值对齐、局部可恢复性分别记录，不把前两者当成第三者的证据。

### 模型读取

`TomographySequenceDataset` 返回 NumPy 字典，可交给 PyTorch 默认 DataLoader：

| `inputs` 字段 | 单个 K=8 样本形状 | 含义 |
| --- | --- | --- |
| `points_sensor_m` | `[8,1000,3]` | 每帧原始传感器系命中点 |
| `points_local_m` | `[8,1000,3]` | 全部历史点变换到当前局部系 |
| `point_mask` | `[8,1000]` | 命中有效位；缺测点零填充 |
| `relative_sensor_poses` | `[8,4,4]` | 当前局部系←历史传感器系 |
| `relative_body_poses` | `[8,4,4]` | 当前局部系←历史机身系 |
| `time_age_s` | `[8]` | 当前扫描结束时间减历史结束时间，约 0.7→0 s |

`targets` 中 cost、height、support、available 及各 mask 为 `[6,64,64]`。加载时把未知 cost/height 零填充以保持数值有限，训练必须先索引各自的有效掩码再计算损失，不能把零填充当安全/高度零。绝对世界位姿、完整地图和源层索引只保留作诊断，不在 `inputs` 中。

窗口不跨 episode、不取未来帧。K=8 从 `[0..7]` 到 `[122..129]`，起点 frame0 仅作诊断；不足 K 的帧不伪造完整训练窗口。**掩码依赖历史长度，K=1 必须重新导出，不能直接截取 K=8 输入并沿用标签掩码。**

### T 的方向与查表对象

记 `T_A_B` 为把 B 系齐次坐标变到 A 系的 4×4 刚体变换。当前仿真由已知机身位姿和雷达安装外参计算 `T_W_S(t) = T_W_B(t) @ T_B_S`，包含旋转和平移；只有雷达世界 XYZ 还不能确定完整 T，当前也未用 ICP/SLAM 估计它。

雷达点变世界坐标：`p_W = R_W_S(t) @ p_S + t_W_S(t)`。历史点落到当前局部系：`p_L0 = inv(T_W_L0) @ T_W_S(t) @ p_S`（齐次形式）。每帧使用各自的时刻位姿。当前场景的世界系与 VST 地图系一致；若以后不一致，还需显式乘 `T_Map_World`。

实际监督使用局部规则网格：`q_L → T_W_L @ q_L → 原生 VST XY 索引 → 按实际地面高度分带/选层 → cost、height、独立 mask`。因此输入点云与监督格心不必一一对应。雷达命中和过去窗口的自由射线覆盖用于确定可监督区域；不能将每个三维回波的 Z 直接当成地面高度或支撑标签。查询的是 VST 栅格通行代价，不是整条路径的优化总目标。

```python
from isaac_nav.data.tomography_pairs import TomographySequenceDataset

dataset = TomographySequenceDataset("outputs/mid360_tomography_pairs_20261104")
sample = dataset[0]
print(sample["inputs"]["points_local_m"].shape)  # (8,1000,3)
```

## 可复现入口

从仓库根目录执行；离线配对和数值核验不启动 Isaac，也不需要 GPU。使用现有 `env_isaaclab` 可同时满足 NumPy/trimesh 和原生检查的依赖。

```bash
conda activate env_isaaclab
python isaac_training/scripts/perception/pair_mid360_tomography.py \
  --scene outputs/ramp_platform_side_20261103_cost60_threshold10 \
  --capture outputs/isaac_mid360_native_viewer_20261104 \
  --output outputs/my_pairs_k8

python isaac_training/scripts/validation/validate_mid360_tomography_pairs.py \
  --dataset outputs/my_pairs_k8

# 单帧使用新的目录，重新计算观测掩码。
python isaac_training/scripts/perception/pair_mid360_tomography.py \
  --scene outputs/ramp_platform_side_20261103_cost60_threshold10 \
  --capture outputs/isaac_mid360_native_viewer_20261104 \
  --history-frames 1 --output outputs/my_pairs_k1

# 原生视口截图；加 --gui --keep-open 可在窗口循环查看三处锚点。
python isaac_training/scripts/perception/inspect_mid360_tomography_pairs.py \
  --dataset outputs/my_pairs_k8
```

每次导出使用新目录。原生检查也保留已有 `validation/native`，重拍时应选择新的派生包/证据目录。manifest 引用本机原始扫描的绝对路径并校验 SHA256；换机器需带上源资产和原始扫描，重新导出 manifest。`outputs/` 不随 Git 源码分发。

### 交互查看已有标签

以下入口直接打开 D1 保存的 `validation/native/paired_scene.usda`，复用已有原生场景，在内存中切换标签显示，保持源数据不变：

```bash
python isaac_training/scripts/visualization/view_mid360_tomography_labels.py \
  --dataset outputs/mid360_tomography_pairs_20261104 \
  --output outputs/my_label_viewer --frame 52
```

默认保留 Isaac Sim GUI。`Tomography labels | D1` 面板提供帧滑块、Start/Ramp/Platform、模式、高度带、点云开关和相机复位。三个模式分别显示：参与代价损失的标签、全部可用教师代价、代价掩码（绿=参与，紫=排除）。cost 使用绿0→黄30→红60，青色是因果窗口点云；没有教师记录的位置隐藏，不能理解为安全或无支撑。

启动时保存三个模式的原生视口截图并验证面板回调，随后保持交互；加 `--headless` 可只截图后退出。原生截图通过 capture completion 和 renderer async flush 等待完成，避免未写完时过早退出。实际查看记录：`outputs/mid360_tomography_label_viewer_20261104_v2/report.json`；frame52 有 4046 个可用教师单元，其中 331 个进入 cost 损失、342 个进入 height 损失，历史点云 2371 个命中。

## 文件与验收

- `dataset_spec.json`：坐标、单位、轴、时间、高度带、掩码、输入边界、原始假设和源文件指纹。
- `manifest.json`：逐帧原始引用/标签分片/hash、episode、连续窗口。
- `labels/frame_XXXXX.npz`：局部代价/高度/支撑、独立掩码、源层与原生网格位置、观测覆盖和历史帧号。
- `pairing_report.json`：配对与全部窗口加载的形状、数量与有效比例。
- `validation/alignment_report.json`：独立原始三角网格高度误差、源标签精确一致性、时序与回归检查。
- `validation/native/*.png`、`report.json`、`paired_scene.usda`：实际 Isaac 原生截图和叠加场景。青色为保存的 PhysX 命中点，绿→红为监督 cost 0→60；标签上移 2.2 cm 仅避免显示遮挡，不改数值标签。
- K=8 的 `completion_report.json`：本轮最终验收汇总与证据指纹。

K=8 全部 130 帧核验：原始传感器→世界变换最大残差 `2.39e-7 m`；72,119 个高度监督单元对同 XY 原始向上三角面的最大高度差 `0.02535 m`，均值 `0.00231 m`。K=1 也通过全部 130 窗口核验。该几何检查不验证机器人足迹支撑、机身净空或路线可执行性。

原生锚点为 frame0（起点）、frame52（坡中）、frame129（平台）。三者实际显示了对应点云、标签、机身原点、传感器原点与局部 XY 轴。合成回归另覆盖同 XY 上下层、同高重复层、同带异高排除、yaw90/XY 轴映射、无回波不算自由、历史长度不一致拒绝加载。

初版边缘检查失败产物留在 `outputs/mid360_tomography_pairs_20261104_draft_boundary_check`，有 `failed_do_not_train` 标记；当前引用的正式 D1 包是经过收紧连续性检查后重新生成并通过核验的版本。
