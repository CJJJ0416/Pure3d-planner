# 原始密度与阶段替换

全部 XYZ 使用地图坐标，单位米，Z 向上。保留完整 Mesh 表面 2 cm 采样密度，不做 10 cm 输入降采样。整图先生成 Subtraction，再提取区块，同一全局点在重叠区块中的删除状态一致。

| Split | 地形 seed | 区块数 |
|---|---|---:|
| Train | 836100、836101、836102、836103 | 32 |
| Val | 836200 | 8 |
| Test | 836300 | 8 |

835012 是预览 seed，不进入正式划分。代表块由有／无栏杆直梯、往复梯、旋转梯、斜坡、走廊门洞、平台边界和楼面锚点确定。包含锚点的核心优先，冲突时选距离核心边界最近的未选块，距离相同按 block ID。核心 4×4×3 m，四周上下文 1 m。

## 永久数据

根目录是 `datasets/recon_terrain_curriculum_v1`。

- `base/terrain_<seed>/`：Mesh、原始密度 PCD/NPY、结构元数据、区块索引和 overview。
- `blocks/<terrain>/<block>/`：完整区块、全局行号、固定网格、Mesh 标签、密集评价参考。
- `features/<frontend_id>/<grid_id>/`：完整输入的 PTv3 特征和独立执行 Filter 的完整输入对照。
- `heldout/<terrain>/{light,medium,heavy}/`：固定验证／测试缺损及条件特征。加完整输入共四种版本。
- `permanent.json`、`heldout.json`：永久清单。

`frontend_id` 包含 Filter checkpoint SHA256 和阈值。每个缺损块独立执行冻结 PTv3 和 Filter，不能从完整缓存截行。缓存字段为 `xyz、features[M,486]、inverse、keep_index、global_index、keep_score`。原始保留点坐标和行顺序不变。

固定网格由核心边界＋上下文按 24 cm 对齐，与删除比例无关。删除索引、Filter 索引只用于监督分组和评估，不作为模型输入。GT latent 在 A 阶段训练完成后才生成，按 A 编码器身份缓存。

## 当前训练版本

| Stage | Epoch | 整图缺损范围 | 目标 |
|---|---|---|---:|
| 1 | 1–10 | 2–5% | 3.5% |
| 2 | 11–20 | 5–10% | 7.5% |
| 3 | 21–35 | 10–20% | 15% |
| 4 | 36–50 | 20–30% | 25% |

每图每阶段一份缺损。固定 seed 驱动不规则空洞、弯曲断带和局部稀疏的持续区域流，首次达到整图目标即停止，不截断最后区域。报告实际比例和各块核心比例。所有 32 个缺损块每轮各参与一次，另选 8 个完整输入对照，合计 40 次更新，其中完整对照占 20%。块不再按比例被淘汰。A 阶段只用 32 个完整输入，每轮 32 次更新。

- `train_current/` 只保存当前阶段四图的缺损 PCD、删除索引和条件缓存。
- `train_pending/` 在晋级时准备下一阶段，完整后才发布。
- `manifest.json` 是训练读取的原子发布清单，记录 `stage_id、data_version`。
- `stage_state.json` 记录当前版本及待晋级阶段。

每完成 B 阶段第 10、20、35 轮先保存 `last.pt`，释放旧 DataLoader。独立准备进程生成下一阶段，避免消耗训练 RNG。完成后切换目录、原子发布清单、删除旧版本并重建 DataLoader。模型、优化器、学习率调度和 global step 不重置。切换过程中的 `train_previous` 仅用于恢复目录重命名，不是版本归档，发布后立即删除。

准备中断时 `train_current` 保留，重启复用 pending 中已完成的原子缓存。checkpoint 的 `curriculum_stage、data_version、pending_stage` 用于恢复匹配。固定 GT、完整输入、验证／测试不在替换范围。正式数据只初始化 stage 1，不提前生成 2–4。

## Mesh 标签

五尺度单元通过 Mesh 三角面精确求交得到标签。正单元监督单元内坐标偏移，最近完整 PCD 点只用于观测／缺失分组。空洞里的完整 Mesh 表面仍是正样本。标签 0 表示没有目标表面通过，不等于可通行。Loss、组权重和 A 阶段验证门槛见 `SUPERVISION_LOSS.md`。
