# 保留双输出的离线对照

本文中的对照实验命令依赖本地保留的实验模块，该模块不随源码发布。正式在线入口见 [Entry](../README.md)。
真实 KNN 邻域统计、编码前孤立点过滤、完整链路计时和楼梯误删审计见 [KNN_FILTER_COMPARE.md](KNN_FILTER_COMPARE.md)。

当前在线 Entry 仍只发布原始 cost。本工具不发布 ROS 消息、不改变在线模型或规划器。

全部行输出以下字段：

| 字段 | 含义 |
|---|---|
| XYZ | 原始坐标与点序 |
| trav_cost | 原始通行代价，不覆盖 |
| keep_score | 保留评分，越高越倾向保留 |
| weighted_cost | 两路分数的乘积，仅作对照 |
| valid | keep_score ≥ 指定阈值的独立掩码 |

分数均不按校准概率解释。低 keep_score 点不是已确认自由空间。surface A* 的 cost<0.5 是可搜索门槛，因此单独降低代价不能代表剔除点。真正使用 valid 时，应在图节点和邻域聚合之前过滤，而不是仅隐藏 RViz 显示点。

## 编码前删除模拟低分上下文，再真实重算 cost

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws/utils/PointTransformerV3:/home/cjh/Pure3D_ws \
MPLCONFIGDIR=/tmp/entry-context-filter-matplotlib \
python3 -m Entry.experiments.context_filter_compare --output /tmp/entry_context_filter
```

GPU 模型使用原 Tiny best49，条件不变。读取保留的异常帧 ROI 和完整上下文快照，分别固定各自的地面查询点。四组条件为未过滤基线、注入 30 点、编码前删除这 30 点，以及从原有上下文按固定空间区域删除候选点。所有代价均由模型重新预测，不使用乘积替代。

模拟分数规则在推理前保存，不按 cost 高低选点。已有候选点不等于已确认幽灵点，可能包括真实结构。异常帧的原在线完整上下文已清理，因此重新计算其未过滤基线，历史 cost 不作为直接过滤前值。

生成 before/after/delta 静态图、同序 PCD/NPZ、上下文移除点、keep_score、模型指纹和逐次强制重算报告。查看隔离域 RViz：

```bash
source common/unitree_gazebo/sim/env.sh
ROS_DOMAIN_ID=186 ROS_LOCALHOST_ONLY=1 \
python3 -m Entry.experiments.context_filter_compare --view-only --output /tmp/entry_context_filter \
  --dataset complete_context_snapshot --case injected
```

切换 Before / After 图层比较。`--case candidates --dataset recorded_spike_frame` 查看已有点的模拟候选过滤。两路共用完全相同的显示行、world 坐标和固定 0–1 色标。粉色点表示本次删除的上下文点。该工具不修改在线 cost 或导航，不启动 Gazebo。

## 没有 Filter 权重时模拟

```bash
cd /home/cjh/Pure3D_ws
MPLCONFIGDIR=/tmp/entry-score-matplotlib python3 -m Entry.experiments.simulate_scores \
  --output /tmp/entry_simulated_scores
```

默认读取已保存的在线第 76 帧 XYZ+cost，保留原始数据，追加 30 个合成点。keep_score 全部手动赋值。输出显式标记为模拟，不评价真实 Filter 准确率，不重跑 Cost，不测在线耗时。另用当前 surface 后端演示乘积将合成幽灵桥纳入可搜索图的反例。

## 两路已有预测文件

```bash
python3 -m Entry.experiments.score_compare \
  --cost /absolute/cost.pcd --filter /absolute/filter.pcd \
  --keep-threshold 0.8 --frame world --plot \
  --output /tmp/entry_dual_scores
```

输入 PCD 分别具有 XYZ+trav_cost 与 XYZ+keep_score，或者 NPZ 中分别有 coord、trav_cost/keep_score。要求同一快照、同一坐标系、XYZ 和点序完全一致。不会用最近邻把不同点强行配对。0.8 为示例阈值，报告同时扫 0.2、0.5、0.8、0.95，未利用真值选优。

输出 scores.pcd、scores.npz、comparison.json，保留全部点。NPZ 另有 raw_pass、product_pass、gated_pass，分别对应原始低 cost、乘积低 cost、valid 与原始低 cost 的交集。报告直接统计相乘后跨越 0.5 的点数，这不是误报率。

## 有 Filter 权重时预测保存的完整上下文

```bash
PYTHONPATH=/home/cjh/Pure3D_ws/utils/PointTransformerV3:/home/cjh/Pure3D_ws \
MPLCONFIGDIR=/tmp/entry-score-matplotlib \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Entry.experiments.score_compare \
  --snapshot /home/cjh/Pure3D_ws/datasets/fixtures/entry/stable_context_snapshot \
  --filter-checkpoint /absolute/filter_best.pt \
  --keep-threshold 0.8 --frame world --plot \
  --output /tmp/entry_real_dual_scores
```

可指定 --expected-filter-sha256。snapshot 需要 context_xyz.npy、query_xyz.npy、stable_cost.npy。Filter 用自身 checkpoint 的编码器和完整上下文打分，再通过精确 XYZ 索引恢复 ROI 查询点序。这是独立离线模型对照，不声称已有单编码器在线双头。缺少权重时直接报错，不自动生成模拟分数。
