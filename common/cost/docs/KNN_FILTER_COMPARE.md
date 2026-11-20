# KNN 编码前孤立点过滤对照

本文中的对照实验命令依赖本地保留的实验模块，该模块不随源码发布。正式在线入口见 [Entry](../../README.md)。
该对照入口只处理保存点云，不启动 Gazebo。已接受的保守最小统计规则也已接入在线局部 cost，在线配置见 [LOCAL_COST.md](LOCAL_COST.md)。

输入来自上次上下文实验保留的 `outputs/entry_local/context_filter_20261116/{complete_context_snapshot,recorded_spike_frame}/context_xyz.npy`，转换为同序二进制 PCD 后读取。主样本为 71062 点完整 LIO 上下文，原 ROI 的 34462 点作为查询。另测 24187 点异常帧 ROI。没有添加合成点。

## 固定过滤规则

- 标准 SOR：计算每点除自身以外最近 16 个点的平均距离。超过全体均值加 2 倍总体标准差时删除。
- 保守版：同时满足 SOR 外点和 0.15 m 范围内其他邻居少于 3 个才删除。
- 完整诊断还计算第 16/64 近邻距离、0.15/0.5 m 邻居数量和平均距离与远邻距离之比。
- 最小统计版只计算保守规则实际需要的 16 近邻均值和 0.15 m 邻居计数。它必须与完整保守版得到逐点一致的掩码和 cost。

cost 和网格真值均不参与过滤阈值或删除掩码。不会因某点 cost 较高而删除它，也不会用 mesh 标签恢复被误删的点。

## 运行

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws/utils/PointTransformerV3:/home/cjh/Pure3D_ws  MPLCONFIGDIR=/tmp/entry-knn-matplotlib  /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m common.experiments.knn_filter_compare  --output /tmp/entry_knn_compare --workers 4 --repeats 5
```

使用相同 Tiny best49 checkpoint 和条件，core 4×4×3 m、halo 1 m 不变。每条件预热 1 次、测 5 次，轮换测试顺序，强制完整重编码。记录索引、KNN、各半径计数、掩码、编码、head、完整推理以及读 PCD 到 cost 返回的墙钟耗时。完整推理包含快照准备、NPY传输、GPU预处理/编码/head、结果读取，模型初始化和落盘/显示另计。

删除的查询点不补回。原始与过滤后 cost 只比较共同保留的同一批 XYZ，另保存所有删除点，避免把删除一个高代价点直接算作预测改善。

## 楼梯误删审计

使用原 Gazebo 碰撞 STL，按三角形无序顶点双向匹配验证另存的场景 OBJ 和楼梯组件。重复三角形中心会消歧，不仅比较中心位置。保留原 world 坐标，不做 ICP。

报告距静态楼梯表面 3 cm 和 5 cm 内被删点数、向上水平面与竖直面统计、每个踏面/平台高度组的观测保留情况。近表面标记是容差内的几何代理，不能证明所有远离场景的点都是幽灵。没有扫描到的楼梯部分不在审计范围。

输出包含 `removed.pcd`、`removed_stair_near_3cm.pcd`、`audit_arrays.npz`、`stair_audit.json`、逐点 cost 的 `paired.npz` 和分阶段计时。实验中参考网格只负责审计，不影响保留/删除决策。

## RViz

```bash
source Entrance/shell/_internal/sim_env.bash
ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1  python3 -m common.experiments.knn_filter_compare --view-only --output /tmp/entry_knn_compare  --dataset complete_context_snapshot --method conservative
```

切换 Before / After 看同一批保留点的 cost，使用共同显示行、world 和固定 0–1 绿红色标。粉色是删除的上下文点，黄色是距静态楼梯表面 3 cm 内的删除点。`--method sor` 查看标准 SOR，`--dataset recorded_spike_frame` 查看异常帧和稀疏上层平台。
