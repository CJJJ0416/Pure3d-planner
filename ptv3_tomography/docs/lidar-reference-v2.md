# 当前雷达参考接口与小内存检查

> 历史记录。2026-11-06 已切换为原生 VST 标签可用性监督，文中的三重筛选、旧验证与旧 v1 入口已删除。当前代码见 [原生监督清理](TEACHER_SUPERVISION_CLEANUP.md)。

2026-11-04。D1-R0 的 v2 配对和数值接口已实现；原生检查发生用户报告的内存耗尽，报告未完整写入，整体任务保留未确认。当前默认只运行 CPU 小样本，不自动启动 Isaac Sim 或训练。

## v2 数据契约

数据：`outputs/mid360_lidar_reference_v2_20261104`，复用原 130 帧 PhysX 扫描，保留 v1 数据与代码。v2 中的每个感知样本只读取当前帧；不沿用旧 K=8 观测掩码，也不添加历史 Attention。

雷达 odom 是 `T_O_S_t`，描述雷达自身在统一 O 系下的完整位姿，不再额外乘机身外参。当前仿真 O=W，`lidar_odometry.npz` 显式保存这一约定。

```text
当前雷达原始点 P_t^S                       → PTv3 单帧输入
inv(T_O_S_t) @ T_O_S_prev                  → previous_sensor_in_current
t_t - t_prev                              → dt_s
R_O_S_t.T @ up_O                          → gravity_up_sensor
```

loader 的模型输入只有 `points_sensor_m`、`point_mask`、`previous_sensor_in_current`、`dt_s`、`gravity_up_sensor` 和 `episode_start`；绝对世界位姿和教师地图只供离线配对与诊断。

输出点完整表达在原始雷达坐标 S 中。地图参数轴明确采用 `[重力高度带, sensor_x, sensor_y]`：雷达 XY 格心间距 0.1 m，高度为 `h=up_S·p_S=ground_W_z-lidar_W_z`。因为雷达 pitch=45°，`p_S.z` 不能直接当成物理高度。这也不是等间隔的正交 sensor-XYZ 体素。

辅助解码的固定空间 query 可由已知格心 `(x_S,y_S)`、高度带中心 `h_b` 和重力向量生成：`z_S=(h_b-up_S.x*x_S-up_S.y*y_S)/up_S.z`，不需要教师高度。标签另保存实际 `surface_points_sensor_m`。

配对时把原生 VST 的每个栅格看作常高水平小面，求其与雷达 Z 方向格心柱的交点，并检查交点确实查询回同一原生单元。相同物理高度的重复层合并；同带异高则排除。`abs(sensor_z·up)` 太小时明确拒绝生成标签，防止近水平雷达 Z 方向下发生退化。

高度和代价掩码独立；只允许当前帧有效命中及命中前的自由射线提供证据。代价覆盖仍是局部可恢复性的工程代理。支撑全部 `unknown(-1)`、`mask_support=0`，支撑损失关闭。

`LidarStepDataset` 提供逐帧接口。`LidarOrderedSequenceDataset` 提供 17 个非重叠有序块，每块最多8步；实际130帧恰好读取一次，末尾6步为 padding，`step_mask=False` 必须跳过状态更新和损失。仅回合首帧重置；块边界 detach 状态而不重置。该有序读取契约已核验，实际 SRU 网络执行仍属于 P1。

## 当前强制的小检查预算

配置：perception_small_check.json（历史验收入口已归档）。

| 项目 | 当前设置 |
| --- | --- |
| 样本 | 起点/坡中/平台共3个独立帧 |
| 点数 | 每帧最多256，缺少时padding并配有效位 |
| 地图 | 中心32×32，6个重力高度带，0.1m分辨率 |
| 批量与并行 | batch=1，CPU逐帧，workers=0，单计算线程 |
| 系统内存 | 独立systemd scope，MemoryMax=2G，SwapMax=0 |
| 原生渲染 | 不自动启动Isaac，不同时运行渲染与训练 |
| P1-T1训练验收 | batch1、256点、32×32、5步更新及另重放2步，6GiB硬限制、swap0，已通过 |
| 下一步SRU预算 | 展开2步、单环境，尚未执行 |

减少输入点和地图范围后，脚本会重新生成监督掩码，不把完整扫描的掩码直接套在稀疏输入上。三个诊断帧互相独立，不把它们作为连续 SRU 序列递推。

默认入口带强制内存限制；若不能创建受限scope，停止而不退回无上限执行：

```bash
conda activate env_isaaclab
python scripts/validation/run_lidar_small_check.py \
  --dataset outputs/mid360_lidar_reference_v2_20261104 \
  --output outputs/mid360_lidar_reference_v2_20261104/validation/my_small_check.json
```

每次使用新报告文件。不要直接重跑完整原生截图脚本；已生成的截图可以直接查看，后续原生低内存验收另行按单帧、低分辨率、独立进程开展。

## 现有证据

- `pairing_report.json`：130帧，16,513个高度、13,992个代价监督单元；代价≥10为5,883，cost60在当前单帧保守掩码内为0；不为增加负例而放宽掩码。
- `validation/alignment_report.json`：此前全部130帧数值检查通过；原始网格最大高度误差2.36cm，完整雷达点坐标回变换最大残差2.38e-7m，重力高度残差4.77e-7m。
- `validation/small_memory_check.json` 及 `small_memory_guarded_check.json`：3帧小内存检查，256点与32×32标签重新构造掩码，记录实际 RSS 与硬限制；不包含PTv3训练或原生渲染验收。
- `validation/native/`：6张截图已生成，但 `report.json` 为未完整写入文件；不得据此宣称该运行完整通过。资源故障另记录在 `resource_interruption.json`。

已完成P1-T1微型官方PTv3监督训练与恢复（历史验收入口已归档），采用本v2数值接口并重新计算减点/裁图后的掩码。CPU训练全程6GiB硬限制，5步更新及独立进程恢复验证通过。SRU接入和原生资源验收仍未完成，不同时扩大样本数、地图尺寸和时序展开。
