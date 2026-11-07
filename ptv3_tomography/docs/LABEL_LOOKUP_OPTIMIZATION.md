# 已知 Tomography 的直接标签查表

2026-11-05。单帧 Plan A 的监督契约保持 mid360_lidar_reference_v2：雷达完整姿态、6×64×64 格子、重力相对高度、独立 cost/height 掩码与未知支撑标记。

## 查询方式

pair_mid360_lidar_reference.py 每条采集序列创建一次 LidarLabelIndex，缓存场景有效格子。每帧先把局部 XY 与高度范围逆映射为保守世界坐标包围盒，在这个局部范围读取原图已有的地面高度。按高度分组，仅投影每组包围盒的四个角确定查询格子，再把这些局部查询坐标转到世界坐标，直接索引原生 Tomography。

查询保留所有相关原生层。同一高度的重复层按最低 cost、最低层号处理，不同高度落到同一目标格时仍使用原有歧义规则。cost=0 有效，cost=60 保留，未匹配或歧义位置保持原掩码。中间查询每次处理至多 32 个高度组，避免把所有高度和格子同时展开。

这里采用监督专用的高度匹配。原生规划器 DecideLayer 和导航表面的 decide_layer 带可通行筛选及 0.2 m 高度分支选择，直接调用会改变监督标签，因此没有修改原生规划器或将其通行阈值应用到监督中。

## 本地验证

同一 40 帧、每帧 1000 条射线、同一 CPU/NumPy 环境，顺序执行，无 profiler。完整标签准备包含读取、查询、观测掩码、NPZ 压缩、SHA256、逐步与有序加载器校验。

| 项目 | 原版 | 直接查表 |
| --- | --- | --- |
| 标签准备 | 24.85 s | 3.10 s |
| 标签准备吞吐 | 1.61 帧/s | 12.92 帧/s |
| 加上 portable 导出 | 25.01 s | 3.27 s |

标签准备加速 8.03 倍，含 portable 为 7.65 倍。40 个标签 NPZ 的 SHA256 全部相同，即压缩文件字节也相同。另对照 71 个用例，覆盖九区域、真实 PhysX 返回、平台最后一帧、不同倾角、奇数/非正方形网格、不同分辨率、重复层、不同地面、cost 0/60、空地图与越界，所有标签数组一致。近水平雷达仍按原规则拒绝生成标签。46 个上游 PTv3 文件哈希保持不变。

本地结果在仓库 outputs/local_label_query_20261105，包括 benchmark_summary.json、parity_report.json、冻结原版代码、重现脚本、原始标签和平台边界对照图。此目录由 Git 忽略。这里是小规模本地 CPU 验证，服务器及完整 288 组的吞吐尚未实测。

## 新采集短 epoch 验收

另一次完整试跑使用同一九区域场景，新采集 ramp_01 和 open_stairs_01 各 20 帧，共 40 帧。完整 PTv3 与地图头训练 1 epoch、10 次参数更新，checkpoint 保存和加载预测通过。场景/VST 资产复用，点云重新采集。

进程墙钟共 25.74 秒：采集 6.64 秒，标签准备 2.75 秒，portable 导出 0.30 秒，训练进程 12.45 秒，checkpoint 加载和三帧预测验收 3.60 秒。采集循环为 0.21 秒，训练两份相同 checkpoint 的序列化及 fsync 合计 3.17 秒。原始证据位于 outputs/short_epoch_direct_lookup_20261105。该试跑是当前版本完整链路验收，上表的加速比来自单独的同数据标签准备对照。

## 使用与已有服务器数据

原配对 CLI 参数不变，现有采集数据可复用。下面 SCENE_DIR、CAPTURE_DIR 指向已经采集的资产。PAIR_OUTPUT、PORTABLE_OUTPUT 必须选择尚不存在的新目录。完成的 portable 数据保持原样，待处理 capture 使用新的配对入口，不需要重采集。

```bash
"$TRAIN_PY" "$PTV3_PROJECT/scripts/data/pair_mid360_lidar_reference.py" \
  --scene "$SCENE_DIR" --capture "$CAPTURE_DIR" --output "$PAIR_OUTPUT"

"$TRAIN_PY" "$PTV3_PROJECT/scripts/data/make_dataset_portable.py" \
  --dataset "$PAIR_OUTPUT" --output "$PORTABLE_OUTPUT"
```

pairing_report.json 新增 timing，分别记录 query_s、masks_s、compressed_save_s、total_s 和 frames_per_s。旧协议的源码指纹会检测到代码改变，不应覆盖 protocol 或删除来源校验来强行恢复整个 campaign。服务器应复用既有 288 组采集资产，只继续数据准备阶段。本文没有增加路线多进程调度。

## 标签可视化

新入口 scripts/visualization/render_label_lookup_comparison.py 复用已有 RGB 和相机，输出 cost/height 标签与逐格差异图。实际运行在已有 matplotlib、Pillow、NumPy 的绘图环境，不要求启动 Isaac Sim。

```bash
"$PLOT_PY" "$PTV3_PROJECT/scripts/visualization/render_label_lookup_comparison.py" \
  --reference-labels "$REFERENCE_LABEL_NPZ" --direct-labels "$DIRECT_LABEL_NPZ" \
  --rgb-reference "$SAVED_RGB_CAMERA_JSON" --benchmark "$BENCHMARK_SUMMARY_JSON" \
  --output "$NEW_COMPARISON_PNG"
```
