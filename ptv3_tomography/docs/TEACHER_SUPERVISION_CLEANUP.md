# 原生 VST 监督与旧代码清理

2026-11-06。用户确认：网络学习 VST 教师规则，去掉额外的高度连续、地表回波匹配和邻域覆盖筛选。后续不新增防御性编程，根据实际测试问题再修改。

## 当前路径

`vst.pickle` 各层 cost、ground → 完整位姿转换 → 局部 GT → 标签可用性 mask → PTv3/地图头 SmoothL1。

```python
mask_cost = target_available.copy()
mask_height = target_available.copy()
```

未观测但有教师标签的位置也参与监督。未增加新模型、代价修正或导航规则。原生 VST 的层简化与几何含义沿用现有实现。

## 清理

- 删除高度连续、表面命中匹配、射线自由空间采样、邻域覆盖统计及其配置阈值。
- 删除旧 body/yaw-only v1 配对、历史窗口数据加载器和未使用的 SRU 序列包装。
- 删除旧 v1 单帧累积训练器与 A100/RTX4060 旧串行配置，当前只保留真实 batch 训练。
- 删除 Isaac 目录重复的标签模块和旧验证/资源限制脚本，正式配对统一使用当前工程入口。
- 移除任务层文件哈希拒绝、配置/坐标/形状预检查、严格源码契约比较。文件格式中保留来源元数据。
- 清除逐点可视化中重复前向和整套断言，特征恢复调用 `restore_point_features(features, inverse)`。
- 调度器完成任务后按状态复用结果，不重复遍历并哈希所有数据文件。

未知标签的掩码、坐标映射、插值权重、batch 边界以及优化器和断点保存属于当前计算流程。官方 PTv3 和 VST 第三方源代码未改动。

## 本地执行

结果位于 `runs/teacher_supervision_cleanup_20261106/`。`cleanup_manifest.json` 记录修改与删除路径，旧源码保存在该目录的 `source_before/`。这些本地产物均被 Git 忽略。

侧坡平台 seed 20261116，81 帧真实 Warp 扫描，使用清理后的正式入口重新配对：约 1.70 秒，330,466 个累计帧格参与 cost 和 height，其中 cost≥50 为 35,255，cost=60 为 24,510。新旧 cost、高度及 target_available 差异均为 0。

本地 RTX4060 使用完整模型、真实 batch=16、workers=2，先运行一轮，再由 last.pt 恢复下一轮。已完成 2 个 epoch、8 次参数更新，训练 Loss 为 0.4029 → 0.2812。last.pt 的完整状态恢复成功，best.pt 与 last.pt 均已保存。另完成 3 帧地图导出和 773 个原始有效点的特征恢复。运行记录与 Loss 位于 `train/logs/`，汇总见本地 `report.json`。64 帧 train 和 17 帧 dev 来自同一个 seed，仅用于检查执行链，不作为未见场景评估。

历史三重掩码审查保存在 [SUPERVISION_AUDIT.md](SUPERVISION_AUDIT.md)，其中“正式掩码未修改”描述的是此次切换之前的实验。


## Isaac Sim 原生显示

复用同一份采集场景 terrain.obj。查看器支持已有 USD stage，也可直接加载捕获场景网格。已在本机原生窗口展示第 40 帧的 3,906 个 GT 网格并保存原生截图。界面可切换 cost、重力高度、GT、PTv3 预测、误差和高度带。

先用 `render_plan_a_comparison.py --maps <地图导出目录>` 生成对比面板，再执行：

```bash
python src/ptv3_tomography/scripts/visualization/view_plan_a_isaac.py \
  --maps <地图导出目录> \
  --output runs/<实验名>/native_isaac \
  --frame 40 --mode 1
```

使用本机 Isaac Sim Python 环境。mode=1 为 GT cost，mode=0 为 PTv3 cost。未指定 stage 时采用数据集记录的场景网格。原生截图与 USD 保存在 output 目录，运行结果不纳入 Git。
