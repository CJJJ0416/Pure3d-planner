# 完整3D输入、冻结Tiny与轮换场景训练

本轮从`cost_sliced_tiny_rotating100_20261113/best.pt`第90轮初始化任务头，新建优化器，训练100轮。

执行调整：用户随后要求在第50轮停止并测试。当前配置`stop_epoch=50`，原`epochs=100`仅保留学习率调度预算。
本次实际使用前5批、20栋训练建筑。运行目录沿用原名称，最终轮次以summary.json和progress.json为准。

## 数据与特征

完整建筑PCD在PyTorch CUDA上按5 cm世界系体素分组，每格保留原始编号最小的真实点。
不取质心，保留地面、台阶立面、墙体和天花板。核心块4×4×3 m，外扩1 m上下文。
Tiny使用官方utonia_tiny，scale=0.5、grid_size=0.01，486维上采样特征，始终eval并冻结。
5 cm是输入几何采样密度，Tiny内部预处理仍使用原配置，对应物理2 cm网格。

Tomogram仅提供原生支撑XYZ、设施标签与各机器人条件的cost，查询点不插入Tiny输入。
Triton开放寻址体素哈希从27邻域开始逐圈搜索，用未访问体素距离下界确定精确最近代表点。
同距按原始点编号选择。不建立CPU KD-tree，不使用可靠性筛选，不删除原生监督点。

映射顺序为Tomogram编号→5 cm代表点→原始点编号→块内Tiny体素→缓存特征行。
缓存只保存查询引用的FP16特征，及inverse、匹配距离、来源坐标索引。上下文参与编码，核心查询参与损失。
不同机器人条件共享XYZ、点序、对应索引和冻结特征。缓存位于每场景features目录。

## 固定配置

- Tiny编码BF16，Adapter、FiLM、CostHead与损失FP32。任务头结构、486维输入和四参数接口不变。
- 四参数顺序为clearance_min、slope_max、step_max、stand_ratio。
- 原生教师p=2、障碍代价50、interval_free=0.5，标签归一化为0–1。
- 两套40槽条件库逐轮交替。SmoothL1 beta=0.1，绝对损失+2倍条件差值损失，变化阈值0.02。
- 设施/背景等权，变化/稳定等权，四参数等权。
- 真batch2，每轮完整遍历四栋建筑的全部核心块，末批不足两个时沿用原有重复首块方式补齐。
- 新建AdamW，weight decay=0.05，前5%从1e-6预热至1e-4，再余弦下降至1e-6。按轮内实际进度调度。

## 场景轮换及存储

每10轮整体替换四栋训练建筑，共10批40栋。seed为2026111400–2026111439，沿用WFC域随机配置。
保留每个场景的实际配置、overview、条件、摘要及耗时。批次替换前关闭旧DataLoader，删除上批Mesh、PCD、标签、特征和WFC临时缓存。
验证scene_20261113、测试scene_20261114沿用原XYZ、标签、条件、核心划分，通过只读引用保存。
完整3D特征重新生成，位于本轮独立fixed目录，原有切片缓存不写入、不复用。

配置：`/home/cjh/Pure3D_ws/Traversability/configs/full3d_tiny_rotating100.json`。
运行：`/home/cjh/Pure3D_ws/runs/traversability/cost_full3d_tiny_rotating100_20261113/`。
数据：`/home/cjh/Pure3D_ws/datasets/cost_full3d_tiny_rotating100_20261113/`。

## 启动与恢复

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws HF_HUB_OFFLINE=1 TZ=Asia/Shanghai \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -u \
  -m Traversability.experiments.full3d_rotating \
  --config /home/cjh/Pure3D_ws/Traversability/configs/full3d_tiny_rotating100.json
```

流水线先完成第1轮，真实退出后从本次last.pt恢复第2–100轮，再执行独立PCD推理和报告。
重新运行入口前核对execution.json中的父、子PID，确认旧训练进程已经退出。
恢复保留模型、AdamW、学习率阶段、随机状态、累计步数和当前数据批次。缺失临时场景按保存seed重建。

每轮按固定验证的归一化Cost MAE+四参数平均响应误差选优。完整3D初始化参与选优，旧切片结果仅作参照。
仅保存best.pt、last.pt，完整模型中包含编码器及推理元数据。progress.json记录阶段、轮次、loss、验证指标和占用。

## 推理与结果

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws HF_HUB_OFFLINE=1 \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Traversability.inference.full3d \
  --checkpoint /home/cjh/Pure3D_ws/runs/traversability/cost_full3d_tiny_rotating100_20261113/best.pt \
  --pcd /absolute/path/building.pcd --condition 0.5 0.6 0.4 0.05 \
  --output /home/cjh/Pure3D_ws/runs/full3d_prediction
```

推理只输入PCD与机器人条件。原生几何处理生成支撑查询XYZ，不计算教师cost。Tiny读取完整3D上下文。
旧Traversability.inference.sliced入口按checkpoint架构分发，新旧权重均保留各自的数据路径。

REPORT.md、summary.json给出GT对比、四参数响应及旧第90轮模型参照。
training_verification.json、resume_verification.json核对冻结与恢复。
固定建筑full3d_encoding.json和matching_audit.csv记录精确最近邻核验、来源及距离。
timing_summary.json分别统计读取、体素、分块、对应、编码、缓存及训练耗时。
独立推理必须与同XYZ缓存评估一致。此次本机执行，不修改规划器，不创建定时任务。
