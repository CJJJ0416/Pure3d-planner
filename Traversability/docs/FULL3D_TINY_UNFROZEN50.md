# 完整3D模型：Tiny不冻结的同起点50轮训练

对照运行是`cost_full3d_tiny_rotating100_20261113`实际完成的50轮。
本轮仍从切片版第90轮best.pt初始化，复现同一批20个seed、同一标签配方和训练预算。
改变Tiny的训练策略，保留现有网络层、分块、GPU对应和损失。训练态启用Tiny原有随机层，验证使用eval。

## 数据与训练

- seed：2026111400–2026111419，每10轮替换4栋建筑。
- 原始PCD和场景配置按旧归档重建，逐栋核对PCD SHA256、Tomogram XYZ SHA256、条件表、条件库和配对。
- 完整PCD按5 cm体素保留真实点，核心4×4×3 m，上下文1 m。
- Tiny的scale=0.5、grid_size=0.01、486维特征和模型参数量保持。
- 仅保存几何、标签与映射。训练不读取features.npy，不跨步缓存Tiny特征。
- 两块分别前向编码，合并查询特征和标签后计算同一个batch损失、执行一次反向和优化器更新。
- 真batch2。Tiny编码BF16，查询特征沿用FP16接口，任务头与损失FP32。特征索引和精度转换保留梯度。
- Tiny从第1步起参与优化。新建AdamW，Tiny峰值LR=1e-5、任务头1e-4，最低分别1e-7和1e-6，weight decay=0.05。
- 沿用原100轮学习率阶段，前5%预热，第50轮停止。调度按每轮实际块数的轮内进度计算。
- 损失仍为绝对损失+2倍条件差值损失。SmoothL1 beta=0.1，变化阈值0.02，设施/背景、变化/稳定、四参数均衡。
- 教师仍为p=2、cost_barrier=50、interval_free=0.5，0–1标签。两套条件库逐轮交替，不增加在线旋转或监督筛选。

## 入口

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws HF_HUB_OFFLINE=1 TZ=Asia/Shanghai \
MPLCONFIGDIR=/tmp/pure3d_unfrozen_matplotlib \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -u \
  -m Traversability.experiments.full3d_unfrozen \
  --config /home/cjh/Pure3D_ws/Traversability/configs/full3d_tiny_unfrozen50.json
```

流水线第1轮结束后真实退出，再从本轮last.pt恢复第2–50轮。重启前核对execution.json中的PID，避免重复运行。
单独训练入口为`Traversability.training.full3d_finetune`，支持`--resume`和`--stop-after-epoch`。
恢复包括实际学到的Tiny、任务头、AdamW、学习率阶段、随机状态和数据批次。

数据位于`/home/cjh/Pure3D_ws/datasets/cost_full3d_tiny_unfrozen50_20261113/`。
运行位于`/home/cjh/Pure3D_ws/runs/traversability/cost_full3d_tiny_unfrozen50_20261113/`。
只保留当前4栋训练建筑的大文件，上一批Mesh、PCD、标签、映射和WFC缓存在切换时删除。
20栋建筑的参数、overview、条件与核验记录永久保留。原冻结版运行及其数据保持。

## 验证与测试

初始化验证与冻结版同起点核对，最大预测差异要求小于1e-5。
首个真实更新记录Tiny及三个任务头的梯度与参数变化，记录位置first_update.json。
每轮在线验证原固定建筑，两套条件库共享当轮同窗口编码结果。
选优分数仍为归一化Cost MAE+四参数平均响应误差，初始化参与选优。

第50轮结束后评估固定测试建筑，独立输入完整PCD复跑，再预测building_SUST。
SUST只作最终回归，不参与训练与选优。SUST比较统一XYZ、参数和VST规则参考。
冻结版会在同次测试中重新推理，用于记录单次完整PCD耗时，旧QueryPool结果引用已保存的同XYZ直接查询。
旧QueryPool还会在固定测试建筑的相同Tomogram XYZ及两套80条件槽上独立评估，记录四参数响应、绝对误差和阈值错误率。
该参考评估使用旧模型自身的分块及局部/全局近邻，不改变当前完整3D模型的训练或推理。

best.pt、last.pt保持独立encoder与model字段，encoder保存当时训练后的真实权重。
现有Traversability.inference.full3d入口可以直接加载，原切片兼容入口也保持。
progress.json展示50轮进度、loss、验证指标、梯度、双学习率、显存和临时磁盘占用。

最终REPORT.md给出冻结/不冻结的固定测试、SUST、四参数响应、误阻断与错误放行率，以及GT/预测/误差图。
此次只训练Cost链，不连接Filter/Recon，不修改规划器，不创建定时任务。
