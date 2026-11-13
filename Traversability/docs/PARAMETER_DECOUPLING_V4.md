# 四参数解耦与局部几何阈值增强训练

从 `runs/cost_all_stairs_dr_v3/best.pt` 开始，保持 PTv3、FiLM、CostHead 的结构与全部四维参数。教师仍使用原生VST、p=2、障碍代价50、归一化0–1，interval_free固定0.5m。源权重保持不变。

## 数据

输出 `datasets/wfc_parameter_decoupling_v4/`。直梯、折返梯、旋梯、连续斜坡各36场景，分别24训练、6验证、6测试，总计96/24/24。另重标注4个旧训练建筑和1个旧验证建筑。每个新seed包含场景PCD、实际几何与overview.png。姿态、镜像、平移先作用于几何，再计算教师，不在线旋转已标注的几何。

四参数范围为 clearance_min=[0.2,1.0]、slope_max=[0.18,0.70]、step_max=[0.08,0.55]、stand_ratio=[0.02,0.50]。

每场景两套条件库，每套40组：原4固定组加四参数各9值。第一套固定另外三参数为基准，第二套使用独立随机背景参数。坡度/阶高根据原生局部梯度选临界值，站立比例依据整数站立格数选跳变点。每组9值的36个条件对全部监督。无范围内临界值时等距采样，保留教师稳定的条件组。

完整监督保存在原生查询XYZ、cost.npy和conditions.npy中。每场景另存原生基准vst_relaxed.pickle与参数清单。设施ROI仅用于临界值采样、区域加权和报告，不删除标签。

## A/B与训练

编码器BF16，FiLM和CostHead强制FP32，训练/推理共用同一解码函数。原有配置不设置decoder_precision时保持原行为。

| 版本 | 绝对损失 | 条件λ | 四参数权重 |
|---|---|---:|---|
| A | 全点平均 | 1 | 1/2/1/2 |
| B | 设施/背景实际存在区域等权 | 2 | 1/1/1/1 |

SmoothL1 beta=0.1，变化阈值0.02，变化/稳定区等权。每轮480块，其中新设施336块、每类84块，旧回放144块。真batch2，每轮240次更新，前2轮冻结编码器。头学习率5e-5、编码器5e-6，AdamW weight_decay=0.05，余弦总周期始终30轮。

A/B同初始化、同种子、同采样顺序，各跑5轮。按第5轮验证分数选一个版本，从其last.pt接到30轮。两版分数差小于1e-6时比较旧验证MAE，再相同选A。

## 固定选优

两套条件库均完整评估。每场景先汇总，再对场景等权。每个参数的响应误差是变化区差值MAE和稳定区差值MAE的平均，有实际样本的分区参与统计。

`S = 0.7 × (新场景MAE + 四参数平均响应误差) + 0.3 × (旧验证MAE + 四参数平均响应误差)`。

最终best.pt在初始化和获胜版本的所有轮次中选择，last.pt保存第30轮。若初始化仍最佳，报告明确标记没有改善。测试集不参与选优。

## 执行与输出

从工作区根目录运行：

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Traversability.experiments.parameter_decoupling
```

该入口使用glocal2-nav/.venv生成几何和原生GPU教师，使用pure3d-ptv3训练。pipeline.pid和execution.json记录活动进程。已完成阶段有完成记录，中断后再次执行复用数据并从本次last.pt恢复。运行中的同一流水线不重复启动。

结果统一位于 `runs/traversability/cost_parameter_decoupling_v4/`：

- progress.json：数据进度、当前版本、轮次、分项loss及四参数验证历史，由流水线本身更新，不创建定时任务。
- A/、B/：同预算短跑和获胜版续训。best.pt、last.pt是统一符号链接入口。
- plots/：24个独立seed的GT/Before/After、四参数响应、R4阶高与坡度回归。
- all_stairs_audit/：已知大图8处楼梯的固定回归。
- predictions/straight、switchback、spiral、ramp、large_terrain：原始输入PCD全部点的prediction.pcd。
- summary.json、REPORT.md：独立测试、旧建筑保留能力及预先规定验收目标的实测结果。

新验证平均响应误差降低15%、四参数变化误差分别改善、旧MAE增幅不超过5%为验收目标。R4阶高0.30→0.40的幅度比目标0.7–1.3且方向≥90%，坡度0.18→0.70稳定踏面漂移目标≤0.02。验收结果不触发教师或超参数的自动更改。
