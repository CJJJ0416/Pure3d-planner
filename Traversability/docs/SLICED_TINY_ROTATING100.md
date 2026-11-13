# 冻结Tiny，轮换场景续训Cost至100轮

本机执行，配置为 `Traversability/configs/sliced_tiny_rotating100.json`。
从 `runs/traversability/cost_sliced_tiny_local50_20261113/last.pt` 第50轮继续至第100轮。
原第30轮best作为选优候选，源运行及其权重保持不变。

## 模型与监督

完整建筑PCD → 原生几何切片与高度合并 → 4×4×3m核心块、1m上下文 → 冻结官方Utonia Tiny → 486维缓存 → Adapter → FiLM → CostHead。
只更新200001个任务头参数，Tiny全程eval，编码BF16、缓存float16、任务头和损失FP32。
所有机器人条件共享同一份几何、XYZ顺序和特征。输出为切片支撑XYZ及0–1代价。

教师保持原生规则，p=2、cost_barrier=50、interval_free=0.5。学生参数为clearance_min、slope_max、step_max、stand_ratio。
每场景两套40槽条件库，固定基准及随机背景分别按四参数扫描。每轴9值、36对。
损失为SmoothL1(beta=0.1)的绝对项加2倍条件差值项。设施与背景、变化与稳定区等权，变化阈值0.02，四参数等权。

## 轮换与存储

第51–55轮使用seed 2026111300–2026111303，随后每5轮换4个seed，最后第96–100轮使用2026111336–2026111339。
沿用WFC域随机配置和完整Mesh 3cm表面采样。每轮遍历当前四建筑的全部训练块，真实batch2，条件库按累计轮次交替。
几何生成在现有pure3d-ptv3环境中执行，原生GPU VST使用glocal2-nav/.venv。WFC小型依赖放在工作区datasets/cache/wfc_runtime，不更换Conda环境。

当前批的Mesh、PCD、标签、分块、特征及WFC几何缓存放在新数据根目录active内，下一批前删除。
验证与测试通过符号链接引用旧数据，标签和特征不重算、不复制。
每个seed的实际配置、overview、条件、数据摘要、耗时和日志保存在新运行的scene_archive中，40个场景仅累积轻量记录。
CuPy/Triton编译缓存与小型Python依赖复用，不属于按场景增长的几何缓存。

## 续训与恢复

保留源last.pt的模型、AdamW状态和累计步数。新增50轮采用独立学习率阶段：前5%从1e-6升到1e-4，随后余弦降至1e-6。
学习率按累计轮内进度计算，适应各批不同块数。每轮保存last.pt，包括优化器、调度进度、随机状态、已完成轮次和数据批次。
恢复只重复未完成的轮次，使用协议中的确定seed重建缺失临时数据。已完成的旧训练池不再保留大文件。

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws HF_HUB_OFFLINE=1 TZ=Asia/Shanghai \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -u \
  -m Traversability.experiments.sliced_rotating \
  --config Traversability/configs/sliced_tiny_rotating100.json
```

同一入口可继续已有运行。首次在第51轮末实际保存并退出，随后新进程从last.pt恢复，核验优化器和轮次衔接。
恢复前确认原流水线已退出，避免并行写同一运行目录。

## 评估与交付

每轮完整验证原固定建筑两套条件库，沿用Cost MAE加四参数平均响应误差选优。测试只在最终best选定后执行。
progress.json包含当前阶段、累计轮次、数据批次、loss、固定验证MAE、参数响应和临时磁盘占用。
第100轮完成后生成完整测试PCD、独立PCD输入推理、GT/原最佳/新最佳/误差图、条件响应曲线及REPORT.md。
training_verification.json记录Tiny不变、累计100轮、优化器步数，storage_verification.json记录9次替换和最终仅保留4场景。
若原第30轮仍最佳，报告明确保留原模型。本轮不接Recon，不修改规划器，不启动服务器或定时任务。
