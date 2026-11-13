> 历史实验设计参考。旧运行结果与旧权重已按要求清理。当前操作入口见 [完整训练流程](TRAINING_WORKFLOW.md)，当前保留权重见 `runs/traversability/model.json`。

# 条件响应对照训练

入口和全部结果保存在 `runs/cost_response_ablation_20261110/`。旧数据和旧权重保留。

四版均从 `runs/cost_stand_lambda_20261110/best.pt` 初始化，重置优化器。PTv3、QueryPool、FiLM、CostHead 结构相同。每版训练12轮，每轮随机访问342个建筑块，batch=2，共2052次更新。头学习率3e-4，编码器3e-5，余弦衰减，前2轮冻结编码器。不同版本的条件数和单步计算量不同，因此这是相同更新预算的比较。

| 版本 | 损失 | 训练几何 | 学生条件/场景 |
|---|---|---|---:|
| A_current | 原绝对SmoothL1 + 相对nominal差异，lambda=1 | 原4个建筑 | 24 |
| B_balanced | 绝对SmoothL1 + 单参数变化/稳定区域均衡差异 | 原4个建筑 | 24 |
| C_paired | 同B | 原4个建筑 | 40 |
| D_geometry | 同B | 原4个 + 新2个建筑 | 40 |

B/C/D在教师归一化代价差绝对值大于0.02时归为变化区域，其余归为基本稳定区域。这只影响差异损失的归约权重，不删除任何标签。每个参数内部两区域分别求均值，各占50%。一个batch缺少某区域时只对实际存在的区域归约。四参数再按[1,2,1,2]加权平均，lambda=1。变化方向使用教师实际差值，未施加手工单调规则。

C/D沿用14个固定条件和10个随机联合条件，再选前4个随机条件为基准，每个基准生成4个单参数变化，新增16组。变化值在该参数归一化范围上平移半个范围并循环回范围内。其他参数逐值保持不变，interval_free固定0.5m。几何和教师XYZ不作为学生条件输入，教师cost只参与损失。

D新增seed20261130和20261131。参考站立比例改为0.25用于几何生成，坡度/踏步/支撑特征每档4例，其余每档2例，踏步深度范围0.12–0.48m。教师的实际机器人参数仍由robots_response.json逐组覆盖。每个新场景保存overview.png。

## 验证与测试

所有版本统一在原数据集 `datasets/wfc_capability_dr_stand_v3` 的固定val建筑及24组条件上验证。每个epoch计算全部只改变一个参数的条件对，统计变化点差值MAE、响应幅度、方向正确率、稳定点差值MAE、整体Cost MAE和危险低估率。

best.pt选择分数为归一化Cost MAE，加上四参数加权平均的“变化点差值MAE + 0.25 × 稳定点差值MAE”。参数权重为[1,2,1,2]。四版采用相同分数。版本赢家在运行test之前写入validation_selection.json。test建筑不用于挑选checkpoint或版本。

原四参数模型同样在相同数据上重新评估。固定test包括seed20261114和20261127。此前已经分析过这些测试建筑，它们是固定回归测试集，不等同于完全未经查看的新场景泛化证据。

## 执行

```bash
PYTHONPATH="$PWD" /home/cjh/glocal2-nav/.venv/bin/python runs/cost_response_ablation_20261110/prepare_data.py
PYTHONPATH="$PWD" /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python runs/cost_response_ablation_20261110/run_experiments.py
```

数据生成入口读取Traversability/configs/dataset_response.json和robots_response.json。原条件配置及旧数据保持可用。每版仅保留best.pt和last.pt，运行日志和评估产物存放在对应版本目录。

额外固定的新测试seed为20261207，使用原场景生成模板和原24组学生条件。该seed在四版训练完成并按val确定赢家之后才评估。根目录best.pt/last.pt链接到验证集选出的版本，四版各自权重同时保留。

## 已完成结果

四版均完成12轮、2052次更新。按固定val选出D_geometry，第12轮best.pt。验证分数A=0.2660、B=0.2494、C=0.2643、D=0.2481，D与B接近。原固定test的Cost MAE由旧模型3.773降至D的3.643，新seed20261207上则由4.315变为4.316，整体代价误差持平。条件响应改善，但不能据此声称全部指标或所有新场景均改善。

结果报告：runs/cost_response_ablation_20261110/REPORT.md。源码复现配置为Traversability/configs/train_response.json，初始化权重仍使用runs/cost_stand_lambda_20261110/best.pt。模型结构保持原四参数FiLM，尚未加入多层FiLM。

## D方案继续至50轮

从12轮断点继续到总计50轮，输出runs/cost_response_D50_20261110。训练入口支持`--resume <last.pt> --epochs 50 --output <run>`。延长周期时，保留优化器动量和随机状态，按当前epoch在新余弦周期中的位置恢复学习率，避免原12轮已归零的学习率使续训不更新。该接续不是从第1轮就使用50轮周期的独立重跑。

已完成50轮、8550次更新。新增38轮训练进程耗时1674秒，CUDA分配峰值4.54 GiB。验证综合分数选出的best.pt为第27轮，分数0.2317。last.pt为第50轮，验证综合分数0.2327。固定test Cost MAE从D12的3.643降至best的3.549、last的3.471。seed20261207的Cost MAE从4.316降至4.202、4.115，坡度和站立比例的差值MAE也下降。该seed沿用上轮测试，本轮作为固定回归测试。完整指标、GT对比图及三组机器人参数的完整预测PCD见runs/cost_response_D50_20261110/REPORT.md。
