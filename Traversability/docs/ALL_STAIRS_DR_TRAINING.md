# 全类型楼梯域随机监督与增强训练

本轮从`runs/cost_wide_stair_v2/best.pt`继续微调。保持归一化cost为0..1、教师障碍代价50、距离曲线p=2、原网络结构和损失。

## 数据

- 从seed20261127的4处直梯、2处折返梯、2处旋转梯提取几何参数，不把该地图的原始PCD、GT或裁剪块放入训练。
- 每个模板生成8个训练、2个验证、2个测试独立seed，共96个场景，64/16/16划分。
- 每个seed独立采样几何和机器人条件。每个模板的一半场景在原尺寸附近变化，另一半作宽范围随机化。
- 踏步高度0.12..0.44m，直梯/折返梯踏深0.18..0.85m，梯宽0.55..1.90m，整梯目标高差2.3..4.0m。尺寸锚点保留模板尺寸。
- 旋转梯内半径0.18..0.95m、转角0.65..1.6圈，实际踏深由半径、转角和踏步数共同决定，并记录在scene.json。
- 独立随机栏杆、左右镜像、全局朝向、XY相位、井口侧向间距、平台尺寸、侧墙以及两层/三层背景。
- 每个seed保存terrain.ply、完整表面采样PCD、5cm降采样PCD、overview.png和实际几何参数。
- 每个条件重新运行真实原生VST，保留全部原生Tomogram查询。训练场景使用两组条件bank，共76组标签。验证/测试另含stand扫描，共47组。
- clearance_min范围0.2..1.0，slope_max范围0.18..0.70，step_max范围0.08..0.55，stand_ratio范围0.02..0.50，interval_free固定0.5m。
- 旧4个训练建筑和1个旧验证建筑复用上一轮已重建的四参数监督与缓存。

## 训练

- 新增30轮，真实batch=2，workers=2，BF16，前2轮冻结编码器，之后联合微调。
- 每轮480块，其中336块来自新楼梯，144块来自旧建筑回放。
- 新楼梯按8个模板等额采样，每个模板每轮42块，避免大井口或长梯因块数更多而占据更高比例。
- 代价头学习率5e-5，编码器5e-6，沿用30轮cosine调度。
- SmoothL1 beta=0.1，条件差值lambda=1，变化点阈值0.02，四参数权重[1,2,1,2]，均沿用当前checkpoint。
- 每5轮固定验证。沿用原选优规则，先检查错误放行与非楼梯误差，再比较楼梯断裂率、stand差值MAE、整体MAE。
- 只保存本轮best.pt和last.pt，不覆盖之前的两个基线best.pt。

## 执行

配置：`Traversability/configs/all_stairs_dr_v3.json`。

```bash
cd /home/cjh/Pure3D_ws
PYTHONPATH=/home/cjh/Pure3D_ws HF_HUB_OFFLINE=1 \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
-m Traversability.experiments.all_stair_experiment
```

流水线自动完成数据生成、GPU教师标签、分块、初始验证、训练、独立测试、旧建筑回归、大地图全部8处楼梯回归与完整PCD推理。

几何工具与教师使用`/home/cjh/glocal2-nav/.venv/bin/python`。训练与推理使用本机`pure3d-ptv3`环境。

数据位于`datasets/wfc_all_stairs_dr_v3`，运行产物位于`runs/cost_all_stairs_dr_v3`。

运行目录中的`progress.json`每5秒自动刷新，包含文字进度条、完整训练轮数、日志更新步数、训练loss、固定验证Cost MAE、楼梯断裂率、错误放行率、stand差值MAE、非楼梯误差与历史趋势。验证指标每5轮更新，始终显示对应epoch。测试完成后补充独立测试、旧建筑验证与全部8处楼梯回归结果。流程结束后停止刷新。训练批次进度按每50步输出的日志计算。

当前不使用Codex定时监控。训练流水线和本地进度JSON刷新独立运行。

## 评估

比较微调前后的固定验证、16个独立测试seed、旧建筑验证，以及原地图全部8处楼梯的MAE、误阻断、误放行、stand条件响应。

原地图参与过模板选择，归类为已知地图回归。独立随机seed没有复用原PCD，但属于相同模板分布，不能把该测试直接解释为所有新建筑上的泛化。

完整PCD推理分别选取直梯、折返梯、旋转梯测试场景和原大地图。每处原楼梯提供GT/Before/After俯视与三维对比图，统一XYZ、p=2和色标。
