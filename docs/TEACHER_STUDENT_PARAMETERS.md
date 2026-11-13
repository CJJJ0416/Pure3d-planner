# VST 教师与 trav_cost 学生参数

更新日期：2026-11-10。

## 学生四维条件

顺序为 `[clearance_min, slope_max, step_max, stand_ratio]`，归一化尺度为 `[1.0, 0.5, 0.3, 1.0]`。

| 学生参数 | VST 教师字段 | 基准 | 联合采样范围 |
|---|---|---:|---:|
| clearance_min | trav.safe_margin | 0.5 m | 0.2–1.0 m |
| slope_max | trav.slope_max | 0.4 rad | 0.18–0.65 rad |
| step_max | trav.step_max | 0.3 m | 0.08–0.4 m |
| stand_ratio | trav.standable_ratio | 0.1 | 0.05–0.5 |

stand_ratio是邻域内满足可站立条件的网格比例要求。原生台阶分支使用 `int(stand_ratio * kernel_size**2)-1` 作为计数阈值；kernel_size=7时，0.1对应3，0.3对应13。更大的比例要求更严格。平坦分支仍按原VST规则计算。

教师interval_free固定0.5m，不作为学生输入。interval_min也独立保留为0.5m。

## 教师五维存储

教师条件顺序为 `[clearance_min, slope_max, step_max, interval_free, stand_ratio]`，末尾追加stand_ratio以保留前四列含义。教师尺度为 `[1.0,0.5,0.3,1.0,1.0]`。PointBlocks按名称选择第0、1、2、4列，并仅训练interval_free=0.5的条件。

完整原生教师配置仍为 `common/vst_global_planner/configs/teacher_building.json`。map.resolution=0.1、ground_h=0、slice_dh=0.5、kernel_size=7、interval_min=0.5、cost_barrier=50、inflation=0.1、adjust_cost_p=2等原生规则保留，standable_ratio改为条件覆盖项。地图、几何与机器人条件分别记录在pickle中。

## 每场景监督条件

`Traversability/configs/robots.json` 定义14组固定学生条件，覆盖基准、四项参数各三档变化，以及 `[0.9,0.6,0.3,0.1]` 联合条件。每个场景按自身seed追加10组四参数联合分层随机采样，合计24组学生条件。

另外保留tall_robot（interval_free=1.1m）教师诊断，不进入当前学生训练。每场景共25组教师pickle与Tomogram PCD。

训练、验证和测试建筑使用相同固定条件与各自的随机条件。新数据集为 `datasets/wfc_capability_dr_stand_v3`，训练4个建筑，验证1个，测试2个（包括此前失败的seed 20261127）。旧三参数数据 `datasets/wfc_capability_dr_v2` 与旧权重保留作对照。

## 损失与权重

训练启用 `condition_difference_weight=1.0`：

`L = SmoothL1(pred, GT) + lambda * SmoothL1(pred(k)-pred(nominal), GT(k)-GT(nominal))`。

条件差异项学习教师在同一点随参数变化的代价差，不强制最终VST代价逐点严格单调。训练保留全部原生Tomogram查询点，不添加支撑、连续性、回波或覆盖筛选。

FiLM输入从3扩展为4，训练入口 `--initialize` 按参数名复制旧条件权重，新stand_ratio列从0开始参与训练。模型其他权重作为初始化，优化器与轮数重新开始。新checkpoint记录四维condition_names和condition_scale。旧三参数checkpoint仍按自身元数据执行三参数推理。

## 输出

每组教师保存 `vst_<profile>.pickle` 与 `tomogram_<profile>.pcd`。Tomogram直接导出原生合并ground XYZ和cost/50，不映射到建筑原始点作为训练标签。部署仍在建筑PCD全部原始XYZ预测cost，输入输出保持地图原点。

`robot_conditions.json`、`teacher.json`、`manifest.json`、`blocks.json`记录条件与数据版本。教师概览展示6组代表条件，全部25组监督均保存。
