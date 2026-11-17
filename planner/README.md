# DC 三维点云规划

在线 Entry 新增 `SparseSurfacePlanner`（`planner/surface.py`）：通行性只由 cost 判定，不做坡度、阶高、法向、净空或机身碰撞过滤。C++ 单次稀疏聚合同步生成低 cost 搜索表面和完整 cost 查询体素；无 EDT/DC，GPMP 不再重读、排序原始点云。坐标仅用于分层和相邻柱配对，缺测区域不补边。在线默认选择与运行说明见 `Entry/EXPLORATION.md`；以下独立静态工具仍默认使用 DC，历史结果保持原义。

正式方案为 **XYZ + trav_cost PCD → 原始点 cost < 0.5 筛选 → 均值体素 → DC 表面 → 跨阶连边 → C++ 单向 A* → 反重力 GPMP**。唯一消融为同一张 DC 图关闭跨阶连边。点云、代价、节点、起终点、搜索目标和 GPMP 全部共用。

输入 cost 已由 `Traversability` 预测，范围为0–1。规划器不重新运行网络，不读取 OBJ、分层地图或合成 Mesh 缓存。地图原点与米制 XYZ 保持不变。更新机器人条件后先重新预测 cost，再使用相同条件构图。

`dc_input_cost_max=0.5` 在原始点级、体素均值与距离场构建前执行严格 `<` 筛选，排除非有限 XYZ/cost 和负代价；cost 恰为 0.5 的点也排除。DC 几何、法向、来源锚点及净空检查使用筛选结果，表面拓扑可能随之改变。完整输入 PCD 与 cost 保留；可选 GPMP 仍按完整原始 cost 的体素场查询代价，并在首次使用时加载。设置 `dc_input_cost_max=null` 可复现旧版未筛选 DC 几何。`map.json` 的 `dc_input_filter` 记录筛选前后点数，`geometry_field` 记录实际网格尺寸。

## 文件

```text
planner/
├── plan.py                  # 正式CLI，默认有跨阶边
├── planner_wrapper.py       # DCSurfacePlanner，同图多次查询
├── io.py                    # 配置、地图、路径、状态导出
├── config/
│   ├── param.py             # 唯一配置定义
│   ├── dc.json              # 默认完整地图配置
│   └── rot_stair.json       # 已验证真实楼梯区块配置
├── geometry/
│   ├── voxel.py             # 均值体素、实测占用壳几何场
│   └── dual_contouring.py   # Hermite/QEF DC、表面法向
├── graph/
│   ├── adjacency.py         # 三角边、cost剪枝、跨阶边
│   ├── astar.py             # 原生搜索接口
│   └── astar.cpp            # 仅保留单向A*
├── optimization/gpmp.py     # 已采用的高度曲线与水平走廊优化
├── visualization/
│   ├── plot.py              # 读取正式结果绘图
│   └── rviz.py              # 正式方案、消融和静态参考轨迹
├── interactive.py          # 保留RViz交互选点，调用相同DC规划器
├── docs/
└── requirements.txt
```

## 运行

使用现有本机环境，不安装或更换环境：

```bash
cd /home/cjh/Pure3D_ws
/home/cjh/glocal2-nav/.venv/bin/python -m planner.plan \
  --pcd runs/cost/cost_wide_stair_v2/predictions/rot_stairs_20261110_142602/prediction.pcd \
  --config planner/config/rot_stair.json \
  --start 0 4 0.2 --goal 0 4 3.3 \
  --output outputs/planner/rot_stair_lower_middle_20261111 --plot
```

增加 `--ablation` 在同一次地图准备后运行有/无跨阶边两组。`--without-step` 只运行消融。默认 `dc.json` 不裁剪输入。真实楼梯示例的 ROI 明确保存在 `rot_stair.json`，没有自动缩小地图或调整阈值。

运行目录只有 `map/`、`with_step/`、可选 `without_step/` 及配置、查询、summary、REPORT、图。地图只保存可视化所需点和边，不导出密集几何场和重复CSR缓存。编译产物保存在该run的 `native/`，不写入源码目录。无路径时记录失败状态，不保留该输出位置的旧路径。

旧版未启用输入 cost 筛选的验收结果位于 `outputs/planner/rot_stair_lower_middle_20261111`，以下表格为历史记录。当前启用筛选后应另存新运行目录验证：

| 方案 | A*长度 | cost积分 | GPMP长度 | GPMP收敛 |
|---|---:|---:|---:|---|
| DC + 跨阶边 | 30.749033m | 3.352990 | 28.686338m | True |
| DC 无跨阶边 | 31.153218m | 3.443571 | 28.675990m | True |

## RViz

```bash
source /opt/ros/humble/setup.bash
cd /home/cjh/Pure3D_ws
/usr/bin/python3 -m planner.visualization.rviz \
  --run outputs/planner/rot_stair_lower_middle_20261111
```

默认显示紫色正式方案。左侧可开启深紫色无跨阶边消融、原三角边与新增跨阶边。细线是A*，粗线是GPMP。

最新VST对照仅保留为静态结果，不保留独立测速或VST规划实验脚本。查看该参考：

```bash
/usr/bin/python3 -m planner.visualization.rviz \
  --run outputs/planner/rot_stair_lower_middle_20261111 \
  --reference outputs/planner/rot_stair_lower_middle_20261111/reference_vst/reference.json
```

橙色为已保存的VST参考。该选点在VST15下无路，VST25下A*32.067899m、返回GPMP28.776659m。VST使用教师cost和原生优化器，原生接口不提供GPMP收敛标记。原始数据、参数及共同cost评价保存在 `reference_vst/`，不作为额外消融方案。

交互选点仍可通过现有预测可视化入口使用：

```bash
/usr/bin/python3 -m Traversability.visualization.rviz \
  --prediction runs/cost/cost_wide_stair_v2/predictions/rot_stairs_20261110_142602/prediction.pcd \
  --planning outputs/planner/rot_stair_lower_middle_20261111/with_step --interactive
```

## 接口与参数

```python
from planner import DCSurfacePlanner, PlannerConfig

config = PlannerConfig(roi_min=(-2, -4, -4), roi_max=(12, 8, 10))
planner = DCSurfacePlanner(config)
planner.load_pcd("prediction.pcd")
result = planner.plan((0, 4, 0.2), (0, 4, 3.3))
ablation = planner.plan((0, 4, 0.2), (0, 4, 3.3), step_edges=False)
```

机器人条件记录为 `clearance_min、slope_max、step_max、stand_ratio`。连边读取 `slope_max` 和 `step_max`，水平邻域半径默认0.3m，最小高差0.05m。新增跨阶边的来源PCD点沿反重力方向在 `interval_free=0.5m` 内不能遇到其它已占用体素，避免新连边跳到地板底面壳。自身占用体素跳过。该条件不删除原三角网格节点或边，预测cost保持原值。A*权重5，节点cost上限0.5。GPMP保持0.5m统一反重力偏移、0.2m水平走廊、0.4m高度平滑及±0.1m节点高度带。

起终点XYZ表示输入PCD上的支撑位置。规划器先定位最近的可搜索PCD体素锚点，再选择该来源体素对应的DC表面。RViz标记保存PCD锚点，避免将DC坐标重复吸附。每次查询的来源编号和坐标保存在 `endpoint_snap.json`，规划失败时同样保留。[端点吸附规则](docs/DC_ENDPOINT_ANCHOR.md)说明对应关系与接口。

[流程和数据约定](docs/POINTCLOUD_ASTAR_GPMP_PLAN.md)说明表面、搜索与机体轨迹的坐标关系。[A*权重说明](docs/ASTAR_COST_WEIGHT.md)给出长度与cost的目标函数。

几何场描述实测占用体素集合的边界，不填洞、不膨胀，不把正值当作射线确认的自由空间。跨阶边是显式几何运动假设。当前保留的真实楼梯案例是参考路径验证，不能由此推断全部缺测、重影场景或实机执行效果。
