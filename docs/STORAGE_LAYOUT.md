# 本地产物存放规则

更新日期：2026-11-16。

| 目录 | 内容 |
|---|---|
| `datasets/` | 原始PCD、完整几何、教师标签、条件表、数据划分、场景overview、采集包和固定测试输入 |
| `weights/{filter,cost}/<实验名>/` | 实际checkpoint文件及来源记录 |
| `runs/{filter,cost}/<实验名>/` | 训练配置、历史、日志、指标、曲线，以及指向weights的checkpoint链接 |
| `outputs/` | 新的推理、ROS、可视化和导航输出 |
| `.cache/` | 本机编译产物、原生库及Gazebo colcon工作目录 |
| `docs/reports/` | 保留的历史诊断报告和图，不包含大数组或地图缓存 |

`runs` 不再存放监督数据、真实checkpoint副本、仿真工作空间、导航地图缓存或独立推理点云。训练仍可通过 `runs/<模块>/<实验名>/best.pt` 和 `last.pt` 加载与恢复。它们是相对符号链接，真实文件在 `weights`。复制到另一台机器时同时复制两个目录，或直接使用weights中的路径。

Filter和Cost的正式保存函数均已接入 `utils/checkpoints.py`。继续使用原先的原子写入方式，checkpoint内容、优化器和恢复接口不变。显式指定工作区之外的输出路径时沿用该路径。

轮换Cost场景的永久参数、overview和条件归档写到 `datasets/cost/<实验名>/scene_archive/`，批次记录写到同级 `generations/`。可替换的大文件仍使用各训练配置中的data目录。

## 本次保留内容

- 7组Cost模型，13个best/last文件，全部保持原始SHA256并完成CPU加载核验。
- 当前Small微调模型：`weights/cost/cost_full3d_small_unfrozen_e200_20261114-005204/best.pt`，原第56轮。
- 两份Small训练的配置、历史、曲线与选优记录分别位于 `runs/cost/`。
- Small诊断的完整几何和教师标签位于 `datasets/cost/small_diagnosis_20261114/`，报告位于 `docs/reports/cost/cost_small_diagnosis_20261114/REPORT.md`。
- 旋梯与SUST的VST参考位于 `datasets/cost/teacher_references/`。
- 原始ROS采集包位于 `datasets/recordings/entry_bag_mission/`。
- 当前离线工具引用的真实输入和固定规划测试图位于 `datasets/fixtures/entry/`、`datasets/benchmarks/surface_planning/`。
- 当前 Gazebo 使用 EOLP 的 Building-no-rails，terrain 导出仍位于 `datasets/simulation/gazebo_terrain/`，colcon 产物位于 `.cache/gazebo_ws/`。

历史配置中已有的缺失模型不会被替换为其他权重。

完整迁移清单、原始目录盘点和核验位于 `outputs/maintenance/runs_cleanup_20261116/`。未修改已有datasets中的训练/验证划分、教师规则、机器人参数或模型结构，也未启动训练。
