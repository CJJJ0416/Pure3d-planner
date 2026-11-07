# 新楼梯场景：原权重诊断与完整 VST costmap

2026-11-04。本轮没有训练或微调。原4060单epoch checkpoint的SHA-256保持为`416d597b68962be23511312efb34276b0831feb520e7535deae21f95ce8d72c2`。

## 完成内容

- [x] 生成不同于训练场景的新几何。使用ETH Zurich `leggedrobotics/terrain-generator`的已固定版本楼梯模块，再拼装地面、平台、两个障碍物。5级楼梯，高12cm、踏面40cm、宽1.8m，平台高0.6m。原始模块指纹与来源记录在场景报告中。本轮没有运行完整WFC流程。
- [x] 使用相同VST参数执行GPU建图，保存原始cost、地面高度和地图坐标元数据。
- [x] Isaac Sim PhysX沿参考路径采集40帧，每帧1000射线。运动是位姿驱动的传感器采样，没有四足动力学执行。
- [x] 全部标签与坐标独立核验。3704个高度单元、3344个cost单元，标签高度对源网格最大误差0.1075mm。1863个cost单元≥10，监督掩码内没有cost=60。支撑继续unknown。
- [x] 完整PTv3原权重推理，保存40份6×64×64地图。官方46个文件、checkpoint、标签来源和原场景均通过指纹复核。
- [x] 起点0、楼梯中段16、平台39的全部高度带对照图。第16帧同视角RGB/GT/预测对比图复用逐单元遮挡记录，32个可见cost单元、36个可见高度单元。旧frame52图像兼容检查逐像素相同。
- [x] Isaac原生视口生成第16帧GT、预测和误差六张截图。预测3D点使用预测高度。此次为headless截图，未留下交互窗口。
- [x] 直接读取完整VST costmap，显示两个104×104层全部21632格，0.1m分辨率，不使用局部裁图或监督掩码。标出采集参考路径、起终点与第16帧雷达。原始cost=60用红色，其中无地面观测的格子额外加斜纹。
- [x] Isaac原生视口显示完整VST层，使用保存的世界坐标地面高度。为避免深度冲突只作1.5cm显示抬升，原地图数值不变。无地面高度的格子仅在3D中省略，完整2D图仍保留全部格子。

## 结果

| 数据 | cost MAE，原始0–60尺度 | 高度MAE |
|---|---:|---:|
| 原训练场景130帧 | 7.5453 | 7.807cm |
| 新楼梯场景40帧 | 11.5192 | 27.957cm |

两场景的标签与高度带分布不同，此表用于定位迁移问题。新场景主要高度带[-0.5,0)包含3478个单元，预测平均偏高28.31cm。当前单场景、单epoch模型尚未表现出稳定的跨场景性能。

RGB对照图中的点位共享GT表面锚点，颜色分别来自GT或真实预测。Isaac原生预测截图使用预测高度形成3D坐标。完整VST图来自全场景离线建图，不等于单帧LiDAR全部可见区域。

## 证据与入口

所有结果保存在`outputs/terrain_generator_transfer_20261104/`。`transfer_report.json`汇总原权重诊断，`scene/geometry_report.json`记录生成来源，`labels/validation/alignment_report.json`记录标签验收。

- `maps/plots/frame_00016_rgb_gt_prediction.png`：同视角RGB、GT、预测。
- `maps/plots/full_vst_costmaps.png`及同名JSON：完整VST两层原始costmap与坐标验收。
- `native/report.json`：PTv3/GT原生截图。
- `native_full_vst/report.json`：完整VST原生叠加截图和源文件指纹。

新增入口位于`scripts/data/generate_terrain_generator_case.py`、`scripts/visualization/render_full_vst_costmap.py`、`scripts/visualization/view_full_vst_isaac.py`。各入口支持`--help`。场景生成与GPU VST使用工作区`.venv`，PhysX与可视化使用`env_isaaclab`，模型推理使用`.venv_ptv3_gpu`。

下一步扩展多场景、多路线训练数据，按场景谱系隔离验证集，增加训练轮次后重复本次诊断。危险cost=60与支撑标签另需补充有效覆盖。
