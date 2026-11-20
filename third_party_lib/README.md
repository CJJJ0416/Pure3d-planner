# 独立参考工程与外部基线

| 组件 | 职责 |
|---|---|
| [ptv3_tomography](ptv3_tomography/README.md) | 早期独立 PTv3 单帧地图预测工程与历史训练说明 |
| [vst_global_planner](vst_global_planner/README.md) | 原生 VST/Tomography 教师、传统 A* 与 GPMP 基线 |

各组件保留原有源码、来源记录、许可证和独立 Python 项目配置。Pure3D 的 Filter、Traversability 和 planner 通过明确路径调用对应组件，训练与推理实现继续由所属模块维护。

本目录与用户主目录下的 `$HOME/third_party_lib/libtorch` 分别管理；后者仍是机器人控制器现有的外部 libtorch 依赖。
