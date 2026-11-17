# Entry：局部点云 cost 与 A* 导航

当前管线：Faster-LIO → 稳定体素上下文 → KNN → PTv3 + cost head → A* → `/pure_path` → SCAN。模式 4 的 A* 严格使用 `cost < 0.8`，失败停车。SCAN 保留原有局部优化。

## 目录与接口

每个模块保留一份实际实现。模块间使用完整包名直接导入，不保留根目录兼容转发、动态模块别名或重复配置链接。

| 位置 | 职责 |
|---|---|
| `predict.py` | 常驻 GPU worker 的实际命令行入口 |
| `utils/predict.py` | PTv3 与 cost head 推理、整图与过滤后推理接口 |
| `utils/io.py` | JSON/NumPy 原子读写、点云消息与位姿转换。ROS 依赖延迟导入 |
| `utils/core.py` | 地图、快照、分块和存储状态 |
| `utils/box.py` | 基于 LIO 的独立 6×6×5 m 盒子 |
| `utils/local_map.py` | 局部体素、稳定上下文、定时调度及延迟统计 |
| `utils/knn.py` | 邻域统计、保守删除规则与过滤缓存 |
| `utils/client.py` | ROS 接收端与 GPU worker 的进程通信 |
| `utils/surface.py` | 在线和实验共用的代价点云 A* 查询 |
| `utils/arrays.py`、`utils/voxel_filter.py` | 原生数组与体素过滤接口 |
| `runtime/` | ROS 接收、cost 会话、显示发布和耗时监测 |
| `planning/` | 人工目标、A* 参考与 SCAN 交接 |
| `rendering/` | RViz 点云、Marker 和离线显示 |
| `launch/`、`configs/` | 实际启动脚本、唯一 YAML 与 RViz 配置 |
| `experiments/` | 本地对照实验，不随源码提交 |
| `mission/` | 可复现的历史 bag、node 和 VST 工具 |
| `native/`、`docs/` | C++ 源码和使用说明 |

运行节点和公共库不能导入实验模块。跨节点共享函数应进入 `utils/`，不应从另一个 ROS 运行入口导入。历史实验仅在本地保留，不是正式入口的依赖。输出存放于工作区 `outputs/`，权重位于 `weights/`。

## 启动

```bash
cd /home/cjh/Pure3D_ws
./total_terminal.sh --flight-type 4 --cost-limit 0.8
# 独立启动 cost 与耗时监测：
./run_cost.sh --model tiny --rviz
# 只启动局部接收与推理节点：
./Entry/launch/local.sh --model tiny --rviz
```

实际 Python 入口是 `Entry.runtime.local`、`Entry.runtime.cost_session`、`Entry.planning.local_astar_follow` 和 `Entry.predict`。GPU worker 通常由接收节点管理。

[局部预测与 KNN](docs/LOCAL_COST.md)、[A* → SCAN](docs/PURE_PATH_FOLLOW.md)、[整图工具](docs/WHOLE_MAP.md)、[历史 bag 实验](docs/BAG_WORKFLOW.md)。

## 代码维护

`utils/` 应可在不加载 ROS 或 CUDA 模型的情况下导入，模型权重仅在实例化预测器时加载。新增入口直接放入所属目录，修改所有调用方和说明，不增加旧路径跳板。

2026-11-17 源码清理前后各完成 124 项既有回归检查，结果通过。按源码发布要求移除了测试与一次性验证脚本，历史结果保存在本机 `outputs/maintenance/source_release_20261117/`。后续修改仍需按变更范围验证实际入口、消息接口和原生构建。
