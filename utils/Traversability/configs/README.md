# 预测参数

修改 [predict.json](predict.json) 调整 full3D 预测。两个预测入口默认读取此文件，`--config` 可选择另一份预测配置。

| 参数 | 含义 |
|---|---|
| `checkpoint` | 已训练权重，相对路径按工程根目录解析 |
| `condition.clearance_min` | 所需净空，单位 m |
| `condition.slope_max` | 最大坡度，单位 rad |
| `condition.step_max` | 最大跨阶高度，单位 m |
| `condition.stand_ratio` | 支撑比例条件 |
| `device`、`cpu_threads` | 预测设备与 CPU 线程数 |
| `geometry_voxel_size_m` | 输入几何代表点的体素尺寸，单位 m |
| `window.core_size_m`、`window.context_m` | 三维核心块尺寸与上下文扩展距离，单位 m |
| `geometry_python` | `inference.full3d` 生成支撑查询时使用的解释器，`full3d_raw` 不使用此项 |

默认使用现有 Small checkpoint，条件为 `[0.5, 0.4, 0.2, 0.0]`。网络结构、条件归一化和编码器权重从 checkpoint 读取。可选运行参数省略时沿用 checkpoint，`checkpoint` 和 `condition` 也可由命令行提供。命令行中的 `--checkpoint`、`--condition`、`--device` 优先于配置文件。

保留输入 PCD 的全部原始 XYZ 和行序：

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m utils.Traversability.inference.full3d_raw \
  --config utils/Traversability/configs/predict.json \
  --pcd /absolute/path/to/input.pcd --output outputs/cost/prediction
```

需要原生几何支撑点输出时，将入口换为 `utils.Traversability.inference.full3d`。实际生效参数写入输出目录的 `prediction_config.json`。

训练模板位于 `utils/Traversability/training/configs/`，数据生成模板位于 `utils/Traversability/data/configs/`，历史楼梯模板位于 `utils/Traversability/experiments/configs/`。训练过程记录在 `runs/cost/<运行名>/`。本目录的加载器只接受预测字段。

在线 ROS 预测由 `Entrance/config/local_cost.yaml` 配置。本文件用于上述离线 full3D 预测入口。
