# 两条扫描路径、Filter 数据与静态参考

## 已实现的调度

`batch.py` 使用一个 CUDA 生产者，依次生成不同轨迹的逐点扫描。
扫描完成一条，就启动其独立 Faster-LIO 消费者，同时生成下一条。
最多两条任务处于 LIO / 监督处理中，不重复启动多个 GPU 采集进程。
配置见同目录 `batch_config.json`，首批复用 seed 821110 的建筑，生成两条不同路径。
每条 333 帧、10 Hz、10,000 射线/帧、45°安装角，保持原有噪声和 10×回放。

```bash
cd /home/cjh/Pure3D_ws
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m common.scanning.batch \
  --schedule pipeline
```

`--schedule serial` 使用相同数据和处理步骤生成串行对照。
默认输出分别为 `datasets/scanning_shared_two_routes_20261113/{pipeline,serial}`，
日志、时间线和指标位于 `outputs/scanning_shared_two_routes_20261113/`。
已有结果不自动覆盖，也不自动重试。再次实验用 `--output` 和 `--report-output` 指定新目录。

两条 LIO 分别使用 ROS_DOMAIN_ID 81、82，独立 `/clock`、节点、参数、日志、原始点和配准结果。
域内话题名称相同，DDS 域隔离保证不会混合。首条生成静态 Mesh，另一条只读共享它。
每条路径使用独立 observation_seed，分别生成测量噪声和 IMU 误差。

这是“下一条 Warp 与上一条 LIO 重叠”的旧版入口。
新版 `terrain_batch.py` 已将多个轨迹的射线合成一次 GPU launch，并共用静态 Mesh，
见同目录 [大 terrain 合批采集](TERRAIN_BATCH.md)。
本次实测硬件为 RTX 4060 Laptop，不是 RTX 4090。

## 幽灵点来源

- 显式悬浮回波：在 `mixed` / `dynamic` 原始扫描阶段，以 0.003 概率选择静态命中射线，
  将其测量距离向传感器拉近 0.3–1.0 m。保留逐点时间、原命中面片、实例和位移。
  这是人为仿真测量扰动，不是经过标定的 MID-360 物理回波模型。
- 动态拖影：人物和可移动物体沿轨迹运动，不同位置产生真实射线命中并经 LIO 累积。
- 配准及边缘缺陷：保留带噪 IMU 及边缘混合回波，IMU 时间始终准确，最终由 LIO 输出形成。
  不直接向最终地图添加随机点，也不按每帧真值重新对齐地图。

`explicit_ghost` 和 `ghost_pull_m` 为教师元数据。静态及显式悬浮回波统一按最终点到
完整静态 Mesh 的距离评分 `exp(-0.5*(d/0.03)^2)`。落回真实表面附近的点可获得高分。
运动实例来源点固定为 0。未扫描区域不生成 Filter 标签。

## Filter 数据与完整静态参考

每条路径的 `mixed/rough.pcd` 只保存 XYZ。`attributes.npz` 保存逐点来源与 keep_target。Filter 使用已有 `arrays_000`、`windows_000_*` 和 `boxes.json` 接口，批次 `manifest.json` 可由 `Filter.data.features.scene_records` 读取。

完整静态 Mesh 点用于数据质量、观测覆盖和可视化核对，不拼入模型输入。新导出使用 `reference_index.npy`、`reference_core.npy`、`reference_offsets.npy`，`training_pair.json` 记录原始观测与静态参考窗口。读取器也按旧数据自身元数据加载历史数组。

```python
from common.scanning.training_data import SharedScanBoxes

dataset = SharedScanBoxes('/absolute/path/to/route/mixed')
sample = dataset[0]
xyz = sample['xyz']
keep_target = sample['keep_target']
clean_xyz = sample['clean_xyz']  # 静态参考，供核对和显示
core_mask = sample['core_mask']
```

源实例、射线、时间、GT 和来源分类不进入学生输入。此入口生成数据，不启动模型训练。

两条路径属于同一建筑 `geometry_group_id`，均标记为 train。
未来按建筑 seed 划分 train / val / test，不能将它们拆到不同集合。

## 实际对照

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m common.scanning.inspect_batch \
  --serial outputs/scanning_shared_two_routes_20261113/serial/report.json \
  --pipeline outputs/scanning_shared_two_routes_20261113/pipeline/report.json \
  --output outputs/scanning_shared_two_routes_20261113/comparison
```

直接核对串行与流水线的原始扫描、射线和点序，检查核心区覆盖唯一性、标签公式与配对读取，
输出完整点云监督图和处理时间线。不启动模型训练。
