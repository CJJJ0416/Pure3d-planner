# 数据

每个建筑一个文件夹，点云、机器人条件和 VST 标签放在一起：

```text
building_001/
├── raw.pcd
├── map.pcd
├── robots.json
├── vst_robot_001.pickle
├── vst_robot_001.meta.json
└── vst_robot_002.pickle
```

raw.pcd 是原始扫描，map.pcd 是 cost 阶段使用的建筑点云。已有数据也可在清单中直接引用。

训练、验证和测试按建筑划分，统一记录在 splits.json。场景目录不进入 Git。可再生缓存统一写入 datasets/cache/<dataset_version>/。

当前已有原始扫描：rot_stairs/raw.pcd，5,886,268 点。该文件从原位置移动，点云内容未改动。

2026-11-16从runs迁入的内容按以下类别保存：

- `cost/`：场景参数、条件表、overview、原生教师标签和Small诊断几何。
- `recordings/`：原始ROS采集包。
- `fixtures/entry/`：当前离线入口使用的固定真实输入。
- `benchmarks/surface_planning/`：固定地图和查询，保存的student cost仅作为测速输入，不作为教师GT。
- `simulation/`：当前仿真世界和出生位姿。

训练产生的实际权重统一位于 `../weights/`，不存入数据目录。详细规则见 `../docs/STORAGE_LAYOUT.md`。
