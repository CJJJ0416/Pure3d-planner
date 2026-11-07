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
