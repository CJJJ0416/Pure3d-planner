# PTv3 历史工程目录

更新日期：2026-11-19。工程已从 Pure3D_ws 根目录移至 `third_party_lib/ptv3_tomography/`，保留早期单帧 LiDAR 预测分层 cost/height 地图的训练、推理和可视化代码。

```text
Pure3D_ws/
├── common/
│   ├── ptv3_tomography/                  # 历史学习实验工程
│   │   ├── pyproject.toml
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── docs/
│   │   └── src/ptv3_tomography/          # Python 包，导入名保持不变
│   └── vst_global_planner/              # 当前原生 VST 教师与传统规划器
├── utils/PointTransformerV3/            # 共享的 PTv3 核心
├── utils/Traversability/                      # 当前条件化代价训练与预测
├── common/                               # 当前在线预测与导航交接
├── outputs/ptv3_tomography/             # 历史产物约定，不表示数据已存在
└── runs/                                # 运行记录
```

当前 `Traversability`、`Entry` 和 `planner` 不导入本工程。`third_party_lib/vst_global_planner` 继续提供几何代价、教师标签与支撑查询，不依赖本工程的网络训练。

## 入口路径

从 Pure3D_ws 根目录执行。以下命令仅检查训练入口参数，不启动训练：

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python \
  third_party_lib/ptv3_tomography/scripts/training/run_pretraining.py --help
```

需要恢复旧实验时，使用实际存在的数据登记文件和独立运行名：

```bash
python third_party_lib/ptv3_tomography/scripts/training/run_pretraining.py \
  --registry /absolute/path/to/registry.json \
  --config third_party_lib/ptv3_tomography/configs/perception_pretrain_rtx4060_batch.json \
  --output runs/legacy_ptv3_local_run
```

包安装路径为 `utils/PointTransformerV3` 和 `third_party_lib/ptv3_tomography`。服务器打包入口沿用此布局。旧采集、场景生成及 Isaac 可视化脚本仍可能需要原 glocal2-nav/Isaac 环境，当前迁移不包含这些外部工程或历史数据。

## 历史记录与路径解析

`src/ptv3_tomography/paths.py` 将工程根定位到本目录，工作区根定位到 Pure3D_ws，PTv3 核心定位到 `utils/PointTransformerV3`。已有数据记录中的旧工程绝对路径在读取时映射，原数据与 checkpoint 不改写。

`MIGRATION.json` 保留早期迁移来源、原绝对路径和文件指纹。其他文档保留历史实验数值，缺失历史文件的链接已改为纯文本路径。此前本地训练记录见 [LOCAL_BATCH_TRAINING.md](LOCAL_BATCH_TRAINING.md)。

本次移动与路径、文档检查的备份和核验记录位于 `outputs/maintenance/ptv3_relocation_20261119/`，没有启动训练、Isaac 或导航。
