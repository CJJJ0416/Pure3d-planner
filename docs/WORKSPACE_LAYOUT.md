# Pure3D 文件架构

更新日期：2026-11-07。工程根目录：`/home/cjh/Pure3D_ws`。

采用扁平结构：任务代码直接放任务目录，每个建筑的数据集中在一个文件夹，每次运行的结果集中在一个文件夹。当前先做条件化 cost。

## 日常目录

```text
Pure3D_ws/
├── Traversability/       # 当前：条件化 cost 训练与推理
├── Reconstruction/       # 后续：点云清理与结构修复
├── common/               # VST、场景生成工具
│   ├── vst_global_planner/
│   └── geneator/
├── planner/              # 三维代价地图与规划器接口
├── datasets/             # 点云、VST 监督、数据划分、缓存
├── utils/                # PTv3、共用点云 I/O、坐标、分块、编码器封装
│   └── PointTransformerV3/
├── runs/                 # 权重、日志、预测点云、对比图
├── docs/                 # 架构、数据契约、训练说明
├── README.md
├── environment-ptv3.yml
└── ptv3_tomography/       # 保留的旧工程参考
```

外部工具沿用自身内部结构。项目任务目录按文件组织。

`common/` 与 `utils/` 的职责固定如下：

| 目录 | 内容 |
|---|---|
| `common/vst_global_planner/` | VST 标签生成内核、原生规划器和相关场景工具 |
| `common/geneator/` | 已迁入的场景生成、修改、回放和环境脚本 |
| `utils/PointTransformerV3/` | 预训练 PTv3 编码器源码与加载接口 |
| `utils/` 下的共用 Python 文件 | 后续确有需要时加入点云 I/O、坐标转换、分块和编码器封装 |

任务代码通过这些目录复用工具，不向任务目录复制另一份 VST 或 PTv3。

## 任务代码直接放一层

当前 cost 阶段后续按功能增加文件：

```text
Traversability/
├── README.md
├── dataset.py            # 点云、机器人条件和 VST 标签读取
├── model.py              # 预训练编码器 + 条件化 cost 头
├── prepare.py            # 数据准备
├── train.py              # 训练、损失、断点保存
├── predict.py            # 分块推理、融合与地图导出
├── visualize.py          # GT、预测和误差对比
├── robots.json           # 机器人能力参数
├── dataset.json          # 数据路径与划分
├── train_local.json
└── train_server.json
```

目前 `Traversability/` 正式目录只有 README。以上是后续文件职责，不创建占位模型或空训练入口。`Reconstruction/` 同样按文件组织，`planner/` 保留已有实现。

两个阶段确有共用逻辑时，放在 `utils/pointcloud.py` 等文件中。`common/` 专门管理 VST、场景生成工具。

## 每个建筑的数据放一起

```text
datasets/
├── README.md
├── splits.json
├── cache/                  # 可再生的数据缓存
├── rot_stairs/
│   └── raw.pcd             # 已有真实建筑扫描
└── building_001/
    ├── raw.pcd             # 原始点云，需要时保留
    ├── map.pcd             # 当前 cost 阶段输入的建筑 PCD
    ├── robots.json
    ├── vst_robot_001.pickle
    ├── vst_robot_001.meta.json
    ├── vst_robot_002.pickle
    └── vst_robot_002.meta.json
```

- 同一建筑的点云、机器人条件和标签放在一起，不拆到多组目录。
- `splits.json` 按建筑划分 train、val、test。同一建筑的分块和机器人配置属于同一划分。
- `robots.json` 记录机器人能力参数和 VST 参数，`vst_<robot_id>` 与其对应。参数字段及映射在数据契约中确定。
- 已有修复 PCD 和 VST 可以在数据清单中直接引用，无需先复制所有数据。
- `rot_stairs/raw.pcd` 是原始扫描，尚未声明为修复完成的数据。

## 每次实验只放一层

```text
runs/
└── cost_<experiment_name>/
    ├── best.pt
    ├── last.pt
    ├── config.json
    ├── dataset_manifest.json
    ├── train.log
    ├── metrics.jsonl
    ├── summary.json
    ├── plots/
    └── predictions/
        └── building_001_robot_001.pcd
```

训练权重只保留 best.pt 和 last.pt。默认按验证集选择 best，last 保存恢复训练所需状态。没有独立验证集的短运行只报告流程结果。

重建实验使用 `runs/reconstruction_<experiment_name>/`。不叠加任务类型目录和逐阶段 tasks 目录。官方预训练权重继续使用现有缓存，不复制进源码目录。

数据缓存统一放在 `datasets/cache/<dataset_version>/`。运行目录通过数据清单记录本次使用的缓存版本。

## 当前整理

- 已移除上一版仅含 .gitkeep 的空功能子目录。
- VST 统一位于 `common/vst_global_planner/`，场景脚本位于 `common/geneator/`，PTv3 位于 `utils/PointTransformerV3/`。
- 旧工程目录统一为 `ptv3_tomography/`，继续作为参考保留。
- 已将 `Reconstruction/pcd_files/rot_stairs.pcd` 移到 `datasets/rot_stairs/raw.pcd`，同步更新教程和示例中的引用。
- 现有 Notebook、场景脚本、第三方源码和旧工程内容保留。
- 此前的训练草稿及标签仍在 `runs/conditioned_cost_local_20261107/`，尚未作为正式训练版本验收。
- 本次只整理文件，不安装依赖、不启动训练、不提交 Git。

## Git

源码、配置、文档和 datasets/splits.json 进入 Git。datasets 内的场景目录与 runs 由 .gitignore 排除。
