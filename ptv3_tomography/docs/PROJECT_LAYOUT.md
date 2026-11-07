# PTv3 目录与本地运行

2026-11-05，按用户要求将任务工程与官方核心库分开。

```text
glocal2-nav/
├── src/ptv3_tomography/                   # 自有训练工程
│   ├── pyproject.toml
│   ├── configs/
│   ├── scripts/
│   ├── docs/
│   └── src/ptv3_tomography/               # 可导入的 Python 包
├── third_party/PointTransformerV3/        # 固定版本的官方核心库
│   ├── pyproject.toml
│   ├── TOMOGRAPHY_UPSTREAM_PIN.json
│   └── src/point_transformer_v3/
├── third_party/vst_global_planner/        # 场景、VST 与规划器
├── outputs/ptv3_tomography/               # 已有数据和历史产物
└── runs/<实验名>/                        # 新训练的 best.pt、last.pt、日志、评估和缓存
```

外层 `src/ptv3_tomography` 是工程目录，内层 `src/ptv3_tomography` 是 setuptools 的标准 src-layout Python 包。配置、脚本和文档随自有工程维护。官方核心的 46 份固定文件内容保持不变。

## 安装与启动

以下命令从 glocal2-nav 根目录执行。在服务器激活 `/opt/conda/envs/rl_env_glocal2`，本地使用 `.venv_ptv3_gpu/bin/python`。

```bash
python -m pip install 'hatchling>=1.27' 'editables>=0.3,<1' 'setuptools>=68'
python -m pip install --no-build-isolation --no-deps \
  -e third_party/PointTransformerV3 -e src/ptv3_tomography

python src/ptv3_tomography/scripts/training/run_pretraining.py \
  --registry outputs/ptv3_tomography/dataset/registry.json \
  --config src/ptv3_tomography/configs/perception_pretrain_rtx4060_batch.json \
  --output runs/local_run_01
```

registry 路径填写实际已准备的数据登记文件。新训练默认将缓存写入 `runs/<实验名>/cache/`，权重只保留顶层的 `best.pt` 和 `last.pt`，训练日志在 `logs/`。H100 后续使用 `configs/perception_pretrain_h100_batch.json`。迁移包导出时也保持 `src/ptv3_tomography` 与 `third_party/PointTransformerV3` 两个独立目录。

## 旧产物路径

任务的 `paths.py` 集中管理 PROJECT_ROOT、WORKSPACE_ROOT、PTV3_ROOT、OUTPUT_ROOT 与 RUN_ROOT。数据清单中的旧绝对路径按以下关系在读取时解析，原始清单、数据与 checkpoint 内容不改写。

| 旧位置 | 当前位置 |
| --- | --- |
| 原先嵌在官方库下的 Tomography 工程 | `src/ptv3_tomography` |
| 原先放在 src 下的官方核心 | `third_party/PointTransformerV3` |
| 原训练工程下的 outputs | `outputs/ptv3_tomography` |

整理执行时，原训练工程的 outputs 和仓库根目录此前训练结果已经不存在。本次迁移了现有源码，并创建统一结果目录，没有迁移或重新生成数据与权重。此前训练数值保留在 [LOCAL_BATCH_TRAINING.md](LOCAL_BATCH_TRAINING.md)，其本地结果路径属于历史记录。

## 已同步的入口

训练与缓存准备、地图推理、逐点恢复、RGB/Isaac 可视化、GPU 射线采集脚本、分段调度器、portable 打包与文档链接均改为新目录。

源码仍由 glocal2-nav/master 统一管理。新训练结果写入 Git 忽略的 `runs/`，历史结果和迁移包仍保存在 `outputs/`。当前未连接服务器或启动 Isaac。

2026-11-06：标签引擎集中于本工程。正式配对统一调用本工程入口，Isaac 目录旧标签模块、v1 串行训练器及重复校验脚本已删除。监督与清理说明见 [TEACHER_SUPERVISION_CLEANUP.md](TEACHER_SUPERVISION_CLEANUP.md)。
