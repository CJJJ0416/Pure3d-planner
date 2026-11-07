# PTv3 服务器从零训练：Headless 与原生 VST 监督

更新：2026-11-06。本次重建场景、VST、路线、点云和标签，PTv3 与地图头随机初始化，优化器、学习率调度和训练计数从零开始。旧 80 epoch 实验的数据、权重、缓存和评估历史均不作为本次输入。

```text
九区域场景与域随机化
    ↓
VST 地图与规划路线 → 纯 CUDA Warp LiDAR 采集
    ↓
原生 VST cost / ground → 当前 LiDAR 局部 GT
    ↓
当前帧 XYZ + 重力 up → 完整 PTv3 → 地图头
    ↓
预测 cost / 地面高度，与全部可用 GT 比较
    ↓
每 10 epoch 固定验证 + 新 terrain seed 测试
```

## 1. 服务器现有环境

以下配置由用户于 2026-11-06 提供。场景生成、采集、标签、训练和绘图统一使用现有 `rl_env_glocal2`。

| 项目 | 配置 |
| --- | --- |
| 工程目录 | `/kpfs-intern/jialongliu/CJH_ws/glocal2-nav` |
| Conda 环境 | `/opt/conda/envs/rl_env_glocal2` |
| 操作系统 | Ubuntu 22.04.5 |
| Python | 3.11.17 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| PyTorch CUDA / cuDNN | 12.8 / 9.10.2 |
| NVIDIA 驱动 | 565.57.01 |
| Warp / Gymnasium | 1.18.0+cu12 / 1.2.1 |
| NumPy | 1.26.4 |
| 已安装 Isaac Sim / Isaac Lab | 5.1.0.0 / 0.54.2 |
| GPU | 4 × H100 80GB |
| CPU / 内存 | 双路 Xeon Platinum 8468，192 逻辑核 / 600 GiB |

采集直接调用 Warp `mesh_query_ray`，训练使用 PyTorch/CUDA，图片使用 matplotlib Agg。整个服务器流程不启动 Isaac Sim、Isaac Lab、Kit 或窗口。本机 Isaac 查看器作为独立工具保留。

## 2. 激活环境、安装项目

在服务器更新本次源码后执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate rl_env_glocal2
cd /kpfs-intern/jialongliu/CJH_ws/glocal2-nav

export SERVER_PY="$CONDA_PREFIX/bin/python"
export STAIR_NAV_PYTHON="$SERVER_PY"
export MPLBACKEND=Agg
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
# 0 为示例，填写本次实验分配的物理 GPU 编号。
export CUDA_VISIBLE_DEVICES=0

"$SERVER_PY" -m pip install \
  -r src/ptv3_tomography/requirements-capture.txt \
  -r src/ptv3_tomography/requirements-server.txt

"$SERVER_PY" -m pip install --no-build-isolation --no-deps \
  -e third_party/PointTransformerV3 \
  -e src/ptv3_tomography

./scripts/environment/build_native.sh
```

依赖安装补齐 Warp、trimesh、matplotlib、Pillow 和 manifold3d ≥ 3.5.4 等项目依赖，沿用服务器现有 Torch/CUDA。原生规划库由服务器当前 Python 编译。源码和配置通过 Git 上传，原生构建产物在服务器生成。

## 3. 新建实验并重建基线

下面所有相对路径均从工程根目录解释。选择一个新的实验目录，后续步骤使用同一个 `RUN_DIR`。

```bash
export RUN_DIR="runs/ptv3_structural_dr_from_scratch_20261106"
export BASE_SCENE="$RUN_DIR/assets/base_scene"
export BASE_ROUTES="$RUN_DIR/assets/base_routes"

"$SERVER_PY" third_party/vst_global_planner/scripts/scenes/generate_training_platform.py \
  --glocal2-root "$PWD" \
  --config third_party/vst_global_planner/configs/training_platform.json \
  --seed 20261104 --output "$BASE_SCENE" --tomography

"$SERVER_PY" third_party/vst_global_planner/scripts/planning/generate_random_routes.py \
  "$BASE_SCENE" --glocal2-root "$PWD" \
  --seed 20261104 --per-region 2 --output "$BASE_ROUTES"
```

基线包括曲桥、斜坡、开放楼梯、折返楼梯、螺旋楼梯、高台、桥洞、障碍物和矮墙九区域，每区域两条路线。VST 通过当前代码重建，包含最高地面层的顶部哨兵 slice 修复。

## 4. 启动完整采集、训练与测试

```bash
nohup "$SERVER_PY" isaac_training/scripts/server/run_playground_campaign.py \
  --campaign-root "$RUN_DIR" \
  --base-scene "$BASE_SCENE" --base-routes "$BASE_ROUTES" \
  --scene-python "$SERVER_PY" \
  --capture-python "$SERVER_PY" \
  --train-python "$SERVER_PY" \
  --training-config src/ptv3_tomography/configs/perception_pretrain_h100_batch.json \
  --parallel-captures 8 --capture-batch-frames 64 --capture-writers 8 \
  > "$RUN_DIR/campaign.log" 2>&1 &

printf '%s\n' "$!" > "$RUN_DIR/campaign.pid"
```

此命令生成新训练数据并随机初始化模型，不传入旧 campaign 或 checkpoint。调度器先准备数据，再运行 10 段训练，每段 10 epoch，共 100 epoch。段间使用同一套优化器、OneCycle 和训练进度。

当前配方使用所选的一张 H100，进程内 `cuda:0` 对应 `CUDA_VISIBLE_DEVICES` 指定的物理卡。

### 数据与域随机化

| 项目 | 设置 |
| --- | --- |
| 训练场景 | 重建的基线 seed 20261104 + 7 个新 seed |
| 采集路线 | 每场景 9 区域 × 2 条独立随机路线 × 2 传感器配置，共 288 组 |
| 起终点 | 从各设施完整 VST 可通行地面均匀采样，覆盖全部有效高度层；不使用固定锚点 |
| A* 阈值 | 15，VST 障碍代价保持 60 |
| 场地 / 楼梯总高度 | 56 × 48 m / 1.8–2.8 m，踏步高度由总高度和级数共同决定 |
| 开放楼梯 | 12–22 级，踏步深度 0.25–0.60 m，基准宽度 0.8–2.8 m；中心线转角 −80°–80° |
| 楼梯沿程宽窄 | 开放楼梯起点 / 中段 / 终点宽度倍率分别为 0.55–1.50 / 0.40–1.40 / 0.55–1.50，分段线性变化；踏板和侧梁一起变化 |
| 折返楼梯 | 每段 6–12 级，基准宽度 0.8–2.0 m，沿程宽度倍率 0.50–1.40；两段踏步深度分别为 0.26–0.38 / 0.28–0.40 m |
| 旋转楼梯 / 实心楼梯 | 24–36 / 10–16 级；旋转梯外半径 2.0–3.2 m；实心梯深度 0.30–0.45 m、宽度 1.0–1.7 m |
| 平台 | 各平台独立采样宽度和深度；独立大平台为 3.4–8.0 × 3.0–7.0 m；梯顶平台按楼梯末端朝向连接 |
| 高台挖空 | 等概率选择完整板、矩形孔、椭圆孔、双孔、边缘缺口；孔尺寸基于平台宽深的 25%–55%，双孔分别缩小并左右分布；位置随 seed 偏移 |
| 曲桥 | 宽度 0.85–1.40 m，正弦弯曲幅值 0.75–1.65 m 并沿用 ±8% 细扰动 |
| 障碍 | 8–16 个；长方体、椭圆柱体、楔形；XY 尺寸 0.35–1.10 / 0.35–0.95 m，高度 0.25–1.40 m，位置和 yaw 随 seed 变化 |
| 布局 | 各设施独立朝向并重新摆放，保留原有区域隔离和最小间距 |
| 安装位置 | 名义 `[0.2,0,0.08] m`，各轴 ±0.02 m |
| 安装姿态 | roll/yaw ±3°，pitch 35–55° |
| 载体高度 / 速度 | 0.35–0.45 m / 0.4–0.8 m/s |
| 扫描 | 10 Hz，射线数从 250/500/1000 抽样，范围 0.1–10 m |
| 帧数 | 按速度覆盖每条完整路线 |
| 实现 | 纯 CUDA Warp 多帧查询，压缩写入与下一批查询重叠 |

随机化范围位于 `third_party/vst_global_planner/configs/training_platform.json` 的 `geometry_randomization`。场景生成和调度器共用 `domain_randomization.sample_geometry`，同一个场景 seed 得到同一组物理参数。调度器将参数冻结到 `protocol.json`，生成器将实际参数、孔洞和障碍形状保存到 `config.resolved.json` / `scene.json`。`platform_openings` 保存孔型、尺寸和世界中心，楼梯元数据保存转角、逐级梯宽和中心线。关闭该配置的 `enabled` 可生成名义参数场景。

孔洞实际切穿平台板，底层地面保留。弯曲楼梯保留薄踏板和连续侧梁，平台顶板按终端轮廓整体生成。VST 和 Warp 均查询这些实际网格。布尔合并使用双精度并保留中间实体至最终导出。

几何随机化发生在采集前，每个场景生成一次，再重建对应 VST、路线、点云和标签。训练阶段读取固定数据，不会在每个 epoch 重建场景。旧 `protocol.json` 和已完成场景会被调度器复用，因此采用上述新 `RUN_DIR` 开始这次几何随机化实验。

起终点候选满足原生地面有效且 `0 ≤ cost ≤ 15`，同一地面的重复层表示只计一次。每条路线独立抽取一对起终点，保留原有最小间距、A*/GPMP 和网格几何检查。两条路线不再限定为固定入口到出口及其反向。

新采样规则只用于新生成的路线。重新采集时使用新的实验目录，避免调度器复用旧实验中已完成的路线任务。

`parallel-captures` 是一个进程组织的路线数，`capture-batch-frames` 是一次 GPU 查询的帧数，`capture-writers` 是压缩写入线程数。每帧沿用 MID-360 方向表与外参，沿规划路线设定位姿后采集。随机 seed 和参数首次生成后写入本次 `protocol.json`。

## 5. 监督与训练配方

教师来自 `vst.pickle` 的原生分层数组：`data[0]` 为 cost，`data[3]` 为地面高度。通过完整 LiDAR 位姿查询并转换到局部网格。

```python
mask_cost = target_available.copy()
mask_height = target_available.copy()
```

所有可用原生标签参与 Loss，包括当前扫描没有击中的网格。高度连续、地表回波匹配和邻域覆盖三个条件已删除。未知标签留空，局部高度带的选层沿用现有坐标映射规则。

模型输入为当前帧原始 XYZ 和重力 up。地面高度定义为 `h = ground_world_z - lidar_world_z`。世界位姿、教师地图和 mask 用于标签与 Loss。

| 参数 | 本次配置 |
| --- | --- |
| 配方 | `src/ptv3_tomography/configs/perception_pretrain_h100_batch.json` |
| 初始化 | 完整 PTv3 与地图头均为随机参数 |
| 输出 | cost、height 各 `[B,6,64,64]`，分辨率 0.1 m |
| 高度带边界 | `[-2,-1.5,-1,-0.5,0,0.5,1] m` |
| 真实 batch | 32，梯度累计 1 |
| 点数 / 体素 | 最多 1000 个有效输入点 / 0.1 m |
| DataLoader | 8 workers，prefetch=2，persistent workers，pin memory |
| 精度 / 后端 | BF16，attention/conv/layer_norm 均为 auto |
| 地图头 | 三近邻插值，隐藏维度 128，query chunk=2048 |
| 优化器 | AdamW，weight decay=0.05，gradient clip=1.0 |
| 学习率峰值 | Block 参数 1e-4，其他骨干参数及地图头 1e-3 |
| 调度 | OneCycle，100 epoch，warmup fraction=0.05 |
| Loss | cost/60 与 height 各用 SmoothL1，beta=0.1，权重均为 1 |
| 评估 batch | 8 |

多帧通过官方 collate/offset 合批，一次前向、反向和参数更新。Loss 按各帧可用格归一化。输入点数预算与监督范围分别处理，缓存保存原始 cost、高度，加载时直接生成标签可用性 mask。

## 6. 分段测试与模型选择

| 时机 | 数据 | 用途 |
| --- | --- | --- |
| 每 10 epoch | 固定 3 个 terrain-generator seed | 对比各段结果，选择 best.pt |
| 每 10 epoch | 该段新生成的 2 个独立 terrain-generator seed | 观察新场景表现 |
| 100 epoch 结束 | 额外 5 个 terrain-generator seed | 使用选定 best.pt 做最终测试 |

terrain-generator 沿用当前集成的 ETH 楼梯组件及 seed 障碍布局。测试场景各自重建 VST、采集点云和配对标签。

记录 Cost MAE、地面高度 MAE和危险代价低估率。危险低估定义为 `GT cost >= 50 且预测 cost < 30`。best.pt 按固定验证分数 `Cost MAE / 60 + Height MAE（m）` 选择。

每次评估通过 matplotlib Agg 保存三个代表帧的 GT、预测、绝对误差图，覆盖六个高度带，并更新训练曲线。

## 7. 输出与查看

```text
runs/ptv3_structural_dr_from_scratch_20261106/
├── best.pt / last.pt
├── campaign.log / campaign.pid
├── protocol.json / registry.json
├── configs/
├── assets/                    # 本次重建的基线场景和路线
├── cache/                     # 本次训练数组
├── logs/                      # train.log、epochs.json、状态和初始化记录
├── plots/training_metrics.png
├── evaluations/               # 固定、新 seed、最终测试指标和 PNG
└── .work/                     # 本次采集、标签与 portable 数据
```

权重只保留顶层的 `last.pt` 和 `best.pt`。`logs/initialization.json` 记录本次初始模式为 `random`。

```bash
tail -f "$RUN_DIR/campaign.log"
tail -f "$RUN_DIR/logs/train.log"
```

曲线位于 `plots/training_metrics.png`，逐帧图片位于 `evaluations/<阶段>/seed_<seed>/attempt_*/plots/`。汇总指标位于 `evaluations/stage_*_fixed.json`、`stage_*_fresh.json` 和 `final_holdout.json`。

本次新实验中断后，重新激活环境并执行第 4 节相同命令，调度器复用本次已完成的数据并接续本次 `last.pt`。完整状态包含模型、优化器、学习率调度、RNG、epoch 和样本游标。

源码和配置进入 Git，场景、数据、缓存、权重和图片保存在忽略的 `runs/`。

## 8. 本地执行记录

2026-11-06 按本文在新目录 `runs/server_doc_short_from_scratch_20261106` 从零生成基线 seed 20261104，完成九区域 VST 和 18/18 条随机路线。调用同一个调度器的 `--smoke` 模式，生成 20 帧训练点云，完成原生 VST 配对、portable 打包和训练 mmap 缓存。独立 terrain seed 的另 20 帧也完成采集、配对和打包。

训练集 20 帧 GPU 采集循环耗时 0.077 s，包含压缩写入等待，不含初始化。标签配对共 0.852 s，其中查询 0.601 s、压缩保存 0.185 s。训练缓存生成耗时 0.141 s。全部 40 帧的 `mask_cost` 和 `mask_height` 均与 `target_available` 一致。

本地复用原有几何、Warp 和 PTv3 环境，服务器仍按第 2 节使用统一的 `rl_env_glocal2`。本次按用户要求以数据准备验收结束。收到停止要求前已完成 1 epoch / 1 次更新，后续训练和评估已停止。完整短训练和最终 holdout 未完成。

记录位于 `runs/server_doc_short_from_scratch_20261106/data_preparation_summary.json`。该记录验证数据准备链路，服务器正式实验使用新的运行目录。
