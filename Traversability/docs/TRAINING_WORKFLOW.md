# Traversability 冻结 Small 完整训练与服务器迁移指南

更新日期：2026-11-13。所有命令从 Pure3D_ws 根目录执行，默认使用 Bash。

本文只说明完整三维输入、冻结 Utonia Small 预训练编码器的实验，包含服务器环境查询与安装、数据生成、运行命名、恢复、评估和推理。编码器加载官方预训练权重并始终冻结，Adapter、FiLM 和 CostHead 随机初始化后训练。文中命令供实际执行时使用，本次文档整理没有启动数据生成或训练，也没有连接服务器。

## 1. 训练路线与数据契约

```text
WFC 独立场景 seed → Mesh → 3 cm 表面采样 → 完整建筑 terrain.pcd
  ├─ 原生 VST 几何与多组机器人条件 → Tomogram 支撑 XYZ、cost/50、条件配对
  └─ CUDA 5 cm 体素代表点 → 完整三维上下文 → Utonia Small → 972 维特征
                          ↑
          GPU 体素哈希建立 Tomogram 查询到真实代表点的对应
                          ↓
       Adapter 972→256 → 四参数 FiLM → CostHead → sigmoid cost
                          ↓
             核心区域监督：绝对代价损失 + 条件差值损失
```

完整三维输入保留地面、台阶立面、墙体和天花板。每个世界系 5 cm 体素选择原始编号最小的真实点。编码按三维空间窗口执行，核心区为 4×4×3 m，外扩 1 m 上下文。上下文参与编码，每个 Tomogram 查询只在所属核心区计算一次损失。

对应关系为 `Tomogram 行号 → 5 cm 真实代表点 → 原始 PCD 行号 → Small 体素 → 特征行`。GPU 哈希按空间邻域扩展搜索精确最近代表点，同距选择原始编号较小者。Tomogram 查询点不加入编码器输入。该对应阶段不建立全楼 CPU KD-tree。

| 项目 | 固定约定 |
|---|---|
| 坐标 | 地图坐标系，XYZ 单位为米 |
| 学生条件 | `[clearance_min, slope_max, step_max, stand_ratio]` |
| 条件量纲 | 净空 m，坡度 rad，踏步 m，支撑比例无量纲 |
| 教师条件 | `[clearance_min, slope_max, step_max, interval_free, stand_ratio]` |
| 教师常量 | `interval_free=0.5 m`，原生距离曲线 `p=2`，障碍代价 50 |
| 标签 | 原生 Tomogram 的全部有效支撑查询，cost 除以 50，范围 0–1 |
| Small | `utonia_small`，972 维，`scale=0.5`，`grid_size=0.01` |
| 采样层级 | 5 cm 是输入几何体素，Small 内部网格对应约 2 cm 物理尺度 |
| 冻结训练 | Small 为 eval 且冻结，每个新上下文编码一次，查询特征以 FP16 缓存 |

Small 编码使用 BF16，任务头与损失使用 FP32。任务头沿用 `models/sliced.py` 的 `SlicedCostHead`，这个类名不代表完整三维路径使用切片几何作为编码输入。

当前 `inference.full3d` 先从输入 PCD 生成原生几何支撑查询，再用完整三维上下文预测。输出 `prediction.pcd` 的 XYZ 是该支撑点集，并非输入 PCD 全部原始点。推理需要 CuPy 几何环境，不读取教师 cost，也不生成多条件监督标签。

## 2. 目录与入口

| 路径 | 用途 |
|---|---|
| `Traversability/configs/` | 可复用模板与每次实验的配置 |
| `Traversability/data/full3d.py` | 冻结路径的数据生成与固定集引用 |
| `Traversability/data/rotating.py` | WFC、表面采样、条件监督与轮换清理 |
| `Traversability/training/full3d_cache.py` | GPU 对应与完整三维冻结特征缓存 |
| `Traversability/training/full3d.py` | 冻结训练及轮末恢复 |
| `Traversability/experiments/full3d_rotating.py` | 冻结训练、真实恢复、独立推理与报告编排 |
| `Traversability/inference/full3d.py` | 新建筑独立推理 |
| `datasets/<RUN_NAME>/` | 本次场景、标签、索引和缓存 |
| `runs/traversability/<RUN_NAME>/` | 本次权重、日志、指标和报告 |

唯一主线模板为 `Traversability/configs/full3d_small_frozen.json`。Small 加载 `weights/utonia_small.pth`，特征宽度为 972，不依赖旧任务 checkpoint。历史训练文件继续保留，当前指南与新配置只使用冻结 Small 路线。

## 3. runs 命名规则

新实验统一采用：

```text
runs/traversability/cost_<input>_<encoder>_<mode>_e<epochs>_s<seed>_<YYYYMMDD-HHMMSS>_<tag>/
```

| 字段 | 填写方式 |
|---|---|
| `input` | `full3d`，完整三维几何输入 |
| `encoder` | `small` |
| `mode` | 固定为 `frozen` |
| `epochs` | 实际目标轮数，如 `e50`，对应 `stop_epoch` |
| `seed` | 配置的全局随机种子，如 `s2026111400`。场景起始 seed 另记配置 |
| 时间 | 按 `Asia/Shanghai`，精确到秒 |
| `tag` | `server01`、`baseline`、`ablation01` 等，仅使用字母、数字和下划线 |

示例：

```text
cost_full3d_small_frozen_e50_s2026111400_20261113-230000_server01
```

同一实验的名字同时用于：

```text
Traversability/configs/<RUN_NAME>.json
datasets/<RUN_NAME>/
runs/traversability/<RUN_NAME>/
```

恢复同一实验沿用原名和原目录。改变数据、初始化权重、冻结方式、损失或实验预算时使用新名字，并记录来源。历史目录保持原名，通过 `summary.json` 读取真实完成轮数。

`epochs` 是学习率调度预算，`stop_epoch` 是实际停止轮数。本模板使用 `epochs=100, stop_epoch=50`。新建完整 100 轮实验时设两者为 100。新建 50 轮完整余弦周期时设两者为 50，这与默认模板的学习率轨迹不同。

## 4. 查询服务器环境

先登录服务器，在计划存放工程的磁盘上查询。替换服务器地址和目录，下面命令均在服务器运行。

```bash
export WORKSPACE=/data/Pure3D_ws
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

uname -a
cat /etc/os-release
lscpu
free -h
df -h . /tmp
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
command -v python3
python3 --version
command -v conda || true
command -v gcc || true
command -v g++ || true
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
fi
```

记录 GPU 型号、空闲显存、驱动、CPU、内存和磁盘。`nvidia-smi` 的 CUDA 数字表示驱动支持能力，PyTorch 实际构建版本由 `torch.version.cuda` 查询。NVCC 信息与 PyTorch/CuPy 的运行检查分别记录。

当前代码按单 GPU 运行，尚无 DDP 启动入口。用 `CUDA_VISIBLE_DEVICES` 指定物理卡后，配置保持 `device=cuda:0`。训练、教师生成和几何对应均需要 CUDA。显存需求随场景与上下文点数变化，先检查固定集编码和首轮峰值，再决定服务器预算。

本机可读取的参考环境如下，版本来自当前安装查询：

| 环境 | Python | 关键依赖 |
|---|---|---|
| 网络与 WFC | 3.11.17 | torch 2.10.0+cu128、Triton 3.6.0、NumPy 1.26.4、SciPy 1.15.3、point-transformer-v3 0.1.2 |
| 几何与教师 | 3.10.12 | CuPy CUDA12x 13.6.0、NumPy 1.24.4、SciPy 1.8.0、trimesh 4.12.2 |

本机存在 `datasets/cache/wfc_runtime/` 补充包，所以单独导出 `pip freeze` 不包含全部 WFC 依赖。新服务器按下一节安装补齐，并运行导入检查。参考版本不代表服务器已通过验证。

## 5. 迁移代码、初始化权重和固定数据

### 5.1 迁移清单

默认路线是迁移训练所需资产，在服务器创建全新的运行。必需资产如下：

| 资产 | 原因 |
|---|---|
| `Traversability/`、`common/geneator/`、`common/pointcloud_io/`、`common/vst_global_planner/`、`utils/` | 训练、WFC、点云 I/O、教师、编码器源码 |
| `environment-ptv3.yml` | 本地网络环境参考，尚需补齐 WFC 与教师依赖 |
| `datasets/cost_sliced_tiny_local50_20261113/` | 原固定 val/test 的 manifest、数组、核心划分和条件 |
| `datasets/wfc_capability_dr_v2/scene_20261113/`、`scene_20261114/` | 固定集 manifest 引用的原始 `terrain.pcd` 和相邻 `scene.json` |
| `utonia_small.pth` | 冻结特征缓存读取，官方 Small 预训练编码器 |

固定数据目录沿用历史名称，其中的 `tiny` 只属于目录名。本实验复用几何、标签、条件和核心划分，特征全部由 Small 重新生成。

固定数组包括 `coord.npy`、`cost.npy`、`conditions.npy`、`facility.npy`、`layer_cell.npy`，以及 `block_*.npz`、`ready.json`。已有 `features/` 可跳过，完整三维路径会重新编码。

只复制 Git 源码会缺少忽略的大文件。恢复任务只复制 `best.pt` 会缺少数据、条件和路径元数据。

### 5.2 在本机传输

以下命令在原工作站执行。`SERVER` 和 `SERVER_ROOT` 替换为真实值。

```bash
cd /home/cjh/Pure3D_ws
SERVER=user@server
SERVER_ROOT=/data/Pure3D_ws
git rev-parse HEAD > /tmp/traversability_source_commit.txt
git status --short > /tmp/traversability_source_status.txt
ssh "$SERVER" "mkdir -p '$SERVER_ROOT/weights'"

rsync -aL --relative --info=progress2 \
  --exclude='.git/' --exclude='__pycache__/' --exclude='features/' \
  Traversability/ common/geneator/ common/pointcloud_io/ \
  common/vst_global_planner/ utils/ environment-ptv3.yml \
  datasets/cost_sliced_tiny_local50_20261113/ \
  datasets/wfc_capability_dr_v2/scene_20261113/ \
  datasets/wfc_capability_dr_v2/scene_20261114/ \
  "$SERVER:$SERVER_ROOT/"

UTONIA_WEIGHTS=/home/cjh/Pure3D_ws/weights/utonia_small.pth
rsync -aL "$UTONIA_WEIGHTS" "$SERVER:$SERVER_ROOT/weights/utonia_small.pth"
rsync -a /tmp/traversability_source_commit.txt /tmp/traversability_source_status.txt \
  "$SERVER:$SERVER_ROOT/"
sha256sum "$UTONIA_WEIGHTS"
```

`-L` 将所选源符号链接复制为实际内容，避免服务器继续引用本机绝对路径。迁移到已有服务器工程时，先确认目标代码版本。此命令用于独立目标工程，没有清理目标目录的参数。

## 6. 安装服务器环境并检查 CUDA

### 6.1 网络环境与 WFC 依赖

下面以本地使用的 PyTorch CUDA 12.8 构建为例。先核对服务器驱动，再选择可运行的 PyTorch CUDA 构建。保留 `torch>=2.10` 的源码要求。

```bash
cd "$WORKSPACE"
conda create -n pure3d-ptv3-server python=3.11 pip -y
conda run -n pure3d-ptv3-server python -m pip install \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n pure3d-ptv3-server python -m pip install \
  'numpy==1.26.4' scipy trimesh manifold3d networkx pyyaml \
  'opencv-python-headless==4.11.0.86' matplotlib pillow \
  alive-progress perlin-numpy rtree open3d
conda run -n pure3d-ptv3-server python -m pip install -e ./utils/PointTransformerV3

export PTV3_PY="$(conda run -n pure3d-ptv3-server python -c 'import sys; print(sys.executable)')"
export GENERATOR_PY="$PTV3_PY"
```

WFC 的上游源码位于 `common/geneator/terrain-generator/`，包装入口自动加入导入路径。其工具模块导入 Open3D，但场景生成命令本身可 headless 运行。保留上述显式依赖，避免使用 `requirements-wfc.txt` 安装其固定的 CPU torch，覆盖网络环境。

当前 vendored PTv3 使用 PyTorch 与 Triton 实现，主线不需要额外安装 spconv、torch_scatter 或外部 flash-attn。

### 6.2 几何与教师环境

```bash
python3 -m venv "$WORKSPACE/.venvs/trav-geometry"
export GEOMETRY_PY="$WORKSPACE/.venvs/trav-geometry/bin/python"
"$GEOMETRY_PY" -m pip install --upgrade pip
"$GEOMETRY_PY" -m pip install \
  'numpy==1.26.4' scipy trimesh manifold3d matplotlib pillow pyyaml \
  'opencv-python-headless==4.11.0.86' 'cupy-cuda12x==13.6.0'
```

以上为依照代码导入整理的安装步骤，干净服务器上的实际兼容性由下面的检查确定。使用不同 CUDA 系列时匹配 PyTorch 和 CuPy 构建。ROS、RViz 与显示服务只用于可视化。

### 6.3 公共变量与导入检查

后续新终端或 tmux 会话都需要设置这些变量，解释器填写服务器真实路径。

```bash
export WORKSPACE=/data/Pure3D_ws
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE:$WORKSPACE/common/vst_global_planner/src${PYTHONPATH:+:$PYTHONPATH}"
export TZ=Asia/Shanghai
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORKSPACE/datasets/cache/matplotlib"
export CUPY_CACHE_DIR="$WORKSPACE/datasets/cache/cupy"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR" "$CUPY_CACHE_DIR"

"$PTV3_PY" - <<'PY'
import sys
import torch
import triton
import point_transformer_v3
import numpy, scipy, trimesh, networkx, cv2, matplotlib
print('python:', sys.executable)
print('torch:', torch.__version__, 'runtime CUDA:', torch.version.cuda)
print('triton:', triton.__version__)
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
assert torch.cuda.is_bf16_supported(), 'Current encoding path requires BF16 support'
print('GPU:', torch.cuda.get_device_name(0))
a = torch.ones(1024, device='cuda')
assert a.sum().item() == 1024
PY

"$GENERATOR_PY" common/geneator/generate_wfc_multifloor.py --help

"$GEOMETRY_PY" - <<'PY'
import sys
import numpy, scipy, trimesh, cv2, matplotlib
import cupy as cp
from vst_global_planner.conditioned_teacher import ConditionedTeacher
print('python:', sys.executable, 'cupy:', cp.__version__)
cp.show_config()
a = cp.arange(16, dtype=cp.float32)
kernel = cp.ElementwiseKernel('float32 x', 'float32 y', 'y = x + 1', 'trav_env_probe')
assert float(kernel(a).sum()) == 136
print('CuPy kernel OK')
PY

"$PTV3_PY" -m pip check
"$GEOMETRY_PY" -m pip check
```

`generate_wfc_multifloor.py --help` 会验证 WFC 导入链，不生成场景。CuPy 检查编译一个小内核，用于发现 NVRTC、CUDA 库和设备访问问题。

## 7. 修正迁移后的固定数据路径

在服务器副本上执行。脚本只改固定数据中的 JSON 路径，不修改监督数组与核心划分。原本机工程保持原样。

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

workspace = Path(os.environ['WORKSPACE']).resolve()
old = '/home/cjh/Pure3D_ws'
root = workspace / 'datasets/cost_sliced_tiny_local50_20261113'

def relocate(value):
    if isinstance(value, str):
        if value == old or value.startswith(old + '/'):
            return str(workspace) + value[len(old):]
        return value
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    return value

for path in root.rglob('*.json'):
    original = json.loads(path.read_text())
    revised = relocate(original)
    if original != revised:
        path.write_text(json.dumps(revised, ensure_ascii=False, indent=2) + '\n')

meta = json.loads((root / 'manifest.json').read_text())
fixed = [s for s in meta['scenes'] if s['split'] in ['val', 'test']]
assert {s['split'] for s in fixed} == {'val', 'test'}
for scene in fixed:
    pcd = Path(scene['source_pcd'])
    assert pcd.is_file(), pcd
    assert pcd.with_name('scene.json').is_file()
    for name in ['coord', 'cost', 'conditions', 'facility', 'layer_cell']:
        assert (root / scene['id'] / (name + '.npy')).is_file()
    for block in [b for b in meta['blocks'] if b['scene'] == scene['id']]:
        assert (root / scene['id'] / (block['id'] + '.npz')).is_file()
    print(scene['split'], scene['id'], pcd)
PY

sha256sum weights/utonia_small.pth
```

Small 权重大小为 149082467 字节，官方预设 SHA256 为 `3d7afb731771556ebdf20abd1b6a1f022d9b9f105db8446f5eff02037f7fceb9`。先比较文件哈希，再运行训练。恢复还会核对 checkpoint 中冻结编码器与预训练权重张量相等。

## 8. 创建新实验配置

### 8.1 默认 50 轮冻结实验

下面创建默认冻结实验，实际 50 轮、100 轮调度预算、每 10 轮替换四栋场景。服务器重新生成训练数据，固定验证测试保持原始数据。

```bash
cd "$WORKSPACE"
export RUN_NAME="cost_full3d_small_frozen_e50_s2026111400_$(date +%Y%m%d-%H%M%S)_server01"
export RUN_CONFIG="$WORKSPACE/Traversability/configs/$RUN_NAME.json"
export RUN_DIR="$WORKSPACE/runs/traversability/$RUN_NAME"

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['WORKSPACE']).resolve()
name = os.environ['RUN_NAME']
path = Path(os.environ['RUN_CONFIG'])
cfg = json.loads((root / 'Traversability/configs/full3d_small_frozen.json').read_text())
cfg.update(data=f'datasets/{name}', output=f'runs/traversability/{name}',
           generator_python=os.environ['GENERATOR_PY'],
           geometry_python=os.environ['GEOMETRY_PY'],
           neural_python=os.environ['PTV3_PY'],
           seed=2026111400, epochs=100, stop_epoch=50,
           batch_size=2, num_workers=2, cpu_threads=4, device='cuda:0')
cfg['encoder'].update(name='utonia_small', weights=str(root / 'weights/utonia_small.pth'),
                      feature_channels=972, frozen=True)
assert not path.exists(), path
assert not (root / cfg['data']).exists(), cfg['data']
assert not (root / cfg['output']).exists(), cfg['output']
for key in ['generator_python', 'geometry_python', 'neural_python']:
    assert Path(cfg[key]).is_file(), cfg[key]
path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
print(path)
PY

mkdir -p "$RUN_DIR"
nvidia-smi > "$RUN_DIR/environment_gpu.txt"
"$PTV3_PY" -m pip freeze > "$RUN_DIR/environment_neural.txt"
"$GEOMETRY_PY" -m pip freeze > "$RUN_DIR/environment_geometry.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
  git status --short > "$RUN_DIR/source_status.txt"
else
  cp "$WORKSPACE/traversability_source_commit.txt" "$RUN_DIR/source_commit.txt"
  cp "$WORKSPACE/traversability_source_status.txt" "$RUN_DIR/source_status.txt"
fi
cp "$RUN_CONFIG" "$RUN_DIR/requested_config.json"
```

源码存在未提交改动时，提交号与 `source_status.txt` 一起记录，迁移使用的实际代码也应保存副本。训练还会在运行目录写入实际 `config.json`。

### 8.2 关键参数

| 字段 | 默认值与用途 |
|---|---|
| 初始化 | 官方 Small 预训练编码器，随机任务头，无 `source_run` |
| `fixed_data` | 原始固定验证/测试数据目录 |
| `rotation.scenes_per_generation` | 4 |
| `rotation.epochs_per_generation` | 10 |
| `rotation.seed_start` | 2026111400 |
| `rotation.surface_spacing_m` | 0.03 |
| `geometry_voxel_size_m` | 0.05 |
| `batch_size` | 2 个真实窗口 |
| `learning_rate` / `minimum_learning_rate` | 1e-4 / 1e-6 |
| `warmup_fraction` | 调度预算前 5% 预热，然后余弦下降 |
| `weight_decay` | 0.05 |
| `smooth_l1_beta` | 0.1 |
| `condition_difference_weight` | 2.0 |
| `change_threshold` | 0.02 |
| `parameter_weights` | 四参数等权 `[1,1,1,1]` |
| `region_balanced` | 设施与背景等权 |

损失为绝对代价 SmoothL1 加两倍条件差值损失。条件差值分别平衡教师发生变化和保持稳定的情况。训练条件范围为净空 0.2–1.0 m、坡度 0.18–0.70 rad、踏步 0.08–0.55 m、支撑比例 0.02–0.50。

训练逐轮交替使用两套 40 槽条件库，同一几何共享坐标、对应索引与冻结特征。最后不足一个 batch 的核心块按现有实现重复前面的块补齐。

本配置不填写 `source_run`。训练从官方 Small 预训练权重创建冻结编码器，972→256 的 Adapter、四参数 FiLM 和 CostHead 使用新的随机参数。旧任务头权重不参与初始化。Small 的真实预训练权重已经下载并通过哈希核验。

## 9. 生成训练数据与固定特征缓存

### 9.1 自动生成顺序

正式训练入口自动执行下面流程，也支持先生成第 0 批检查数据：

1. 读取 WFC 域随机化模板 `common/geneator/wfc_capability_dr.json`，按独立 seed 生成多层建筑 Mesh。
2. 按 3 cm 对全部三角面采样，生成 `terrain.pcd`。
3. CuPy 原生教师生成几何与各条件的 Tomogram cost，所有条件共用 XYZ。
4. 保存设施掩码、条件配对、核心块索引，验证标签范围与共享行序。
5. CUDA 选择 5 cm 真实代表点，GPU 哈希建立监督对应。
6. Small 编码三维上下文，缓存被核心查询引用的特征。

默认第 0 批 seed 为 2026111400–2026111403，第 1 批为 2026111404–2026111407。50 轮使用五批共 20 栋训练建筑，固定验证为 `scene_20261113`，固定测试为 `scene_20261114`。

### 9.2 手动预生成第 0 批

这是正式实验目录中的准备步骤。一次只准备当前批次，避免提前切换生成编号导致现有 active 数据被清理。

```bash
"$GEOMETRY_PY" -u -m Traversability.data.full3d \
  --config "$RUN_CONFIG" --generation 0 \
  > "$RUN_DIR/prepare_00.log" 2>&1

"$PTV3_PY" -u - <<'PY' > "$RUN_DIR/cache_00.log" 2>&1
import json
import os
import torch
from pathlib import Path
from Traversability.training.full3d_cache import cache_features

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
torch.set_num_threads(cfg['cpu_threads'])
cache_features(cfg)
PY
```

`full3d_cache.py` 提供 Python 函数，没有独立的 `--config` 命令行入口，因此这里显式调用 `cache_features`。正式训练将识别已完成的上下文缓存。

### 9.3 数据检查

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
root = Path(os.environ['WORKSPACE']) / cfg['data']
meta = json.loads((root / 'manifest.json').read_text())
for scene in meta['scenes']:
    folder = root / scene['id']
    xyz = np.load(folder / 'coord.npy', mmap_mode='r')
    cost = np.load(folder / 'cost.npy', mmap_mode='r')
    cond = np.load(folder / 'conditions.npy', mmap_mode='r')
    assert cost.shape == (len(xyz), len(cond))
    assert cond.shape[1] == 4
    assert np.isfinite(xyz).all() and np.isfinite(cost).all()
    assert cost.min() >= 0 and cost.max() <= 1
    report = json.loads((folder / 'full3d_encoding.json').read_text())
    assert report['full_geometry_input'] and not report['cpu_kdtree_used']
    assert report['supervised_queries'] == len(xyz)
    print(scene['id'], scene['split'], xyz.shape, cost.shape,
          'max_match_m=', report['matching']['maximum_m'],
          'peak_GiB=', report['peak_cuda_gib'])
PY
```

检查 `scene_archive/scene_*/overview.png`、`config.resolved.json`、`data_summary.json` 和匹配报告，确认楼层、楼梯实际尺寸、标签和对应距离。固定集数组由原数据引用，编码特征在新目录重建。

### 9.4 轮换和磁盘

```text
datasets/<RUN_NAME>/
├── manifest.json
├── generation.json
├── encoding.json
├── fixed/scene_20261113/       固定标签引用及本次三维特征
├── fixed/scene_20261114/
├── active/source/             当前四栋 Mesh、PCD 与 WFC 缓存
├── active/prepared/           当前四栋监督、核心块与特征
└── scene_*/                   指向当前训练数据或 fixed 的链接
```

第 11、21、31、41 轮进入新批次。轮换会删除旧批次 Mesh、PCD、标签、特征和 WFC 临时缓存，保留运行目录中的轻量 `scene_archive/`、`generations/`、`rotations/`。固定验证测试及其源文件需要持续保留。

新数据实验使用新的训练 seed，避免与固定验证、测试、回归场景重叠。同一几何族的变体应归属同一划分。若重新生成验证测试，保存独立 `fixed_data` 版本和 manifest，本次结果与原固定集结果分别报告。

## 10. 启动、监控与恢复冻结训练

### 10.1 完整流水线

可在 tmux 中运行，保持前述环境变量：

```bash
"$PTV3_PY" -u -m Traversability.experiments.full3d_rotating \
  --config "$RUN_CONFIG" > "$RUN_DIR/pipeline.log" 2>&1
```

入口先训练一轮并真实退出，再加载本次 `last.pt` 继续到 `stop_epoch`，最后执行固定测试、独立 PCD 推理、图表及报告。

编排入口会依据 `last.pt` 和 `summary.json` 判断阶段。重新运行前核对 `execution.json` 中父子 PID，对照实际进程，确认上一任务已退出。同一输出目录同时运行两个进程会互相覆盖状态和缓存。

### 10.2 首轮试跑与直接恢复

需要先检查首轮更新时，可使用训练子入口：

```bash
"$PTV3_PY" -u -m Traversability.training.full3d \
  --config "$RUN_CONFIG" --stop-after-epoch 1 \
  > "$RUN_DIR/train_probe.log" 2>&1
```

核对 `first_update.json`、`history.json`、`resume_verification.json`、`last.pt`。Adapter、条件头和代价头应有梯度与参数变化，Small 应不在优化器中。首轮通过后，运行上一节编排命令即可恢复并完成后处理。

直接恢复训练子入口的命令为：

```bash
"$PTV3_PY" -u -m Traversability.training.full3d \
  --config "$RUN_CONFIG" --resume > "$RUN_DIR/train_resume.log" 2>&1
```

这里的 `--resume` 是布尔开关，checkpoint 固定读取 `config.output/last.pt`，无需在开关后传文件路径。当前完整三维训练按完整 epoch 结束保存状态，恢复模型、AdamW、随机状态、步数和学习率阶段，从下一 epoch 开始。未完成一轮的更新会重做，Ctrl+C 不保证额外保存轮中状态。

与旧 `training.train` 不同，完整三维入口读取本次 `--config`。恢复时保留原数据、输出、seed、损失、调度预算和条件布局，避免同一运行悄然改变协议。

### 10.3 查看进度

以下查看命令在另一个已设置变量的终端运行：

```bash
cat "$RUN_DIR/progress.json"
cat "$RUN_DIR/execution.json"
tail -n 60 "$RUN_DIR/pipeline.log"
tail -n 60 "$RUN_DIR/train_01.log"
tail -n 60 "$RUN_DIR/train_resume_to_50.log"
nvidia-smi
du -sh "$WORKSPACE/datasets/$RUN_NAME" "$RUN_DIR"
```

`train_resume_to_50.log` 名称随目标轮数变化。数据生成细节在 `generation_*.log` 和 `scene_archive/*/generation.log`。

### 10.4 迁移已有中断任务

迁移已有训练时，完整复制运行目录和对应数据目录，使用 `rsync -aL` 解引用数据链接，同时复制固定数据源、原始 PCD 和 Small 预训练权重。迁移前停止旧进程，在新服务器继续使用原运行名。

对目标副本执行以下修正，再恢复：

1. 把配置、manifest、各级 `ready.json`、`supervision.json` 等 JSON 中的旧工程根替换为新根，方法参照第 7 节。对数据和运行目录分别处理。JSON 字典的路径键也需同步修正。
2. 将运行配置中的 `generator_python`、`geometry_python`、`neural_python`、`encoder.weights` 改为服务器路径，检查所有 `source_pcd` 存在。
3. 保留 `history.json`、`initial_val.json`、`resume_verification.json`、`baseline.json`、条件响应及所有 checkpoint。仅有 `last.pt` 不足以恢复该流水线。
4. 对迁移副本的 `best.pt`、`last.pt` 更新其 `config` 中的运行环境路径，保持模型、优化器与随机状态张量不变。推理从 checkpoint 内读取环境配置。下面代码完成该元数据更新。

```bash
"$PTV3_PY" - <<'PY'
import json
import os
from pathlib import Path
import torch

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
root = Path(os.environ['WORKSPACE']).resolve()
output = root / cfg['output']
for name in ['best.pt', 'last.pt']:
    path = output / name
    state = torch.load(path, map_location='cpu', weights_only=False)
    old = state['config']
    for key in ['architecture', 'seed', 'epochs', 'batch_size', 'condition_names',
                'condition_scale', 'rotation', 'window', 'learning_rate',
                'minimum_learning_rate', 'condition_difference_weight']:
        assert old[key] == cfg[key], key
    state['config'] = cfg
    temp = path.with_suffix('.migrating.pt')
    torch.save(state, temp)
    temp.replace(path)
    print(name, 'preserved epoch', state['epoch'], 'step', state['step'])
PY
```

`RUN_CONFIG` 此时指向迁移任务的已修正配置。更新本次训练 checkpoint 的环境元数据会改变文件 SHA256，另记迁移前后的哈希。官方 Small 预训练文件保持原文件内容。

修正后使用本节编排入口继续，或使用训练子入口 `--resume`。如果 `summary.json` 已存在，编排会跳过训练并重做后处理。缺失的 active 数据可按相同 seed 重建，重建后仍需检查实际几何与监督的一致性。

## 11. 选优、测试、独立推理与报告

### 11.1 固定集选优

每轮按固定验证分数选择 `best.pt`：

```text
score = normalized cost MAE + mean four-parameter response error
```

完整三维初始化也参与选优，因此最佳轮次可以是 0。训练结束加载最佳权重，导出固定验证和测试预测。读取：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ['RUN_DIR'])
s = json.loads((run / 'summary.json').read_text())
print('completed epochs:', s['completed_epochs'])
print('selected epoch:', s['selected_epoch'])
for name in ['best_validation', 'test']:
    values = s[name]
    print(name, 'raw MAE:', values['cost_mae_raw'],
          'response error:', values['mean_response_error'])
PY
```

`cost_mae_raw = cost_mae × 50`。响应误差用于检查条件变化的方向与幅度。`predictions/scene_*_prediction.npy` 与固定场景 `coord.npy` 的行序一致，用同一行的 `cost.npy` 评分。

旧 `training.evaluate` 服务于 QueryPool 模型。完整三维 checkpoint 使用当前训练入口导出的 `summary.json`、验证及测试预测进行评价。

### 11.2 独立建筑推理

```bash
export INPUT_PCD="$WORKSPACE/datasets/wfc_capability_dr_v2/scene_20261114/terrain.pcd"
"$PTV3_PY" -u -m Traversability.inference.full3d \
  --checkpoint "$RUN_DIR/best.pt" --pcd "$INPUT_PCD" \
  --condition 0.5 0.6 0.4 0.05 \
  --output "$RUN_DIR/independent_prediction"
```

输出包括 `prediction.pcd`、`prediction.npy`、`prediction.json` 与 `geometry/`。条件顺序固定。独立推理测试结果可与缓存测试输出比较，需先确认 XYZ 完全相同。固定集条件 `[0.5,0.6,0.4,0.05]` 对应 `relaxed` 槽位，测试导出的单组预测使用该槽位。

更换为真实建筑时，先传输 PCD，再设置：

```bash
export INPUT_PCD="$WORKSPACE/datasets/rot_stairs/pcd_files/building_SUST.pcd"
"$PTV3_PY" -u -m Traversability.inference.full3d \
  --checkpoint "$RUN_DIR/best.pt" --pcd "$INPUT_PCD" \
  --condition 0.5 0.6 0.4 0.05 --output "$RUN_DIR/building_SUST"
```

独立推理从 checkpoint 的 `config.geometry_python` 调用几何环境，迁移后需完成第 10.4 节的元数据修正。

### 11.3 真建筑同点评价

SUST 使用单独生成并固定的原生 VST 规则参考，评价 XYZ 必须与预测完全一致。已有参考在 `runs/traversability/building_SUST_comparison_20261113/vst/`，迁移时另行复制并核对其条件、教师配置及输入 PCD 哈希。

参考有效时，可按相同点集计算：

```bash
export SUST_REFERENCE_DIR="$WORKSPACE/runs/traversability/building_SUST_comparison_20261113/vst"
"$GEOMETRY_PY" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

run = Path(os.environ['RUN_DIR'])
ref = Path(os.environ['SUST_REFERENCE_DIR'])
folder = run / 'building_SUST'
xyz = np.load(folder / 'geometry/coord.npy')
assert np.array_equal(xyz, np.load(ref / 'coord.npy'))
pred = np.load(folder / 'prediction.npy')
truth = np.load(ref / 'cost.npy')
assert pred.shape == truth.shape
assert np.isfinite(pred).all() and np.isfinite(truth).all()
low, high = truth <= 0.5, truth > 0.5
assert low.any() and high.any()
metrics = dict(cost_mae_raw=float(50 * np.abs(pred - truth).mean()),
    false_pass_pct=float(100 * ((pred <= 0.5) & high).sum() / high.sum()),
    false_block_pct=float(100 * ((pred > 0.5) & low).sum() / low.sum()),
    false_pass_denominator=int(high.sum()), false_block_denominator=int(low.sum()))
(run / 'building_SUST_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(metrics)
PY
```

这里的 false-pass 是高教师代价点被学生预测为 ≤0.5 的比例，false-block 是低教师代价点被预测为 >0.5 的比例，二者分母不同。这是点级规则一致性指标，与楼梯端点图的错误连通率分别报告。

### 11.4 冻结报告与可视化

完整冻结编排会自动生成 Small 报告。仅重画时运行：

```bash
"$GEOMETRY_PY" -m Traversability.visualization.full3d_frozen_report --run "$RUN_DIR"
```

该报告需要本次训练、独立预测和仍保留的当前训练池，读取当前配置和实际生成数。将 `REPORT.md`、`plots/`、`summary.json` 一起带回本机查看。

RViz 在有显示环境的本机执行，先将 `RUN_DIR` 设为下载后的本地运行目录：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m Traversability.visualization.rviz \
  --prediction "$RUN_DIR/independent_prediction/prediction.pcd"
```

## 12. 产物与完成标准

```text
runs/traversability/<RUN_NAME>/
├── requested_config.json        用户启动配置，手动留档
├── config.json                  训练实际配置
├── best.pt / last.pt            最佳模型与最近轮末恢复点
├── history.json                 每轮损失、验证、吞吐与显存记录
├── selection.json               最佳轮次及选优分数
├── summary.json                 实际轮数、最佳轮次、固定验证/测试
├── first_update.json            首次真实梯度与更新
├── resume_verification.json     恢复状态记录
├── training_verification.json   完成后的训练核验
├── predictions/                 固定测试预测
├── validation_predictions/      固定验证预测
├── scene_archive/               每栋实际配置、条件、概览和摘要
├── generations/、rotations/     轮换批次与删除旧 active 的记录
├── independent_prediction/      独立 PCD 推理结果
└── REPORT.md、plots/            冻结编排自动生成的报告
```

`progress.json`、`execution.json`、`protocol.json`、阶段日志和存储核验由相应入口写入。直接训练子入口不生成编排专属文件。

完成后核对：

1. `summary.completed_epochs` 与实际目标一致，`selected_epoch` 有明确验证依据。
2. `training_verification.json` 的 epoch 序列和优化器步数一致。核对冻结编码器与预训练来源相等。
3. 独立推理与固定测试有相同 XYZ，预测差异在约定容差内。现有报告使用最大差异 `<1e-5`。
4. 固定验证、固定测试、真实建筑 holdout 分别报告，教师条件和点集保持一致。
5. 官方预训练权重保留，正式训练目录只维护 `best.pt` 和 `last.pt` 两个训练 checkpoint。

本次 Small 配置尚未执行正式训练，实际指标由新运行产生。

## 13. 常见问题

| 现象 | 检查与处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 检查任务是否分配 GPU、设备可见性、驱动及 PyTorch CUDA 构建 |
| CuPy 导入成功但内核失败 | 运行第 6 节的小内核，检查 CuPy CUDA 系列和 NVRTC 库 |
| `rtree`、`perlin_numpy`、`alive_progress` 缺失 | 安装 WFC 显式依赖，原工作站可能从 `datasets/cache/wfc_runtime/` 导入 |
| 冻结编码器一致性核验失败 | 核对 Small 权重哈希、972 维配置，以及本次 checkpoint 的编码器张量 |
| `source_pcd` 不存在或指向旧根 | 同步原始固定场景并修正 manifest、ready 等 JSON 中的绝对路径 |
| 最近代表点不在 1 m 上下文内 | 检查坐标、尺度、PCD 与监督来源、匹配距离，再决定是否建立新窗口配置和缓存 |
| OOM | 先判断发生在几何对应、Small 编码还是任务头。Small 比先前编码器规模更大，GPU 显存峰值需在服务器实测。冻结编码峰值不能仅靠减少任务头 batch 解决 |
| 修改窗口或体素后仍读旧特征 | 使用新数据目录，缓存以完成标记复用，不自动按全部配置变更失效 |
| 日志停止但 `progress.status=running` | 对照 PID 与进程退出码，进度文件可能保留异常退出前的最后状态 |
