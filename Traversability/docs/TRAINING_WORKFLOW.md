# 从空 datasets 开始：冻结 Utonia Small 完整训练流程

更新日期：2026-11-13。以下主流程全部在服务器执行，使用 Bash，从已同步代码的 Pure3D_ws 根目录运行。

起点是服务器 `datasets/` 为空。固定验证集、固定测试集和轮换训练集都在服务器生成。模型使用官方 Small 预训练权重，编码器始终冻结，Adapter、FiLM 和 CostHead 随机初始化后训练。

按顺序执行：**设置环境和实验配置 → 生成固定场景 Mesh/PCD → 生成固定监督 → 生成首批训练数据 → 缓存 Small 特征 → 首轮训练检查 → 恢复至目标轮数 → 测试与独立推理**。本指南中的命令供服务器执行，本次文档修改没有启动真实数据生成或训练。

## 1. 设置目录、解释器和运行名称

代码、网络环境、几何环境和 Small 权重需要就绪。环境未安装时，先执行文末附录 A。权重尚未放到服务器时，执行附录 B。这里不需要任何历史数据目录或旧训练 checkpoint。

先在服务器进入真实工程目录，再执行下面的变量设置。解释器名称不同，只修改 `PTV3_PY` 和 `GEOMETRY_PY` 两行。

```bash
# 先 cd 到服务器上的 Pure3D_ws 根目录
export WORKSPACE="$(pwd -P)"
test -f "$WORKSPACE/Traversability/configs/full3d_small_frozen.json"

export PTV3_PY="$(conda run -n pure3d-ptv3-server python -c 'import sys; print(sys.executable)')"
export GEOMETRY_PY="$WORKSPACE/.venvs/trav-geometry/bin/python"
export GENERATOR_PY="$PTV3_PY"

export PYTHONPATH="$WORKSPACE:$WORKSPACE/common/vst_global_planner/src${PYTHONPATH:+:$PYTHONPATH}"
export TZ=Asia/Shanghai
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORKSPACE/datasets/cache/matplotlib"
export CUPY_CACHE_DIR="$WORKSPACE/datasets/cache/cupy"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export RUN_NAME="cost_full3d_small_frozen_e50_s2026111400_$(date +%Y%m%d-%H%M%S)_server01"
export RUN_CONFIG="$WORKSPACE/Traversability/configs/$RUN_NAME.json"
export RUN_DIR="$WORKSPACE/runs/traversability/$RUN_NAME"
export FIXED_ROOT="$WORKSPACE/datasets/cost_full3d_small_fixed_v1"

mkdir -p "$WORKSPACE/datasets" "$MPLCONFIGDIR" "$CUPY_CACHE_DIR" "$RUN_DIR"
test -x "$PTV3_PY"
test -x "$GEOMETRY_PY"
test -f "$WORKSPACE/weights/utonia_small.pth"
set -euo pipefail
```

`PTV3_PY` 负责 Small 编码和训练，也用于 WFC 生成。`GEOMETRY_PY` 负责 Mesh 采样、CuPy 原生教师和绘图。通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU 后，配置中的设备仍写 `cuda:0`。当前入口使用单 GPU。

新运行的名字统一为：

```text
cost_full3d_small_frozen_e<实际轮数>_s<全局seed>_<YYYYMMDD-HHMMSS>_<实验标签>
```

本例实际训练 50 轮，因此名字含 `e50`。时间使用上海时区，标签为 `server01`。配置、训练数据和运行产物对应：

```text
Traversability/configs/<RUN_NAME>.json
datasets/<RUN_NAME>/
runs/traversability/<RUN_NAME>/
```

恢复同一任务使用原名字和原目录。新的数据配方、模型、损失或预算使用新的名字。

## 2. 创建本次配置与固定集生成计划

固定集独立于轮换训练池，训练期间持续保留。下面设置两个固定场景和四栋一批的训练池，seed 互不重叠。

| 用途 | seed | 保留方式 |
|---|---|---|
| 固定验证 | 2026111301 | 始终保留 |
| 固定测试 | 2026111302 | 始终保留，训练结束后评分 |
| 训练第 0 批 | 2026111400–2026111403 | 第 1–10 轮使用 |
| 训练第 1 批 | 2026111404–2026111407 | 第 11–20 轮使用 |
| 训练后续批次 | 继续递增 | 每 10 轮替换四栋 |

```bash
"$PTV3_PY" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

root = Path(os.environ['WORKSPACE']).resolve()
name = os.environ['RUN_NAME']
path = Path(os.environ['RUN_CONFIG'])
fixed = Path(os.environ['FIXED_ROOT']).resolve()
cfg = json.loads((root / 'Traversability/configs/full3d_small_frozen.json').read_text())
cfg.update(
    data=f'datasets/{name}', output=f'runs/traversability/{name}',
    fixed_data=str(fixed / 'prepared'),
    generator_python=os.environ['GENERATOR_PY'],
    geometry_python=os.environ['GEOMETRY_PY'],
    neural_python=os.environ['PTV3_PY'],
    seed=2026111400, start_epoch=0, epochs=100, stop_epoch=50,
    batch_size=2, num_workers=2, cpu_threads=4, device='cuda:0')
cfg.pop('source_run', None)
cfg['encoder'].update(
    name='utonia_small', weights=str(root / 'weights/utonia_small.pth'),
    sha256='3d7afb731771556ebdf20abd1b6a1f022d9b9f105db8446f5eff02037f7fceb9',
    feature_channels=972, frozen=True)
cfg['rotation'].update(scenes_per_generation=4, epochs_per_generation=10,
                       seed_start=2026111400, surface_spacing_m=0.03)
assert not path.exists(), 'Existing config. Resume using its session.env.'
assert not (root / cfg['data']).exists(), 'Use a new run name for a new experiment.'
assert hashlib.sha256(Path(cfg['encoder']['weights']).read_bytes()).hexdigest() == cfg['encoder']['sha256']
for key in ['generator_python', 'geometry_python', 'neural_python']:
    assert Path(cfg[key]).is_file(), cfg[key]
path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')

jobs = [dict(id='scene_2026111301', seed=2026111301, split='val', folder='scene_2026111301'),
        dict(id='scene_2026111302', seed=2026111302, split='test', folder='scene_2026111302')]
generation_count = math.ceil(cfg['stop_epoch'] / cfg['rotation']['epochs_per_generation'])
training_scene_count = generation_count * cfg['rotation']['scenes_per_generation']
train_seeds = set(range(cfg['rotation']['seed_start'],
                       cfg['rotation']['seed_start'] + training_scene_count))
assert not train_seeds.intersection(j['seed'] for j in jobs)
recipe = dict(
    scenes=jobs, generator_config=cfg['generator_config'],
    generator_sha256=hashlib.sha256((root / cfg['generator_config']).read_bytes()).hexdigest(),
    teacher_config=cfg['teacher_config'],
    teacher_sha256=hashlib.sha256((root / cfg['teacher_config']).read_bytes()).hexdigest(),
    surface_spacing_m=cfg['rotation']['surface_spacing_m'],
    condition_names=cfg['condition_names'], condition_ranges=cfg['condition_ranges'],
    window=cfg['window'])
fixed.mkdir(parents=True, exist_ok=True)
recipe_path = fixed / 'recipe.json'
if recipe_path.exists():
    assert json.loads(recipe_path.read_text()) == recipe, 'Changed fixed-set recipe. Use a new FIXED_ROOT version.'
else:
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + '\n')
print('Run config:', path)
print('Fixed-set recipe:', recipe_path)
PY

cp "$RUN_CONFIG" "$RUN_DIR/requested_config.json"
for key in WORKSPACE PTV3_PY GEOMETRY_PY GENERATOR_PY PYTHONPATH TZ CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE MPLBACKEND MPLCONFIGDIR CUPY_CACHE_DIR OMP_NUM_THREADS OPENBLAS_NUM_THREADS RUN_NAME RUN_CONFIG RUN_DIR FIXED_ROOT; do
  printf 'export %s=%q\n' "$key" "${!key}"
done > "$RUN_DIR/session.env"
```

`epochs=100` 是学习率调度预算，`stop_epoch=50` 是实际停止轮数。这里沿用既定预算。若计划完整训练 100 轮，在创建配置前将 `RUN_NAME` 中的 `e50` 改为 `e100`，并设置 `epochs=100, stop_epoch=100`。训练 seed 排重检查会自动扩展到 40 栋。

本次生成的配置显式覆盖模板中的 `fixed_data`，指向服务器新建的 `FIXED_ROOT/prepared`。后续命令使用 `RUN_CONFIG`，无需旧固定集。

## 3. 从零生成固定验证和测试场景

### 3.1 生成 Mesh，再采样 terrain.pcd

下面脚本先调用现有 WFC 入口生成两栋多层建筑，再用现有 `sample_surface` 按 3 cm 对全部三角面采样。原始 PCD 包含完整建筑几何，5 cm 真实代表点在 GPU 缓存阶段再选择。

```bash
"$GEOMETRY_PY" -u - <<'PY' 2>&1 | tee "$RUN_DIR/fixed_geometry.log"
import hashlib
import json
import os
from pathlib import Path
import subprocess

import trimesh
from common.pointcloud_io import write_pcd
from utils.mesh_sampling import sample_surface
from utils.pointcloud import write_json

root = Path(os.environ['WORKSPACE']).resolve()
fixed = Path(os.environ['FIXED_ROOT']).resolve()
cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
recipe = json.loads((fixed / 'recipe.json').read_text())
for key, hash_key in [('generator_config', 'generator_sha256'), ('teacher_config', 'teacher_sha256')]:
    assert hashlib.sha256((root / recipe[key]).read_bytes()).hexdigest() == recipe[hash_key]
source, logs = fixed / 'source', fixed / 'logs'
source.mkdir(exist_ok=True)
logs.mkdir(exist_ok=True)
rows = []
for job in recipe['scenes']:
    folder = source / job['folder']
    generated = folder / 'geometry_ready.json'
    if not generated.exists():
        command = [cfg['generator_python'], '-u',
            str(root / 'common/geneator/generate_wfc_multifloor.py'),
            '--config', str(root / cfg['generator_config']), '--seed', str(job['seed']),
            '--output', str(folder), '--cache-dir', str(fixed / 'wfc_cache')]
        with (logs / (job['id'] + '_generation.log')).open('a') as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert (folder / 'terrain.obj').is_file() and (folder / 'scene.json').is_file()
        write_json(generated, dict(seed=job['seed'], generator_sha256=recipe['generator_sha256']))
    sampled = folder / 'sampling_ready.json'
    if not sampled.exists():
        mesh = trimesh.load_mesh(folder / 'terrain.obj', process=False)
        points, sampling = sample_surface(mesh, recipe['surface_spacing_m'])
        write_pcd(folder / 'terrain.pcd', points)
        write_json(sampled, dict(points=len(points), spacing_m=recipe['surface_spacing_m'], **sampling))
        del points, mesh
    assert (folder / 'terrain.pcd').is_file()
    rows.append(job)
    write_json(source / 'manifest.json', dict(scenes=rows, complete=len(rows) == len(recipe['scenes'])))
    print('FIXED GEOMETRY READY', job['split'], job['id'], flush=True)

prepare = dict(cfg, source_data=str(source), data=str(fixed / 'prepared'), output=str(logs))
write_json(fixed / 'prepare.json', prepare)
print('Teacher preparation config:', fixed / 'prepare.json', flush=True)
PY
```

生成物：

```text
datasets/cost_full3d_small_fixed_v1/
├── recipe.json
├── prepare.json
├── source/manifest.json
├── source/scene_2026111301/terrain.obj、terrain.pcd、scene.json、overview.png
├── source/scene_2026111302/terrain.obj、terrain.pcd、scene.json、overview.png
├── logs/scene_*_generation.log
└── wfc_cache/
```

先检查两栋场景的 `overview.png` 和 `config.resolved.json`，确认楼层、楼梯及实际尺寸。WFC 阶段日志在 `FIXED_ROOT/logs/`。脚本完成标记支持失败后重跑同一步，使用同一 `RUN_CONFIG` 和 `FIXED_ROOT`。

### 3.2 从 PCD 生成多条件原生教师标签

```bash
"$GEOMETRY_PY" -u -m Traversability.data.sliced \
  --config "$FIXED_ROOT/prepare.json" \
  2>&1 | tee "$RUN_DIR/fixed_teacher.log"
```

这里调用 `data.sliced` 生成原生支撑查询、条件标签和核心索引。该步骤不运行编码器。后续 Small 的编码输入来自完整 `terrain.pcd`。

固定验证和测试各生成两套 40 槽条件库，两套共享四个固定槽，因此 `cost.npy` 包含 76 组独立条件标签。每组标签与同一行 Tomogram XYZ 对齐。

```text
FIXED_ROOT/prepared/
├── manifest.json
└── scene_*/
    ├── coord.npy             原生 Tomogram 支撑 XYZ
    ├── cost.npy              [查询点数, 条件数]，归一化 cost
    ├── conditions.npy        [条件数, 4]，学生条件
    ├── facility.npy          设施区域掩码
    ├── layer_cell.npy        原生层/栅格索引
    ├── block_*.npz           核心与上下文索引
    ├── ready.json            场景、条件库和配对
    ├── ground_truth.pcd
    └── vst_*.pickle          保留的原生教师快照
```

### 3.3 验证固定集可以被训练读取

```bash
"$GEOMETRY_PY" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

fixed = Path(os.environ['FIXED_ROOT'])
root = fixed / 'prepared'
meta = json.loads((root / 'manifest.json').read_text())
assert meta['complete']
assert sorted(s['split'] for s in meta['scenes']) == ['test', 'val']
for scene in meta['scenes']:
    folder = root / scene['id']
    xyz = np.load(folder / 'coord.npy', mmap_mode='r')
    cost = np.load(folder / 'cost.npy', mmap_mode='r')
    cond = np.load(folder / 'conditions.npy', mmap_mode='r')
    assert xyz.shape[1] == 3 and len(xyz) > 0
    assert cond.shape == (len(scene['profiles']), 4)
    assert cost.shape == (len(xyz), len(cond))
    assert scene['shared_xyz_verified']
    assert Path(scene['source_pcd']).is_file()
    assert Path(scene['source_pcd']).with_name('scene.json').is_file()
    assert len(scene['banks']) == 2 and all(len(b) == 40 for b in scene['banks'])
    assert np.isfinite(xyz).all() and np.isfinite(cond).all()
    for start in range(0, len(cost), 65536):
        values = cost[start:start + 65536]
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    counts = np.zeros(len(xyz), dtype=np.int32)
    for block in [b for b in meta['blocks'] if b['scene'] == scene['id']]:
        with np.load(folder / (block['id'] + '.npz')) as data:
            indices = data['point_index']
        assert len(indices) == block['points']
        np.add.at(counts, indices, 1)
    assert np.all(counts == 1), 'Each support query must belong to one core.'
    print('FIXED VERIFIED', scene['split'], scene['id'], xyz.shape, cost.shape)
PY
```

固定源 PCD、`scene.json`、监督和条件从此保持不变。新版本教师、数据配方或几何划分使用新的 `FIXED_ROOT`，避免同一固定集名称代表不同数据。重新生成的固定集结果单独记录，不直接替代历史实验的固定集成绩。

## 4. 生成第 0 批训练数据

固定集准备完成后，生成四栋训练建筑及其多条件标签：

```bash
"$GEOMETRY_PY" -u -m Traversability.data.full3d \
  --config "$RUN_CONFIG" --generation 0 \
  2>&1 | tee "$RUN_DIR/prepare_00.log"
```

执行顺序为 `WFC Mesh → 3 cm terrain.pcd → 原生 VST 几何和条件标签 → 设施掩码与核心块索引`。训练场景使用 seed 2026111400–2026111403。固定验证测试被引用到本次训练数据目录。

```text
datasets/<RUN_NAME>/
├── manifest.json
├── generation.json           index=0，prepared=true，encoded=false
├── fixed/scene_2026111301/    固定标签引用，本次特征稍后单独生成
├── fixed/scene_2026111302/
├── active/source/            当前四栋 Mesh、terrain.pcd 和场景配置
├── active/prepared/          当前四栋条件监督与核心索引
└── scene_*/                  指向 active 或 fixed 的链接
```

此时尚未训练网络，也尚未缓存 Small 特征。检查 `RUN_DIR/scene_archive/scene_*/overview.png`、`config.resolved.json`、`conditions.json` 和 `data_summary.json`。

只生成当前批次。训练在第 11、21、31、41 轮按需生成后续批次，并删除旧 active 大文件。默认 50 轮共使用五批 20 栋训练建筑，磁盘持续保留当前四栋训练建筑和两栋固定建筑。

## 5. 构建 GPU 对应与冻结 Small 特征缓存

```bash
"$PTV3_PY" -u - <<'PY' 2>&1 | tee "$RUN_DIR/cache_00.log"
import json
import os
from pathlib import Path
import torch
from Traversability.training.full3d_cache import cache_features

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
torch.set_num_threads(cfg['cpu_threads'])
cache_features(cfg)
PY
```

该步骤处理首批四栋训练建筑和两栋固定建筑。Small 权重 SHA256 先被校验。编码器保持 eval 与冻结，每个新上下文编码一次。

完整 PCD 在 CUDA 上按 5 cm 体素选真实代表点，GPU 体素哈希将 Tomogram 查询对应到这些点。编码的核心为 4×4×3 m，外扩 1 m 上下文。上下文参与编码，核心查询参与损失。缓存只保存核心查询引用的 972 维 FP16 特征及对应索引。

`full3d_cache.py` 提供函数而没有独立 `--config` CLI，因此这里调用 `cache_features(cfg)`。手动缓存后 `generation.json` 的 `encoded` 仍可能为 false。训练会扫描并确认已有完成标记，然后更新批次状态，已完成的上下文不重新编码。

检查所有场景的特征宽度、匹配距离和显存峰值：

```bash
"$PTV3_PY" - <<'PY'
import json
import os
from pathlib import Path

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
root = Path(os.environ['WORKSPACE']) / cfg['data']
meta = json.loads((root / 'manifest.json').read_text())
for scene in meta['scenes']:
    folder = root / scene['id']
    report = json.loads((folder / 'full3d_encoding.json').read_text())
    assert report['full_geometry_input'] and not report['cpu_kdtree_used']
    assert report['supervised_queries'] == scene['points']
    for block in [b for b in meta['blocks'] if b['scene'] == scene['id']]:
        ready = json.loads((folder / 'features' / block['id'] / 'ready.json').read_text())
        assert ready['feature_channels'] == 972
        assert ready['full3d_input'] and not ready['query_points_inserted_into_encoder']
    print(scene['split'], scene['id'], 'queries=', report['supervised_queries'],
          'max_match_m=', report['matching']['maximum_m'],
          'peak_cuda_GiB=', report['peak_cuda_gib'])
assert json.loads((root / 'encoding.json').read_text())['frozen_weights_and_buffers_equal']
PY
```

如果 OOM 出现在几何对应或 Small 编码阶段，仅减少任务头 batch 不会降低该阶段峰值。先查看最大上下文点数、GPU 空闲显存和报告，再决定新的空间窗口或输入几何配置。修改窗口、体素或权重时使用新的数据目录并重新生成缓存。

## 6. 首轮训练与真实恢复检查

先完成一个真实 epoch，再恢复到完整目标。首轮遍历当前四栋建筑的全部核心块。

```bash
"$PTV3_PY" -u -m Traversability.training.full3d \
  --config "$RUN_CONFIG" --stop-after-epoch 1 \
  2>&1 | tee "$RUN_DIR/train_probe.log"
```

检查首个真实更新和轮末 checkpoint：

```bash
"$PTV3_PY" - <<'PY'
import json
import os
from pathlib import Path
import torch

run = Path(os.environ['RUN_DIR'])
first = json.loads((run / 'first_update.json').read_text())
assert first['true_batch'] == 2
assert first['encoder_name'] == 'utonia_small'
assert not first['encoder_in_optimizer']
assert all(value > 0 for value in first['gradients'].values())
assert all(value > 0 for value in first['max_parameter_change'].values())
state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
assert state['epoch'] == 1 and state['source_epoch'] == 0
assert state['config']['encoder']['frozen']
assert state['model']['adapter.0.weight'].shape == (256, 972)
print('FIRST EPOCH VERIFIED', 'epoch=', state['epoch'], 'step=', state['step'])
PY
```

这里验证 Adapter、条件头和代价头有梯度和参数变化，Small 不在优化器中。首轮成功后启动完整编排：

```bash
"$PTV3_PY" -u -m Traversability.experiments.full3d_rotating \
  --config "$RUN_CONFIG" 2>&1 | tee "$RUN_DIR/pipeline.log"
```

入口检测到 `last.pt` 后，从本次第 1 轮恢复至第 50 轮，按配置轮换训练场景，最后自动运行固定测试、独立 PCD 推理和 Small 报告。可以在 tmux 中执行以上命令，进入新会话后先加载第 8 节的 `session.env`。

## 7. 训练设置、选优与自动产物

| 设置 | 当前值 |
|---|---|
| 模型 | 官方 `utonia_small`，972 维特征，冻结与 eval |
| 任务头 | Adapter 972→256，四参数 FiLM，CostHead，sigmoid 输出 |
| Small 预处理 | `scale=0.5`、`grid_size=0.01`，约 2 cm 物理网格 |
| 输入几何代表点 | 5 cm 世界系体素内原始编号最小的真实点 |
| 监督 | 原生 Tomogram 全部有效支撑查询，cost/50 |
| 学生条件 | `[clearance_min, slope_max, step_max, stand_ratio]` |
| 教师固定值 | `interval_free=0.5 m`，距离曲线 `p=2`，障碍代价 50 |
| 标签条件范围 | 净空 0.2–1.0 m，坡度 0.18–0.70 rad，踏步 0.08–0.55 m，支撑比例 0.02–0.50 |
| 精度 | Small BF16，缓存 FP16，任务头及损失 FP32 |
| 真实 batch | 2 个核心块，末批不足时重复前面的块补齐 |
| 优化器 | 任务头 AdamW，weight decay=0.05 |
| 学习率 | 1e-6 预热至 1e-4，100 轮预算前 5% 预热，再余弦下降 |
| 训练轮数 | 实际 50 轮，在 100 轮调度预算中停止 |
| 损失 | SmoothL1 beta=0.1，绝对代价 + 2 倍条件差值损失 |
| 平衡 | 设施/背景等权，变化/稳定等权，四参数等权 |
| 教师变化阈值 | 归一化差值 0.02 |

训练逐轮交替使用两套条件库。同一几何共享坐标、对应和冻结特征。输入编码不插入教师查询点，对应阶段不建立全楼 CPU KD-tree。

每轮在固定验证集上计算：

```text
score = normalized cost MAE + mean four-parameter response error
```

验证分数最低者更新 `best.pt`。随机任务头的初始化也参与选优，因此最佳轮次可以为 0。训练结束加载最佳权重进行测试，测试集不参与选优。

```text
runs/traversability/<RUN_NAME>/
├── session.env、requested_config.json
├── config.json、protocol.json、execution.json、progress.json
├── best.pt、last.pt
├── history.json、selection.json、summary.json
├── first_update.json、resume_verification.json、training_verification.json
├── scene_archive/             每栋场景实际配置、条件、概览和摘要
├── generations/、rotations/   轮换及删除旧 active 的记录
├── predictions/              固定测试，relaxed 条件的单组预测
├── validation_predictions/
├── independent_prediction/   固定测试原始 PCD 的独立推理
├── storage_verification.json、independent_inference_verification.json
└── REPORT.md、plots/
```

`cost_mae_raw = normalized MAE × 50`。`summary.json` 的验证与测试指标覆盖两套条件槽，单组导出预测对应 `[0.5,0.6,0.4,0.05]` 的 `relaxed` 条件。独立推理输出的 XYZ 是原生几何支撑点集，保持地图坐标系和米单位，并非输入 PCD 的全部原始点。

## 8. 监控、中断和恢复

新终端先加载原任务的环境文件。将路径改为实际运行目录：

```bash
RUN_SESSION=/absolute/path/to/your_run/session.env
source "$RUN_SESSION"
cd "$WORKSPACE"
set -euo pipefail

cat "$RUN_DIR/progress.json"
cat "$RUN_DIR/execution.json"
tail -n 60 "$RUN_DIR/pipeline.log"
tail -n 60 "$RUN_DIR/train_resume_to_50.log"
nvidia-smi
du -sh "$FIXED_ROOT" "$WORKSPACE/datasets/$RUN_NAME" "$RUN_DIR"
```

加载 `session.env` 后，不再运行第 1 节生成新名字的命令。训练阶段日志为 `train_resume_to_<stop_epoch>.log`。场景生成细节在 `generation_*.log` 和 `scene_archive/scene_*/generation.log`。

完整三维训练在 epoch 结束时保存 `last.pt`。中断后从最近完整轮末恢复，未保存的轮内更新会重做。Ctrl+C 不保证额外写入轮中 checkpoint。

恢复前对照 `execution.json` 的 PID，确认旧进程退出，然后执行：

```bash
"$PTV3_PY" -u -m Traversability.experiments.full3d_rotating \
  --config "$RUN_CONFIG" 2>&1 | tee -a "$RUN_DIR/pipeline.log"
```

该命令按阶段继续，训练完成后衔接测试、推理和报告。只恢复训练子入口时使用：

```bash
"$PTV3_PY" -u -m Traversability.training.full3d \
  --config "$RUN_CONFIG" --resume 2>&1 | tee -a "$RUN_DIR/train_resume.log"
```

`--resume` 是布尔开关，固定读取 `config.output/last.pt`。模型、优化器、随机状态、累计步数、调度阶段和数据批次继续使用原任务状态。恢复时保持原条件、损失、窗口、seed 和调度预算。

## 9. 检查最终结果并推理新 PCD

```bash
"$PTV3_PY" - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ['RUN_DIR'])
summary = json.loads((run / 'summary.json').read_text())
check = json.loads((run / 'training_verification.json').read_text())
assert summary['completed_epochs'] == 50
assert check['expected_epochs_complete']
assert check['encoder_weights_and_buffers_equal_source']
assert check['best_encoder_equal_source']
print('Completed:', summary['completed_epochs'], 'selected:', summary['selected_epoch'])
for key in ['best_validation', 'test']:
    row = summary[key]
    print(key, 'raw MAE=', row['cost_mae_raw'], 'response error=', row['mean_response_error'])
comparisons = json.loads((run / 'independent_inference_verification.json').read_text())
assert all(row['xyz_equal'] and row['maximum_prediction_difference'] < 1e-5 for row in comparisons)
print('Frozen encoder and independent inference verified')
PY
```

若实际目标改为 100，以上轮数检查相应改为 100。查看 `REPORT.md`、`plots/training_curves.png`、`plots/*_GT_before_after_error.png`。固定验证和固定测试是合成建筑结果，真实建筑的泛化需要另行准备真实 PCD 和同点集规则参考。

在另一个输入 PCD 上独立预测：

```bash
export INPUT_PCD=/absolute/path/to/new_building.pcd
"$PTV3_PY" -u -m Traversability.inference.full3d \
  --checkpoint "$RUN_DIR/best.pt" --pcd "$INPUT_PCD" \
  --condition 0.5 0.6 0.4 0.05 --output "$RUN_DIR/new_building"
```

输出 `prediction.pcd`、`prediction.npy`、`prediction.json` 与 `geometry/`。推理需要几何环境生成支撑查询，不读取教师 cost、不生成多条件监督。输入坐标和尺度应与训练约定一致。

仅重画本次报告：

```bash
"$GEOMETRY_PY" -m Traversability.visualization.full3d_frozen_report --run "$RUN_DIR"
```

## 附录 A. 服务器环境查询与安装

### A.1 查询现有环境

```bash
uname -a
cat /etc/os-release
lscpu
free -h
df -h . /tmp
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
conda env list
python3 --version
command -v gcc
command -v g++
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
fi
```

`nvidia-smi` 的 CUDA 数字表示驱动支持能力，PyTorch 构建版本用 `torch.version.cuda` 查询。实际可用性以 PyTorch 和 CuPy 内核运行为准。

### A.2 安装两套环境

下面命令从工程根目录执行。PyTorch CUDA 12.8 构建沿用本机参考，先核对服务器驱动。当前 vendored PTv3 要求 `torch>=2.10`。

```bash
export WORKSPACE="$(pwd -P)"
conda create -n pure3d-ptv3-server python=3.11 pip -y
conda run -n pure3d-ptv3-server python -m pip install \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n pure3d-ptv3-server python -m pip install \
  'numpy==1.26.4' scipy trimesh manifold3d networkx pyyaml \
  'opencv-python-headless==4.11.0.86' matplotlib pillow \
  alive-progress perlin-numpy rtree open3d
conda run -n pure3d-ptv3-server python -m pip install -e ./utils/PointTransformerV3

python3 -m venv "$WORKSPACE/.venvs/trav-geometry"
"$WORKSPACE/.venvs/trav-geometry/bin/python" -m pip install --upgrade pip
"$WORKSPACE/.venvs/trav-geometry/bin/python" -m pip install \
  'numpy==1.26.4' scipy trimesh manifold3d matplotlib pillow pyyaml \
  'opencv-python-headless==4.11.0.86' 'cupy-cuda12x==13.6.0'
```

网络环境同时承担 WFC 上游导入链，所以显式安装 Open3D、rtree、alive-progress 和 perlin-numpy。生成可 headless 运行。不要用 `requirements-wfc.txt` 中固定的 CPU torch 覆盖训练环境。当前 PTv3 使用 PyTorch 与 Triton，主线不额外安装 spconv、torch_scatter 或外部 flash-attn。

### A.3 验证设备和依赖

先执行主流程第 1 节的解释器与公共变量设置，再运行：

```bash
"$PTV3_PY" - <<'PY'
import sys
import torch
import triton
import point_transformer_v3
print('python:', sys.executable, 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('triton:', triton.__version__)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.ones(1024, device='cuda').sum().item() == 1024
print('GPU:', torch.cuda.get_device_name(0))
PY

"$GENERATOR_PY" common/geneator/generate_wfc_multifloor.py --help

"$GEOMETRY_PY" - <<'PY'
import cupy as cp
import numpy, scipy, trimesh, cv2, matplotlib
from vst_global_planner.conditioned_teacher import ConditionedTeacher
cp.show_config()
a = cp.arange(16, dtype=cp.float32)
kernel = cp.ElementwiseKernel('float32 x', 'float32 y', 'y = x + 1', 'trav_env_probe')
assert float(kernel(a).sum()) == 136
print('CuPy kernel OK')
PY

"$PTV3_PY" -m pip check
"$GEOMETRY_PY" -m pip check
```

`--help` 检查 WFC 导入，不生成数据。CuPy 小内核检查设备访问、NVRTC 与 CUDA 库。安装步骤需在实际服务器通过这些检查后再生成数据。

任务开始后留档环境：

```bash
nvidia-smi > "$RUN_DIR/environment_gpu.txt"
"$PTV3_PY" -m pip freeze > "$RUN_DIR/environment_neural.txt"
"$GEOMETRY_PY" -m pip freeze > "$RUN_DIR/environment_geometry.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
  git status --short > "$RUN_DIR/source_status.txt"
fi
```

## 附录 B. Small 预训练权重

源码同步不包含被忽略的 `weights/`。将本机 `weights/utonia_small.pth` 单独传到服务器同位置，或在服务器联网下载：

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path
import urllib.request

path = Path('weights/utonia_small.pth')
expected = '3d7afb731771556ebdf20abd1b6a1f022d9b9f105db8446f5eff02037f7fceb9'
path.parent.mkdir(exist_ok=True)
if not path.exists():
    url = 'https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/utonia_small.pth'
    temporary = path.with_suffix('.pth.partial')
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
    assert hashlib.sha256(temporary.read_bytes()).hexdigest() == expected
    temporary.replace(path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print(path, path.stat().st_size, expected)
PY
```

官方 Small 权重大小为 149082467 字节，输出 972 维特征。训练不加载旧任务 checkpoint。权重加载、缓存与恢复阶段分别检查 SHA256、冻结状态和张量一致性。

## 附录 C. 迁移已生成的数据或中断任务

在服务器从空目录开始时，主流程已在服务器本地生成全部数据，不需要路径替换。

将本任务迁移到另一台服务器时，复制 `FIXED_ROOT`、`datasets/<RUN_NAME>/`、`RUN_DIR` 和 Small 权重。使用 `rsync -aL` 解引用数据链接，避免复制后仍指向旧机器。随后在目标副本修正：

1. `RUN_CONFIG`、数据 manifest 和各级 JSON 中的绝对工程根及 `source_pcd`。
2. 配置中的三套解释器路径和 `encoder.weights`。
3. 本次 `best.pt`、`last.pt` 的 `config` 元数据，使推理能调用目标几何环境。保留模型、优化器、`initial_model` 和随机状态张量。
4. `session.env` 中的目录与解释器，之后从该文件恢复环境。

固定源 PCD、标签、条件与核心划分保持。路径元数据变更后的 checkpoint 记录新的 SHA256。恢复既需要 `last.pt`，也需要运行 JSON、固定数据和对应当前训练池。迁移后先核对文件、坐标和冻结权重，再使用原运行名恢复。

## 附录 D. 常见问题

| 现象 | 检查与处理 |
|---|---|
| 找不到旧数据目录 | 从第 2 节创建本次配置，确认 `fixed_data` 为 `FIXED_ROOT/prepared` |
| `fixed_data/manifest.json` 不存在 | 先完成第 3 节的固定场景和监督生成，再运行 `data.full3d` |
| 固定集缺少 `terrain.pcd` | 检查 Mesh 采样是否完成，监督 manifest 引用的源 PCD 要持续保留 |
| WFC 缺依赖 | 按附录 A 补齐网络环境导入链，重新执行 `--help` |
| CuPy 能导入但教师失败 | 先运行 CuPy 小内核，检查 CUDA 系列、NVRTC 和驱动 |
| 匹配点不在 1 m 上下文内 | 检查 PCD/标签的坐标、尺度与来源，查看对应距离，再建立新版本窗口配置 |
| 缓存仍显示旧特征宽度 | Small 必须为 972 维，使用新的数据目录重新编码 |
| `progress.status=running` 但无进程 | 进度可能是异常退出前状态，对照 PID 和阶段日志 |
| 首轮没有生成 `summary.json` | `--stop-after-epoch 1` 是受控轮末退出，完整训练完成后才写最终摘要 |
| 没有真实建筑数据 | 完成生成集训练与测试，真实建筑 holdout 作为独立后续评价 |
