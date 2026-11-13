# 当前入口

2026-11-11：按用户最新要求，训练数据改为直接生成局部目标块。**不再使用下文大场景v2流程作为当前训练入口。** 请使用 [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) 的 Filter.data.generate，输出 datasets/filter_local_10cm_v3。原v2示例保留供查看。

---

# Filter 数据生成：filter_trails_v2

更新日期：2026-11-11。D0、D1、D2 已实现。本次只生成 seed 821000 的四变体，不启动模型训练。

## 单场景完整生成

```bash
MPLCONFIGDIR=/tmp/pure3d_matplotlib /home/cjh/glocal2-nav/.venv/bin/python \
  -m Filter.data.prepare --seed 821000 --variants 4
```

配置是 `Filter/configs/dataset.json`，结果为 `datasets/filter_trails_v2/scene_821000/`。生成器复用 Traversability 的原生 WFC 多层模板、楼梯、斜坡、栏杆与域随机化，再加入带门洞和顶板的走廊。静止物体放在走廊一端，移动路径在另一段生成。

D0 保留 Mesh 面片和构件身份。`scene.obj/clean.obj` 是建筑加静止物体的静态参考，`terrain.obj` 是建筑，`static_objects.obj` 是静止人物、椅子和可移动物体。`roles.json` 的静止人物和家具同样是高保留参考，不能沿用 v1 的非目标删除语义。

D1 对静态 Mesh 做 3 cm 表面采样，再按 5 cm 体素选择原始表面代表点。基础观测使用固定 seed 的密度变化、缺口、条带和轻微噪声，四变体共享这部分观测。额外构造墙、楼梯和栏杆的孤立正确点簇，删除外围环带但保持簇内准确点。

| 变体 | 内容 |
|---|---|
| 000 | 静态对照 |
| 001 | 动态残影＋静态对照 |
| 002 | 配准重影＋静态对照 |
| 003 | 动态残影＋配准重影＋静态对照 |

每层走廊为人物、椅子和可移动物体各生成一条轨迹。每轨迹 8–20 个位置，每位置取 30%–80% 局部表面。直线、弯曲路径交替，行进方向、位置数量及表面抽取方向随机。完整轨迹累积进单张 rough.pcd，模型不读取这些时间步。这里只采样表面点，不计算射线可见性。

D2 用 Open3D CPU 三角面距离查询，一次建立静态索引，四个变体复用。每批查询 262144 点，8 线程。运动来源全部目标为 0，其他点按 `exp(-0.5*(d/0.03)^2)` 评分。`targets_*.json` 记录各类点数、标签分布及耗时。

## 产物

每个变体保存 XYZ 输入 `rough_000.pcd`、逐点来源 `rough_attributes_000.npz`、标签 `targets_000.npz`、变换表 `augmentation_sources_000.npz`、增强记录 `augmentation_000.json`。

显示文件为 `gt_000.pcd`、`filtered_gt_000.pcd`、`dynamic_trails_000.pcd`、`registration_ghosts_000.pcd`、`static_controls_000.pcd`。它们用于检查，不是模型输入。

每场景都有 `overview.png`。每变体有同视角输入、来源、连续 GT、GT 筛除结果的 `comparison_*.png` 和 `window_*.png`。动态轨迹、配准重影和正确断连结构分别有近景。绘图隐藏天花板并减少显示点，原始 PCD 全点保留。

分盒目录为 `boxes/variant_000/box_00000/`，包含 `rough.pcd`、`clean.pcd`、两者属性、`targets.npz` 和 `box.json`。核心 4×4×3 m，外扩 1 m。输入保存完整上下文，loss 仅计算 core_mask。每点保留唯一核心归属和原始 point_index，XYZ 仍相对于地图原点。

## 复用新版静态参考

```bash
MPLCONFIGDIR=/tmp/pure3d_matplotlib /home/cjh/glocal2-nav/.venv/bin/python \
  -m Filter.data.prepare --augment-only datasets/filter_trails_v2/scene_821000 --variants 4
```

该入口仅接受 v2 场景，不把 v1 非目标身份转换成新版参考。批量入口 `python -m Filter.data.generate --batch batch_00` 沿用场景 seed 划分，不能将同一建筑的盒随机分到 train/val/test。

## RViz

```bash
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=85 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 -m Filter.visualization.rviz \
  --full-scene --scene datasets/filter_trails_v2/scene_821000 --variant 3
```

省略 `--full-scene` 可随机查看盒，`--with-trails` 选择含动态或配准重影的盒，`--box` 指定一个盒。显示项分别提供粗糙输入、静态参考、连续 GT、动态轨迹、配准重影、静态对照与 GT 保留结果。颜色为监督，不是网络预测。关闭 RViz 后发布进程退出。

`preparation.json` 记录 Mesh、采样、索引构建、各变体增强、标签、保存、分盒、绘图及总耗时。旧数据目录不覆盖，编码缓存和模型训练留待后续。
