# vst_global_planner

本工程已纳入 glocal2-nav 的 third_party/vst_global_planner。服务器训练流程见 [PTv3 服务器说明](../../src/ptv3_tomography/docs/SERVER_TRAINING.md)。场景、路线和训练数据位于本地 outputs，不随源码提交。独立资产包保存几何、VST 和路线，网页回放需要重新生成。

采用方案 A：PTv3 与多帧历史融合预测显式分层通行地图，再由地图 CNN 和 SRU 学习导航。选定的编码器是 PTv3。架构见 [方案 A](docs/SCHEME_A.md)。

第一阶段建立一个组合通行平台，供后续同时采集不同结构的观测与 Tomography 监督。训练状态以服务器训练文档和实际运行记录为准。

当前场景与标签的实测记录见 [VALIDATION.md](VALIDATION.md)。

**随机路线与传感器回放：[8769 看板](http://127.0.0.1:8769/index.html)。** 已保存九个区域、每区两条原生 VST/A*/GPMP 路线。选择路线后，可在当前建筑内继续随机生成新路线，并查看 RGB、深度与 MID-360 回放。页面提供“全场景”和“当前建筑”视角。

## 场景

- 56 × 48 m 的切角底板，按 RoM-Nav 视频中的紧凑混合地形场重新布置。曲桥下面保留真实贯通孔洞。
- 名义 2.4 m 高、默认随机为 1.8–2.8 m 的两块平台，通过 S 形无栏杆窄桥相连。
- 两条缓坡分别连接独立坡道顶平台与独立大平台。
- 带薄踏板和连续侧梁的空心楼梯，中心线随机弯曲 −80°–80°，起点、中段和终点的宽度独立采样，梯下未填实。
- U 形折返楼梯，中间平台、每段级数、踏步尺寸和梯宽随 seed 变化，转向 180°。
- 450° 旋转楼梯，超过一圈，同一 XY 存在多个支撑高度。
- 有真实拱形净空的桥洞，可从下方通过，也有可到达的顶部。
- 尺寸随机的独立大平台，可为完整板、矩形孔、椭圆孔、双孔或边缘缺口，连同自己的坡道，与桥洞分开摆放。
- ETH terrain-generator 的实心参考楼梯，以及地面和高台上的障碍物。
- 不规则低墙通道与蓝色立柱，形成额外的绕行和窄入口样本。

弯桥、斜坡、开放楼梯、折返梯、旋转梯、独立大平台、桥洞、障碍物、低墙通道分成九个任务区。所有建筑组按 seed 生成独立随机 yaw，保持组内楼梯、坡道和平台的刚性连接。布局器按旋转后的占地与候选进出点摆放，最小结构净间距要求为 1.8 m。各区规划边界不相交，路径隔离由规划数据保证。桥的弯曲幅度和障碍分布也随 seed 变化。此版是可控的混合训练场，不代表随机建筑级布局多样性。未添加桥面防护栏。

默认启用 `configs/training_platform.json` 中的 `geometry_randomization`。同一 seed 确定楼梯总高度、级数、踏步深度、沿程宽度与弯曲、各平台尺寸和孔洞、曲桥宽度与弯曲幅值，以及障碍数量、形状、尺寸、位置和朝向。折返楼梯中间平台、各梯顶平台和旋转梯出口按尺寸更新连接。三类障碍为长方体、椭圆柱体和楔形。实际参数位于 `config.resolved.json`，障碍明细位于 `scene.json` 的 `obstacle_geometry`，平台孔洞位于 `platform_openings`，可据此复现；将 `enabled` 设为 `false` 可使用名义尺寸。开放楼梯为 12–22 级、深度 0.25–0.60 m、基准宽 0.8–2.8 m，沿程宽度倍率为 0.4–1.5。大平台为 3.4–8.0 × 3.0–7.0 m。孔洞切穿平台板，保留下面的地面。每次改变几何都重新构建 VST 监督。

几何布尔使用 manifold3d ≥ 3.5.4 的双精度接口。桥面和梯顶平台先合并二维轮廓再整体挤出，最终导出前保留中间实体，避免共面板段的合并误差。

布局参考 [RoM-Nav 项目页](https://wdc3iii.github.io/rom-nav/) 和用户提供的 `stage1-rom.mp4`。默认 56 × 48 m 是本项目为扩大结构随机化设定的尺寸。参考观察与调整依据见 [playground 记录](docs/PLAYGROUND_REFERENCE.md)。

起终点候选携带 `region_id` 和规划边界。生成标签后，各区有独立裁剪的 VST。教师应加载对应区域的裁剪文件，使路径搜索无法借用其他建筑。雷达仍查询完整场景，避免裁剪底板制造虚假的可见边缘。已保存的路线是规划参考，未代表机器人动力学执行结果。

## 生成

复用现有 glocal2-nav 几何环境即可运行，暂不安装或迁移机器学习依赖：

```bash
export SCENE_PY="$CONDA_PREFIX/bin/python"
cd third_party/vst_global_planner
"$SCENE_PY" scripts/scenes/generate_training_platform.py \
  --glocal2-root ../.. \
  --seed 20261104 --output outputs/playground_rotated_seed20261104
```

输出目录必须不存在。新 seed 使用新输出目录。GPU 驱动与原 VST 环境可用后，在一次新生成命令末尾增加 `--tomography`，重新生成该场景的真实标签，不复用其他地图。

也可以为尚未有标签的现有场景单独生成：

```bash
"$SCENE_PY" scripts/data/build_tomography.py \
  outputs/playground_rotated_seed20261104
```

几何回归检查入口为 `scripts/diagnostics/validate_platform.py`，默认检查三个 seed。GPU 标签验证与几何检查分别记录。

查看场景：

```bash
python3 -m http.server 8780 --bind 127.0.0.1 --directory outputs
```

打开 [随机朝向 Playground](http://127.0.0.1:8780/playground_rotated_seed20261104/index.html)。网页可按朝向角度切换任务区、视角、显示比例，以及单独隐藏结构。显示的 Z 比例放大不改变网格文件的真实米制坐标。

## 输出与范围

| 文件 | 用途 |
| --- | --- |
| terrain.obj / terrain.stl | 布尔合并后的真实碰撞与采样网格 |
| terrain.glb | 按结构分组着色的三维模型 |
| terrain.pcd | 干净、完整场景点云，用于 VST 监督，不是稀疏雷达输入 |
| scene.json / config.resolved.json | 场景、参数、seed、后端源码指纹 |
| geometry_paths.json | 几何支撑探测序列和候选端点，不是已验证导航教师路线 |
| geometry_validation.json | 闭合性、连通实体、支撑面、孔洞与桥洞检查 |
| index.html / preview.png | 独立交互查看与静态概览 |
| details.png | 九个任务区的近景 |
| training_regions.json | 区域、规划边界、结构分组与任务归属 |
| vst.pickle / vst.meta.json | 仅在显式 GPU Tomography 生成成功后存在 |
| regions/<区域>/vst.pickle / tomography.npz | 独立规划用标签裁剪，保持原世界坐标 |
| regions/<区域>/queries.json | 本区的候选起终点 |

`terrain.obj` 是几何权威。`scene.solids` 保留构造来源，不能直接当作完整解析碰撞模型。新增拱形/旋转构件与底板减孔需要后续训练适配器使用真实网格与 VST。

每区在 `construction_to_world` 保存完整刚性变换及 yaw。候选起终点、初始朝向、桥下孔洞和探测序列均应用同一变换。楼梯的原始尺寸字段仍在构造坐标系，世界入口、出口和朝向另有明确字段。不能把原始尺寸字段误读为旋转后的世界坐标。

当前检查只证明静态几何和采样支撑面符合设计。机器人是否能走过窄桥、爬升楼梯，以及完整场景的目标间可通行性，仍需后续模型与控制器执行验证。

先完成多建筑/多 seed 数据划分、批量 LiDAR 时序与噪声、标签局部对齐，再实现方案 A 感知训练。仅更换障碍 seed 不等同于未见建筑泛化。保留原教师、奖励和物理限制约定。

## 随机路线与 8769 看板

路线生成器从每个设施完整的 VST 可通行地面中独立、均匀采样起点和终点，候选点覆盖全部有效高度层，同一地面的重复层表示只计一次。新场景的规划代价阈值为 15。固定入口/出口锚点和障碍区两端采样已移除。`--per-region 2` 表示两条独立随机路线，不保证它们互为正反方向。

当前路线数据在 `outputs/random_routes_seed20261104`。它引用原始几何与 VST，候选点、原生 CSV/NPY、回放包和在线随机槽位单独存储。原始地形目录保持不变。

生成新的一批路线时使用尚不存在的输出目录：

```bash
PYTHONUTF8=1 "$SCENE_PY" \
  scripts/planning/generate_random_routes.py outputs/playground_rotated_seed20261104 \
  --output outputs/random_routes_new --seed 20261105 --per-region 2
PYTHONUTF8=1 "$SCENE_PY" \
  scripts/visualization/build_route_viewer.py \
  outputs/playground_rotated_seed20261104 outputs/random_routes_new
```

安装到现有 8769 服务：

```bash
"$SCENE_PY" -X utf8 \
  scripts/visualization/install_viewer_8769.py outputs/random_routes_new
```

安装器只为 `glocal2-viewer-8769.service` 添加 playground 专用覆盖配置，并保存原配置备份，沿用现有服务与健康检查。页面和接口复用 glocal2-nav，当前项目提供区域选点、离线网格检查和适配层，未修改 glocal2-nav 的训练代码。

路线必须经过原生 A* 和 GPMP，并检查终点高度、区域边界以及实际网格的支撑和上部空间。离线检查使用当前验证配置的半径、最大台阶和高度。它不证明完整机器人能够动态执行，也不代表学生策略的成功率。

完整路径保存在 `paths/*.gpmp.csv`，列为 `x,y,z,yaw,s`。Z 保留原生 GPMP 高度参考，默认包含 0.1 m 偏置。回放按路径弧长插值，不执行机器人控制器。后续教师速度标签仍应依据整条路线，独立于 PGCL 提示点。

```bash
"$SCENE_PY" -X utf8 \
  scripts/diagnostics/check_route_bundle.py outputs/random_routes_seed20261104
```

## 来源

通过指定路径调用 glocal2-nav 的几何模块，没有修改该仓库的训练代码。其 vendor 保留的 ETH terrain-generator 实心楼梯生成器用于参考楼梯。曲桥、旋转踏板和拱形桥洞在本项目扩展，未使用完整 WFC。许可证与来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

