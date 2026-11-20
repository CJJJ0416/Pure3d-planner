# Pure3D 导航框架使用说明（Entrance）

`Entrance/` 是 Pure3D 导航框架的统一启动入口：业务实现按职责分布在 `common/`、`planner/` 与 `utils/`，运行入口、场景选择和集中配置放在本目录。本说明面向日常使用，覆盖单项启动、整套启动、配置修改与排障。

## 1. 目录结构

```text
Entrance/
├── launch/
│   ├── sim/                 仿真单项：Gazebo、LIO、键盘、cost、SCAN、导航、bag
│   │   └── _internal/       场景解析与 A1 启动实现（仅仿真）
│   ├── real/                真机单项：LIO、RViz、TF、real_bag
│   └── _internal/           sim/real 共用的运行库 runtime.py
├── config/                  运行参数与 RViz 布局
│   ├── startup.yaml         默认场景与仿真公共默认值
│   ├── scenes/              各场景的 world、模型、出生点或导出目录
│   ├── sim/                 仿真 LIO 与安装外参
│   └── real/                真机 LIO、外参和 bag profile
└── shell/                   整体构建与四终端启动
    └── _internal/           环境库、终端托管与共用 LIO 运行代码
```

### 1.1 launch 分类

| 目录 | 入口 |
|---|---|
| `launch/sim/` | `gazebo`、`keyboard`、`faster_lio`、`cost`、`local`、`scan`、`navigation`、`whole_map`、`mission`、`rviz`、`map_tf`、`follow`、`compare_context` |
| `launch/real/` | `faster_lio_real`、`rviz_real`、`map_tf_real`、`real_bag` |
| `launch/_internal/` | `runtime.py`：sim/real 共用的环境加载与参数转换 |

仿真 launch 内部的 `sim/_internal/a1.launch.py` 与 `sim/_internal/scenes.py` 只服务 Gazebo 组合，不对外提供入口。

原 `Entrance/launch/<name>.launch.py` 路径已迁移到 `sim/` 或 `real/`，本仓库不保留旧路径转发文件。

约定：

- 所有命令都在工作区根目录 `/home/cjh/Pure3D_ws` 执行。
- 仿真入口在 `Entrance/launch/sim/`，真机入口在 `Entrance/launch/real/`；公共运行库在 `Entrance/launch/_internal/runtime.py`。
- 单项 launch 只需先加载系统 ROS，工作区 overlay 由入口自行加载。
- `<...>` 表示需要替换为实际路径或数值；含空格的 `args` 需要整体加引号。
- 各分类子目录的 `.launch.py` 是推荐入口；ROS 包内的原生示例 launch 仅用于相应包独立调试。
- 文中 `config/...` 路径相对 `Entrance/`；`outputs/...`、`.cache/...` 相对工作区根目录。

## 2. 环境准备

```bash
cd /home/cjh/Pure3D_ws
source /opt/ros/humble/setup.bash

# 首次使用或依赖包有变动时构建
./Entrance/shell/build.sh
```

构建说明：

- 构建输出在 `.cache/gazebo_ws/{build,install,log}`，使用 `--symlink-install`；launch/Python 脚本改动通常无需重新构建，C++ 包（如 `scan_planner`、`unitree_gazebo`）改动仍需重新构建。
- `unitree_gazebo` 会把整个 `Entrance/` 安装到包的 share 目录；修改 launch 目录结构后，如需使用安装副本，请重新运行 `build.sh`。
- `build.sh` 构建 `unitree_gazebo`、`keyboard_input`、`scan_planner` 及其依赖；默认 `MAKEFLAGS=-j2`，可用环境变量覆盖。
- 仿真运行依赖 `LIBTORCH_ROOT`（默认 `~/third_party_lib/libtorch`）。
- Faster-LIO 需在 `PURE3D_LIO_WS`（默认 `~/livox_ws`）中已构建 `faster_lio/run_mapping_online`。真机可指向自己的 LIO 工作区。
- cost 推理使用 `config/local_cost.yaml` 中的 `network_python`（默认 conda 环境 `pure3d-ptv3`）和 `weights/` 下的权重；启动前会检查文件是否存在。
- `sim_env.bash` 会过滤其它工程遗留的 overlay，只保留系统 ROS、仿真 overlay 与 libtorch；不必手动 source 工作区。

## 3. 快速开始

### 3.1 查看场景与预览

```bash
# 列出 config/scenes 下的场景名
ros2 launch Entrance/launch/sim/gazebo.launch.py list_scenes:=true

# 只解析场景资源并打印最终参数，不启动 Gazebo
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=terrain-prepared dry_run:=true
```

| 场景 | 配置 | 行为 |
|---|---|---|
| `building` | `scenes/building.yaml` | Building-no-rails；EOLP 模型优先、包内副本兜底 |
| `terrain-prepared` | `scenes/terrain-prepared.yaml` | 复用导出目录中的 world、STL 与 `spawn.json` |
| `terrain` | `scenes/terrain.yaml` | 先从源 `terrain.obj` 和 `scene.json` 导出，再启动 |
| `sru-singleascent` | `scenes/sru-singleascent.yaml` | 原 seed 111824315 单条上楼场景 |

### 3.2 仿真：Gazebo + A1 + MID360 + 规划器

```bash
# 默认场景 + SCAN，打开 Gazebo 与 RViz
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building

# 无界面运行
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=terrain-prepared gui:=false rviz:=false

# 临时覆盖出生位姿（米、弧度）
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building x:=-5.5 y:=6.0 height:=0.5 yaw:=0.0

# 选择规划器；None 只跑仿真，不发布速度
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building planner:=ego
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building planner:=None

# 模式 4（cost 引导）；必须 planner:=scan
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building flight_type:=4
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `scene` | `startup.yaml` 的 `default_scene` | 场景名，对应 `config/scenes/<名称>.yaml` |
| `planner` | `scan` | `scan`、`ego` 或 `None` |
| `flight_type` | `1` | SCAN 模式：1 人工、2 预设航点、3 参考路径、4 cost 引导 |
| `gui` / `rviz` | `true` | Gazebo 客户端与 RViz |
| `mid360` | `true` | 是否加载 MID360 与 IMU |
| `mid360_samples` / `mid360_downsample` | `10000` / `1` | 每帧采样数与取样步长；增大步长可降负载 |
| `device` / `trace` | `cpu` / `false` | RL 推理设备与调试数据发布 |
| `clock_rate` | `100.0` | Gazebo 时钟发布频率（Hz） |
| `model_path` | 包内楼梯策略 | RL 权重路径 |
| `x` / `y` / `height` / `yaw` | 场景 YAML | 出生位姿，米与弧度 |
| `cost_astar_limit` / `cost_planner_backend` | `0.8` / `light_scan` | 模式 4 的代价上限与跟踪后端 |
| `world` / `scene_models_path` | 场景 YAML | 一般无需手动指定 |
| `dry_run` / `list_scenes` | `false` | 预览/列出场景，不启动进程 |

表中默认值来自 `startup.yaml`；场景 YAML 的 `launch_arguments` 与命令行可覆盖。

任何 launch 都可用 `--show-args` 查看完整参数：

```bash
ros2 launch Entrance/launch/sim/gazebo.launch.py --show-args
```

### 3.3 键盘

```bash
# 独立交互终端；由 SCAN 控制速度时关闭键盘速度
ros2 launch Entrance/launch/sim/keyboard.launch.py publish_cmd_vel:=false

# 纯键盘遥控
ros2 launch Entrance/launch/sim/keyboard.launch.py publish_cmd_vel:=true
```

按键：`2` 蹲下/站立，`7` 进入 RL，`P` 缓停/恢复，`1` 被动。该 launch 需要真实 TTY；在重定向或非交互环境中会直接报错。

### 3.4 Faster-LIO 建图

```bash
# 仿真：站稳并启动 Gazebo 后，另开终端
ros2 launch Entrance/launch/sim/faster_lio.launch.py

# 只生成并核对实际配置，不启动建图
ros2 launch Entrance/launch/sim/faster_lio.launch.py check_config:=true output:=/tmp/pure3d_lio_check

# 真机
ros2 launch Entrance/launch/real/faster_lio_real.launch.py

# 离线 bag（仅仿真契约，要求 ROS_DOMAIN_ID=184）
ros2 launch Entrance/launch/sim/faster_lio.launch.py bag:=/绝对路径/bag目录
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `lio_config` | `sim/` 或 `real/faster_lio.yaml` | 话题、预处理、估计参数 |
| `frames_config` | `sim/` 或 `real/frames.yaml` | 机身/雷达/IMU 外参、地图锚定与 TF |
| `output` | `outputs/gazebo_lio` 或 `outputs/unitree_real/lio` | 运行输出目录 |
| `bag` | 空 | 离线回放目录；需 `metadata.yaml` |
| `check_config` | `false` | 只检查配置 |
| `args` | 空 | 透传给 Faster-LIO，如 `-p pcd_save.pcd_save_en:=false` |

默认输出包括 `faster_lio.actual.yaml`（由 frames 与算法 YAML 合并）、bag 校验报告以及建图日志。建图节点在 `/faster_lio` 命名空间发布，公共 body TF 由 frame 桥接生成。

### 3.5 局部 cost 与推理

```bash
# 完整 cost 会话：局部预测 + 耗时监测 + 可选 RViz
ros2 launch Entrance/launch/sim/cost.launch.py model:=tiny rviz:=true

# 指定配置与输出目录；关闭耗时监测
ros2 launch Entrance/launch/sim/cost.launch.py config:=/绝对路径/local_cost.yaml output:=/tmp/pure3d_cost no_monitor:=true

# 只启动局部接收/预测节点（不含 monitor）
ros2 launch Entrance/launch/sim/local.launch.py model:=tiny mode:=cost rviz:=true
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `config` | `config/local_cost.yaml` | 网络、话题、盒子、KNN、模型注册表 |
| `model` | `cost`: `tiny`；`local`: 空（沿用 YAML） | `tiny` 或 `small` |
| `mode` | `cost`（仅 `local`） | `cost` 或 `cloud` |
| `rviz` | `false` | 打开 `local_cost.rviz` |
| `output` | `output_root/时间戳` | 会话输出目录；`cost` 要求目录尚不存在 |
| `no_monitor` | `false`（仅 `cost`） | 关闭耗时监测 |

`sim/cost.launch.py` 会在输出目录下生成 `local/` 与 `monitor/`，其中 `local/status.json` 表示首次有效代价；`sim/local.launch.py` 只保留单节点。

### 3.6 SCAN、导航与整图工具

```bash
# SCAN 单项（通常已由 sim/gazebo.launch.py 组合，避免重复启动）
ros2 launch Entrance/launch/sim/scan.launch.py flight_type:=4

# 人工目标与整图 A* 导航
ros2 launch Entrance/launch/sim/navigation.launch.py rviz:=true
ros2 launch Entrance/launch/sim/navigation.launch.py planner_backend:=dc

# 整图分块/代价工具
ros2 launch Entrance/launch/sim/whole_map.launch.py mode:=cost rviz:=true
ros2 launch Entrance/launch/sim/whole_map.launch.py mode:=blocks
```

`sim/scan.launch.py` 关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `use_sim_time` | `true` | 使用 Gazebo `/clock` |
| `flight_type` | `3` | 1 人工、2 预设航点、3 参考路径、4 cost 引导 |
| `manual_markers` | `auto` | `auto` 时模式 1/4 保留人工 marker |
| `path_sample_distance` | `auto` | 模式 4 用 0.1 m，其它模式 0.8 m |
| `path_topic` | 空 | 覆盖参考路径话题；模式 4 固定 `/pure_path` |
| `params_file` | 空 | 追加一份参数覆盖文件，后加载者优先 |
| `goal_fallback` | `false` | 是否允许目标回退 |
| `cost_astar_output` | 自动时间戳 | 模式 4 的跟踪输出目录 |
| `cost_astar_limit` | `0.8` | 必须在 `(0, 1]` |
| `cost_planner_backend` | `light_scan` | `light`、`light_scan` 或 `astar` |

`sim/navigation.launch.py` 支持 `config`、`planner_backend`（`surface`/`dc`）、`rviz`、`output`；`--goal x y z` 等未声明参数可通过 `args:="--goal 1 2 0.4"` 透传。`sim/whole_map.launch.py` 支持 `config`、`mode`（`cost`/`blocks`）、`rviz`、`output`。

### 3.7 显示与 TF 调试

```bash
# 仿真/真机建图显示（共用 scan.rviz）
ros2 launch Entrance/launch/sim/rviz.launch.py
ros2 launch Entrance/launch/real/rviz_real.launch.py

# 单独调试 world -> map 锚定与公共 TF
ros2 launch Entrance/launch/sim/map_tf.launch.py
ros2 launch Entrance/launch/real/map_tf_real.launch.py
```

RViz 入口支持 `rviz_config:=/绝对路径/布局.rviz`。正常建图流程已包含 map_anchor；单独运行 `map_tf*` 时应关闭原方案中的同一节点，避免重复发布。

### 3.8 真机 bag 回放

```bash
# 运行一次回放（默认 fastlio_official，默认 tiny）
ros2 launch Entrance/launch/real/real_bag.launch.py action:=run args:="--profile fastlio_official --model tiny --rviz"

# 指定 bag、倍率与输出目录
ros2 launch Entrance/launch/real/real_bag.launch.py action:=run args:="--bag /绝对路径/bag --profile jeplo --rate 0.5 --output /tmp/real_bag"

# 回放后查看结果
ros2 launch Entrance/launch/real/real_bag.launch.py action:=view args:="--run outputs/real_world/<profile>/<bag>/<时间戳>-tiny --rviz"
```

- `action`：`run`（默认）、`observe`、`view`。
- `run` 选项：`--bag`、`--profile`、`--model`、`--rate`、`--output`、`--rviz`。
- profile 对应 `config/real/<profile>_frames.yaml` 与 `<profile>_lio_overrides.yaml`；当前提供 `fastlio_official` 和 `jeplo`。
- 回放使用独立 ROS 域（默认 186）并限制 localhost，避免干扰正在运行的仿真。
- 输入 bag 需为单个 SQLite3 rosbag2，包含 `/livox/lidar` 与 `/livox/imu`，且不能自带 `/clock`。

### 3.9 历史 bag 任务工作流

```bash
# 生成任务计划
ros2 launch Entrance/launch/sim/mission.launch.py action:=plan args:="--output /tmp/pure3d_mission"

# 查看已有任务
ros2 launch Entrance/launch/sim/mission.launch.py action:=view args:="--mission /绝对路径/任务目录 --rviz"
```

`action` 可取 `plan`、`follow`、`record`、`validate`、`play`、`status`、`nodes`、`cost`、`merge`、`view`；其余选项通过 `args:="..."` 传给 `common.bag_tools.bag_mission`。其中 `cost` 子命令默认使用 `config/whole_map.yaml`，可用 `args:="--config /绝对路径/whole_map.yaml"` 覆盖。

### 3.10 独立跟踪与对照实验

```bash
# 只启动局部 A* 跟踪；output 必填
ros2 launch Entrance/launch/sim/follow.launch.py output:=/tmp/pure3d_follow cost_limit:=0.5 backend:=astar

# 本地上下文对照实验
ros2 launch Entrance/launch/sim/compare_context.launch.py args:="--output /tmp/pure3d_context --model tiny --repeats 10 --rviz"
```

`sim/follow.launch.py` 参数：`output`、`cost_limit`、`backend`（`astar`/`light`）；`--bridge-body-height`、`--light-update-hz` 等可通过 `args` 追加。`sim/compare_context.launch.py` 没有固定参数，底层支持 `--input`、`--output`、`--config`、`--model`、`--center`、`--repeats`、`--rviz`、`--view-only`。

## 4. 整套启动（四终端）

```bash
./Entrance/shell/build.sh
./Entrance/shell/total_terminal.sh --help
./Entrance/shell/total_terminal.sh --scene building
./Entrance/shell/total_terminal.sh --scene terrain-prepared --model tiny --flight-type 4
./Entrance/shell/total_terminal.sh --scene building --dry-run
```

常用选项：

| 选项 | 默认 | 说明 |
|---|---|---|
| `--scene` | `startup.yaml` 的 `default_scene` | 场景名 |
| `--model` | `tiny` | cost 模型：`tiny` 或 `small` |
| `--flight-type` | `4` | `1` 人工 SCAN；`4` cost 引导 |
| `--cost-limit` | `0.8` | 模式 4 的 A* 代价上限，`(0, 1]` |
| `--planner-backend` | `light_scan` | `light`、`light_scan` 或 `astar` |
| `--cost-config` | `config/local_cost.yaml` | cost 配置 |
| `--run-delay` | `3` | Gazebo 就绪后的墙钟等待（秒） |
| `--crouch-delay` / `--stand-delay` / `--mode-delay` | `3` / `4` / `3` | 仿真时钟上的模式等待（秒） |
| `--lio-delay` | `3` | 首次点云/TF 就绪后的墙钟等待（秒） |
| `--timeout` | `180` | 单步就绪检查超时（秒） |
| `--dry-run` | 关 | 只打印命令与场景解析，不清理、不启动 |

执行流程：

1. 清理本工作区、同一 `ROS_DOMAIN_ID` 的旧仿真/cost 会话；若残留进程无法退出则拒绝启动。
2. 打开 Gazebo 终端，等待四个控制器激活、里程计发布，并确认 `/clock` 与 `/cmd_vel` 各只有一个发布者。
3. 打开键盘终端（`publish_cmd_vel:=false`），按 Gazebo 日志确认执行 `2 → 2 → 7` 模式切换。
4. 打开 LIO 终端，等待配准点云与 `world → camera_init` TF 可用。
5. 打开 cost 终端，等待首帧有效代价；模式 4 还要求跟踪就绪、代价新鲜且 `/pure_path` 只有一个发布者。
6. 写入 `startup_status.json` 后启动脚本退出；四个终端相互独立，继续运行。

整套启动输出在 `outputs/entry_local/terminal_<时间戳>/`：

```text
terminal_<时间戳>/
├── stages/                 gazebo/keyboard/lio/cost 的阶段状态与退出码
├── gazebo.log              Gazebo 与 SCAN 组合日志
├── keyboard.log           键盘与模式切换日志
├── lio.log                 Faster-LIO 日志
├── cost/                  cost 会话输出（local/、monitor/）
├── processes.json         本次会话进程与配置清单
└── startup_status.json    ready / failed 状态与关键话题发布者数量
```

`outputs/entry_local/current_run.txt` 指向最近一次运行目录。`--help` 与 `--dry-run` 不加载 ROS overlay、不清理旧进程。

## 5. 配置修改指南

| 文件 | 作用 | 常见修改 |
|---|---|---|
| `config/startup.yaml` | 默认场景与所有场景共享的仿真默认值 | `default_scene`、`launch_arguments`（`gui`/`rviz`/`planner`/`flight_type`/`device`） |
| `config/scenes/*.yaml` | 每个场景的资源与出生点 | 新增场景只需增加 YAML，无需改 launch/shell |
| `config/local_planners.yaml` | 规划器共用话题映射 | `odom`、`cloud`、`path`、`cmd_vel` |
| `config/local_cost.yaml` | 局部 cost 推理参数 | 模型注册表、`checkpoint`、`device`、`network_python`、盒子/体素/KNN/上下文 |
| `config/scan.yaml` | SCAN FSM、栅格地图、优化与跟踪 | 模式、地图尺寸、膨胀、速度/加速度约束、控制增益 |
| `config/navigation.yaml` | 人工目标与整图规划 | `planner_backend`、`surface_slice_dh`、代价条件、输出目录 |
| `config/whole_map.yaml` | 整图分块与代价工具 | 分块尺寸、推理帧、到达判定、权重路径 |
| `config/sim/faster_lio.yaml`、`config/sim/frames.yaml` | 仿真 LIO 参数与外参/TF | 话题、`scan_line`、盲区、外参、锚定方式 |
| `config/real/faster_lio.yaml`、`config/real/frames.yaml` | 真机 LIO 参数与外参/TF | 同上；真机外参需按实测填写 |
| `config/real/<profile>_*.yaml` | bag 回放 profile | `fastlio_official`、`jeplo` 的帧/外参与 LIO 覆盖项 |
| `config/*.rviz` | RViz 布局 | 由各 launch 的 `rviz_config` 或 `rviz:=true` 加载 |

### 5.1 场景 YAML

`kind` 决定资源解析方式：

- `kind: model`：从 `model_candidates` 中选第一个含 `model.sdf` 的目录，`world_from_model` 相对模型目录解析；可用 `model_env` 指定自定义副本。
- `kind: world`：`world` 与 `models` 分别指定，`required_model_files` 在启动前检查。
- `kind: terrain`：`prepare: true` 时由 `source` 下的 `terrain.obj` 和 `scene.json` 导出到 `output`；`prepare: false` 直接复用导出目录。出生点读取 `output/spawn.json`。

所有场景都可在 `launch_arguments` 中覆盖公共默认值。`@assets/` 前缀表示相对 `unitree_gazebo` 包资源目录；其它相对路径以工作区根目录为基准。

示例：修改出生点只改场景 YAML：

```yaml
spawn: {x: -5.5, y: 6.0, height: 0.5, yaw: 0.0}
```

临时覆盖仍可用命令行：

```bash
ros2 launch Entrance/launch/sim/gazebo.launch.py scene:=building x:=0 y:=0 height:=0.5 yaw:=0
```

### 5.2 推理与代价配置

`local_cost.yaml` 中的 `models` 是模型注册表：

```yaml
model: small
models:
  small:
    checkpoint: weights/cost/.../best.pt
    expected_epoch: 56
    expected_encoder: utonia_small
```

`model:=tiny` 或 `model:=small` 会按名称选择；`checkpoint` 相对工作区根目录。`network_python` 必须是可执行的 Python 解释器（默认 conda 环境），`device` 为 `cuda:0` 或 `cpu`。权重缺失、epoch/encoder 不符或解释器不存在时会在启动前报错。

`whole_map.yaml`、`navigation.yaml`、`real_bag.yaml` 使用相同的权重字段约定；`real_bag.yaml` 的 `model` 默认为 `tiny`。

### 5.3 话题与 TF

- `local_planners.yaml` 是所有规划器共用的话题入口；不要在单个 launch 中再写死同名话题。
- LIO 在 `/faster_lio` 命名空间发布；公共 body TF 由 frame 桥接生成，原始 TF 重定向到 `/faster_lio/raw_tf`。
- 帧约定：`world → map → camera_init → body`；仿真由 `map_anchor.py` 用 Gazebo 真值建立 `world → map`，真机由 `initial_body`/`fixed` 锚定。
- 安装外参的变换约定为 `p_parent = R(rpy) * p_child + translation`，单位米/弧度，`rpy` 为固定轴 `Rz * Ry * Rx`。

## 6. 参数优先级与环境变量

优先级从高到低：

1. 命令行显式参数（`scene:=...`、`x:=...`、`--flight-type ...`）；
2. 场景 YAML 的 `launch_arguments`；
3. `startup.yaml` 的 `launch_arguments`；
4. launch 内置默认值（`scenes.py` 的 `ARGUMENT_DEFAULTS`）。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `PURE3D_SIM_SCENE` | 覆盖默认场景，优先级高于 `startup.yaml` 的 `default_scene` |
| `PURE3D_BUILDING_MODEL` | `building` 场景的自定义模型目录 |
| `PURE3D_TERRAIN_SCENE` / `PURE3D_TERRAIN_OUTPUT` / `PURE3D_TERRAIN_FLOOR` | terrain 源目录、导出目录、楼层 |
| `PURE3D_LIO_WS` | Faster-LIO 工作区（默认 `~/livox_ws`） |
| `PURE3D_SIM_LIO_CONFIG` / `PURE3D_REAL_LIO_CONFIG` | 仿真/真机 launch 的默认 `lio_config` |
| `PURE3D_LIO_CONFIG` | `lio_runtime.bash` 内部读取，通常由 launch 自动设置 |
| `PURE3D_LIO_FRAMES_CONFIG` | 默认安装外参/锚定 YAML |
| `PURE3D_LIO_OUTPUT` | 默认 LIO 输出目录 |
| `PURE3D_LIO_BAG_OUTPUT` | 离线 bag 输出目录 |
| `PURE3D_LIO_WAIT_TIMEOUT` / `PURE3D_IMU_GRAVITY_NORM` | IMU 等待超时与重力模长 |
| `PURE3D_REAL_UNDERLAY` | 真机硬件消息/驱动 overlay 的 install 目录 |
| `PURE3D_SIM_ROOT` / `LIBTORCH_ROOT` | 仿真构建根目录与 libtorch 路径 |
| `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` | ROS 域与 localhost 限制；整套启动与 bag 回放依赖它隔离会话 |

显式设置配置类环境变量时，它会作为对应 launch 参数的默认值；命令行参数仍然优先。

## 7. 数据流与运行关系

```text
Gazebo / MID360
  ├─ /livox/lidar + /livox/imu
  │     └─ Faster-LIO → /faster_lio/cloud_registered、/faster_lio/Odometry
  │                         └─ cost / whole_map / navigation → 代价与参考路径
  └─ /clock、/Odometry_gazebo、/cmd_vel
                          SCAN / EGO / 跟踪节点 → /cmd_vel
```

- `sim/gazebo.launch.py` 负责 Gazebo、A1、MID360 和所选局部规划器；`planner` 决定唯一的速度控制源。
- `sim/faster_lio.launch.py` 独立建图；`world → map → camera_init → body` 由 frames 配置与 `map_anchor.py` 建立。
- `sim/cost.launch.py` 订阅 LIO 点云/里程计，发布 `/entry/local/cost_cloud`；模式 4 的跟踪节点消费代价并发布 `/pure_path`。
- `sim/scan.launch.py` 把外部话题收拢到 `/scan` 命名空间，内部节点不直接占用公共话题。
- `real/` 下的同名功能使用真机配置与独立 ROS 域；`real/real_bag.launch.py` 不依赖 Gazebo。

## 8. 输出、日志与进程状态

| 路径 | 内容 |
|---|---|
| `outputs/gazebo_lio/` | 仿真 LIO 输出，含 `faster_lio.actual.yaml` |
| `outputs/unitree_real/lio/` | 真机 LIO 输出 |
| `outputs/entry_local/` | 局部 cost 与整套启动输出；`current_run.txt` 指向最近一次运行 |
| `outputs/entry_astar/mode4_*` | 模式 4 的 A* 跟踪输出 |
| `outputs/entry/` | `sim/whole_map.launch.py` 输出 |
| `outputs/entry_exploration/` | `sim/navigation.launch.py` 输出 |
| `outputs/real_world/<profile>/...` | 真机 bag 回放输出 |
| `.cache/gazebo_ws/ros_log/` | 仿真 ROS 日志 |
| `outputs/unitree_real/ros_log/` | 真机 ROS 日志 |

整套启动的每个阶段都有独立日志与状态文件；`processes.json` 记录 PID、命令和配置，`startup_status.json` 记录 `ready`/`failed` 与关键话题发布者数量。启动失败时只清理本次已打开的终端，不会误杀已经就绪的其它会话。

## 9. 常见问题排查

| 现象 | 处理 |
|---|---|
| 不确定参数名/默认值 | `ros2 launch <launch文件> --show-args` |
| 想先确认场景资源 | `sim/gazebo.launch.py scene:=... dry_run:=true`，只打印解析结果 |
| `Faster-LIO is not built` | 设置 `PURE3D_LIO_WS` 指向已构建的 `faster_lio` 工作区 |
| cost 启动即报权重/解释器错误 | 检查 `local_cost.yaml` 的 `models`、`network_python`、`device` 与权重路径 |
| 键盘无响应 | 必须在独立交互终端运行；重定向 stdin 会报 TTY 错误 |
| 等待控制器/点云/TF 超时 | 查看 `gazebo.log`/`lio.log`；确认 `/clock`、`/cmd_vel`、`/faster_lio/cloud_registered` 与 `world → camera_init` TF |
| `Old project processes still active` | 检查同一 `ROS_DOMAIN_ID` 是否有残留；先手动停止旧会话，或换一个域 |
| `terrain` 报源目录缺失 | 改用 `terrain-prepared`，或修正 `scenes/terrain.yaml` 的 `source` |
| RViz 保存后注释消失 | RViz 会重写 `.rviz`，这是预期行为；说明性注释仅作阅读参考 |
| bag 校验失败 | 确认单个 SQLite3、包含 `/livox/lidar` 与 `/livox/imu`、无自带 `/clock`、前 3 秒近似静止 |
| 真机/仿真话题互相干扰 | 用 `ROS_DOMAIN_ID` 隔离；`real_bag` 默认 186，仿真整体启动默认 0 |

排障顺序建议：先 `--show-args` 核对参数，再 `dry_run:=true`/`check_config:=true` 检查配置与资源，最后查看对应阶段的 `.log` 与 `startup_status.json`。

## 10. 使用约束与注意事项

1. **唯一速度源**：`scan`、`ego`、独立跟踪节点不要同时控制 `/cmd_vel`；键盘在 SCAN 模式下应设置 `publish_cmd_vel:=false`。
2. **模式 4 约束**：必须 `planner:=scan`，参考路径固定为 `/pure_path`，`cost_astar_limit` 位于 `(0, 1]`；后端可选 `light`、`light_scan`、`astar`。
3. **场景资源先检查**：新增或迁移场景时，先在 YAML 中声明 `required_model_files`/`source`，再用 `dry_run` 预览；不要直接改 shell 或 launch 的场景分支。
4. **配置分层**：默认值放 `startup.yaml`，场景差异放 `scenes/*.yaml`，临时实验走命令行；这样整套启动与单项启动始终一致。
5. **输出可追溯**：整套启动每次生成独立时间戳目录，调试完成后优先查看 `startup_status.json` 与各阶段日志。
6. **业务实现边界**：本目录只管理运行入口与集中配置；训练、数据生成和各模块内部参数仍由所属模块维护。
7. **RViz 注释**：`.rviz` 是 RViz 可重写的布局文件，注释可能被保存操作丢弃，不影响布局功能。
