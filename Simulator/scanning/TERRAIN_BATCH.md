# 大 terrain 上的 VST 随机路线与合批扫描

```text
现有 WFC terrain Mesh + 原生 VST
    → 整图随机起终点 → 原生 A* + GPMP
    → 保留路径选定的楼层，三维连续轨迹 + IMU
    → 同时刻多轨迹射线 [轨迹, footprint, ray] 合批
    → 一次 Warp launch → 按轨迹拆分原始扫描
    → 独立 ROS 域 + 独立 Faster-LIO
    → 每条 rough.pcd + Filter 评分监督 + Reconstruction 完整参考
```

## 运行

配置为同目录 `terrain_batch_config.json`。默认复用
`datasets/wfc_capability_dr_v2/scene_20261109` 的三层建筑和 `vst_nominal.pickle`。
现有 Mesh、PCD 和 VST 只读，不重新修改教师参数。A* 阈值为 15。

在工作区根目录执行完整路径采集：

```bash
cd /home/cjh/Pure3D_ws
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Simulator.scanning.terrain_batch
```

本地小规模验收命令：

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Simulator.scanning.terrain_batch \
  --frames 200 \
  --output datasets/scanning_terrain_batch_20261113/my_prefix20 \
  --report-output outputs/scanning_terrain_batch_20261113/my_prefix20
```

`--frames 200` 是每条轨迹最多前 20 秒，保持实际运动速度，不把整条长路径挤进 20 秒。
短于该时间的路径按完整路径采集。省略该参数即采完整路径。
连续轨迹采用五次时间进度，位置、姿态至少二阶连续，由同一轨迹导出 IMU。
按最大速度 0.9 m/s、最大偏航角速度 67.5°/s 分别计算每条路线的运动时长，
每条路线使用自己的完整时长。只同步仍在采集的路线的射线时刻。
短路线到达终点后退出射线批次，不停在终点重复扫描，也不被长路线拖慢。
不增加初末固定静止时段。

`--stages plan scan replay prepare` 可分阶段调用。后续阶段读取各自的实际配置。
每次新实验使用新目录，既有采集不自动覆盖。规划失败会重新选点。

## 随机规划和楼层

配置要求四条成功随机路线，随机种子 821111。
起点从整图所有有限且 cost ≤ 15 的唯一地表 XYZ 均匀采样。
终点从距起点 6–16 m 的可通行支持点随机采样，距离采用三维距离，不限制设施。
同一表面在多个 slice 重复出现时只采一次。
每次选定的起终点只规划一次。A* 或 GPMP 返回失败后，重新随机选取起点和终点，
直到凑齐四条成功路线。失败端点和原因即时写入 routes.json 的 attempt_history，
各成功路线保留 failed_attempts。保持原代价阈值，不通过提高阈值放行。
`capture_routes=4` 采集全部四条成功路线，不读取失败尝试的路径文件。

规划调用保留的原生 VST A* + 五次 GPMP。
GPMP 世界 XYZ 经最近 ground 高度确定其支持楼层，生成连续三维支撑路径。
采集轨迹在支撑面上方 0.8 m，LiDAR 与共轴 IMU 都采用 pitch=0.785 rad。
多层建筑不再从顶层向下按 XY 找地面，避免误选到另一层。

原生模块由 `planner_python` 和 `native_dir` 指定，二者 Python ABI 必须一致。
本机使用 `/usr/bin/python3` 与 `datasets/cache/vst_native_py310` 中已编译的 Python 3.10 扩展，
并保留 BUILD_INFO.json。缓存内二进制不进入 Git。迁移时配置服务器已构建的原生库路径，
或用 `third_party_lib/vst_global_planner/native` 源码在目标环境构建。

## GPU 合批和流水线

`ray_batch_routes=2` 让两条轨迹同一帧的射线共享一个静态 Mesh 和一次 Warp launch。
每条仍为 10 Hz、每帧 10,000 条主射线。mixed 配置另查询四条 footprint 辅助射线，
两条轨迹对应一次 100,000 射线查询，然后按固定偏移拆分。辅助射线不当作额外 LiDAR 点导出。
逐点保留 ray_id、offset_ns、trajectory_id、命中面片和实例。

每条路线采完后立即启动其 LIO，GPU 同时继续采集批内其他路线或下一批。默认四条成功路线分为两批，
每批两条一起查询。独立 ROS_DOMAIN_ID 保留成功路线编号，分别为 85、86、87、88。
四条 LIO 可同时处理。正式监督及分块使用一个 CPU 消费者，避免本机重复读取大参考点云。
`replay_workers` 和 `prepare_workers` 可配置。

长序列回放使用 `max_inflight_scans=32`，按 LIO 已完成的里程计帧控制输入窗口。
IMU 和 LiDAR 共同推进，避免长时间按 10×发送导致两个 DDS 队列分别积压并失去对应区间。
所有传感器时间戳保持不变。10×仍是发送目标，实际发送速度受 LIO 消费速度限制，
等待时间单独记入 replay.json 的 flow_control_wait_s。

采集保留 MID-360 方向表、逐点时序、IMU 测量误差、动态物体、显式悬浮回波和边缘混合回波。
IMU 采样时间严格为 200 Hz，不注入偏移或抖动。加速度相关噪声使用有界 tanh 非线性，
参数和公式见同目录 README。陀螺仪噪声保持原配置。
运动物体沿本批支持路径移动。显式悬浮仍是原始测量拉近扰动，不声称已标定传感器物理模型。

## 数据和检查

每条目录使用原有 `mixed/scans`、`registered`、`rough.pcd`、`attributes.npz`、
`arrays_000`、`windows_000_*` 和 `training_pair.json`。
rough.pcd 仅含 XYZ，不额外降采样。teacher 的完整静态参考复用 terrain_downsampled.pcd，
正式逐点评分始终查询 Mesh 三角面，不查询 GT PCD 最近点。

`gt_pcd_diagnostics=false` 关闭整张 GT PCD 和局部扫描间的双向最近邻诊断。
此项在大场景中耗时很大，与正式 Filter 标签和 Reconstruction 完整目标无关。
需要专门检查采样误差时再开启。原来小场景配置默认仍保留诊断。

Filter 使用原有窗口接口。Reconstruction 的配对数据通过
`Simulator.scanning.training_data.SharedScanBoxes` 读取，模型输入仍只有 XYZ。
未开始网络训练或生成旧 Reconstruction 训练器所需的特征缓存。
所有路径属于同一个建筑划分，不能跨 train/val/test 分配。

生成时直接核对起点、中段的合批与逐路查询结果，包括 XYZ、距离、未命中、面片和来源。
正式采集调用次数与帧数相同，不是帧数乘轨迹数。不会新增自动测试框架。
不同路线帧数不同时，调用次数为该组最长路线的帧数，射线量按各路线的实际帧数求和。
ray_batch_*.json 保存各路线独立时长、帧数以及 active_batch_transitions。

```bash
MPLCONFIGDIR=/tmp/pure3d_scanning_mpl \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Simulator.scanning.terrain_plot \
  --report docs/reports/scanning_terrain_batch_20261113/prefix20_fast/report.json
```

图包括整图随机路线、实际采集段、扫描与两类监督、流水线时间。
直接检查包含逐点标签公式、轨迹身份、PCD 字段、窗口对应和唯一核心区归属。
首批已跑通三条独立 PCD，但 route_000 的 LIO 轨迹 RMSE 约 1.31 m，
route_002 约 2.48 cm，route_003 约 18.59 cm。生成流程通过不等于每条数据质量合格，
本次不自动改变噪声、阈值或剔除策略。

完整路线的楼梯覆盖核验：

```bash
MPLCONFIGDIR=/tmp/pure3d_scanning_mpl \
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m Simulator.scanning.terrain_coverage \
  --report docs/reports/scanning_terrain_batch_20261113/independent_full_v2_replay/report.json
```

该入口核对完整扫描帧数和末帧，绘制实际运动距离与高度曲线。
将原始回波的 face_id 与生成器连接结构的原始三角形匹配，统计配准 PCD 中实际命中的楼梯和坡道。
统计排除运动实例、显式悬浮和混合回波。它衡量结构是否被观测，不代表 LIO 精度。

## RViz 查看四组查询

系统 Python 在 ROS Humble 环境运行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=95
export PYTHONPATH=/home/cjh/Pure3D_ws:${PYTHONPATH:-}
/usr/bin/python3 -m Simulator.scanning.route_rviz \
  --routes datasets/scanning_terrain_batch_20261113/prefix20_fast/routes/routes.json \
  --output outputs/scanning_terrain_batch_20261113/rviz_all_queries --height-offset 0.5
```

四组查询按编号着色为青、红、紫、黄色。历史报告中的失败查询只显示起终点，
新重选报告中的成功查询显示完整 GPMP 路线。
圆球为起点，方块为终点。路径和端点显示抬高 0.5 m，原始坐标不变，terrain 和 VST 不抬高。
所有图层分别开关，Views 中保留完整场景、失败查询近景和俯视相机。
