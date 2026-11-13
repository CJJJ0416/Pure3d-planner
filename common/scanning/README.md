# Faster-LIO 增量扫描数据

输入为带身份的静态 Mesh 和运动物体，输出为真实 Faster-LIO 算法配准后的逐点观测。RayCaster 是仿真测量，数据不代表硬件采集。模型输入仍只有单张地图系 XYZ。

## 运行

在工作区根目录使用 Pure3D Python。生成与监督使用该环境，ROS 回放由入口自动切换到系统 Python，加载只读的 `livox_ws` underlay。主配置为同目录 `config.json`。

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m common.scanning.pipeline
```

当前默认速度为原运动段的1.5倍，初末静止均为0秒。相同路线的运动时间
为50/1.5≈33.33秒，以10 Hz保存333个完整扫描帧（约33.3秒）。
`motion_speed_scale` 同时缩放位置与转向时序，IMU由新的连续轨迹重算。
lio_error和mixed启用既有白噪声、偏置、相关抖动及时间噪声。

只采集当前轨迹前20秒时使用以下命令，另选输出目录以保留历史对照：

```bash
/home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m common.scanning.pipeline \
  --profiles coverage --frames 200 \
  --output datasets/filter_fasterlio_scan_v1/seed_821110/fast15_prefix20
```

`--stages generate replay prepare plot` 可选择单独阶段。失败后不自动重试。检查输出后手动调用需要的阶段。

## 逐点时间与来源

每帧 **10,000 射线、10 Hz**，即100,000条候选射线/秒。每帧仍推进原方向表的
24,000行，在该区间内均匀取10,000条，避免减少射线时也把方向图案推进速度变慢。
该跨度由pattern_samples_per_scan控制，实际射线预算由rays_per_scan控制。
点的offset_time覆盖原0.1秒帧周期，使用纳秒、半步采样并严格递增。
MID-360表只提供方向，时间为人工均匀赋值，不声称还原物理雷达时序。
IMU 200 Hz，静止比力模长为9.81 m/s²。两个传感器共点同轴。

当前安装角取自 A1 的 robot.xacro / laser_livox_joint：rpy=[0,0.785,0] rad，
即约45°前倾。传感器位置仍沿用本数据集的支撑面上方0.8 m，不引入A1机身平移。
LiDAR与共轴IMU共同施加安装旋转，IMU比力和角速度均转换到旋转后的IMU坐标。
静止比力约为[-6.934,0,6.939] m/s²。LIO外参保持单位变换，不再次施加45°。
原先 short_alignment 和 short_imu_blind 为水平安装，新的短测使用 short_pitch45。

方向与模型资产统一复用 `common/Mid360_imu_sim`，默认直接读取其中的
`scan_mode/mid360.csv`，不依赖旧 glocal2-nav 目录。原 Gazebo C++ 插件是
瞬时 ODE 扫描，逐点时间全为零，不能直接代表当前滚动扫描。新流程复用方向
和 Livox 消息约定，Warp 查询按每点时间更新传感器与物体位姿。原插件保持原样。

本地 Faster-LIO 的 `PointBodyToWorld` 只复制 XYZ 和 intensity，不复制 curvature。关闭输入抽点，开启 dense 输出后，发布点序仍为原始逐点时间排序。导出器按该时间排序关联来源，同时检查完整点数、唯一递增时间和逐点强度序列。合成 reflectivity 为 ray_id%251+1，仅用于顺序核验，不携带标签，不进入 Filter。

每个 `registered/NNNNN.npz` 保存 LIO XYZ、ray_id、offset_ns 和配准帧时间，和同编号原始扫描精确关联。初始化及初始地图种子帧可能不发布，不能将其算作已导出的扫描。跳过清单与原生日志保存在 replay.json。

## GT 如何对应

重复扫描允许同一表面对应多个观测点。GT PCD 与扫描点不要求点数、点序或坐标完全相同，不能做精确 XYZ 查表或逐行相减。

1. 用首个配准 IMU 位姿与同一时刻真值建立唯一 `map_from_lio`，后续固定。
2. 扫描点到 Mesh 三角表面的距离用于正式连续 GT。运动物体来源点固定为 0，其他点为 exp(-0.5*(d/0.03)^2)。
3. 另计算扫描点到 Mesh 采样 PCD 的最近点距离，用于检查采样误差。该距离包含 GT 点间隔误差，不直接用于正式标签。
4. 计算 GT PCD 到真值融合扫描的反向距离，描述遮挡与未覆盖表面。反向距离大不表示已有观测点错误。
5. 同一射线的真值融合 XYZ 与 LIO XYZ 可直接相减，量化 LIO 额外引入的误差。静态 Mesh 和原始扫描的来源身份仅用于教师。

## 文件与分块

场景级：`static_mesh.ply`、面片身份、`clean.pcd`、连续真值轨迹、方向表身份和实际配置。

每个 profile：原始 scans、IMU、LIO 日志与参数、估计轨迹、registered 增量扫描、增量 offsets、rough.pcd、truth_fused.pcd、gt_scored.pcd、同点序 attributes.npz、共享数组和核心/上下文索引。兼容 Filter 的 arrays_000 和 windows_000 接口。核心 4×4×3 m、上下文 1 m，每个原始点只属于一个核心，不额外降采样。

plots 保存相同视角的 GT/真值融合/LIO 距离图、距离分布以及每 10 秒增量图。全部点参与投影，距离显示采用像素内平均值，不改变保存的点云或标签。

RViz 使用系统 Python，发布后在 RViz 设置 Fixed Frame=map，PointCloud2 的 Channel Name=keep_target，0 为红、1 为绿。clean、rough、truth_fused、dynamic、edge_returns、static_controls 可分别开关：

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/cjh/Pure3D_ws:${PYTHONPATH}
/usr/bin/python3 -m common.scanning.rviz \
  --root datasets/filter_fasterlio_scan_v1/seed_821110/fast15_no_idle --profile lio_error
# 另一个终端，使用相同 ROS_DOMAIN_ID：
rviz2 -d common/scanning/dataset.rviz
```

本入口不启动训练，不修改已有 Filter 权重，不执行 ICP、逐帧真值纠正、自动加大扰动或失败回退。

## Isaac Sim 原生查看

使用 Isaac Sim 环境打开完整配准扫描与 GT PCD，紫红色为扫描、绿色为 GT。
右侧可分别切换点云与 Mesh、拖动累计时间、播放增量扫描及切换楼梯近景。
原始点数、点序与地图坐标保持不变，原生截图和 USD 保存到 profile 的 isaac_view。

```bash
/home/cjh/miniconda3/envs/env_isaaclab/bin/python -m common.scanning.isaac_view \
  --root datasets/filter_fasterlio_scan_v1/seed_821110/fast15_no_idle --profile lio_error
```

## IMU 抖动与 MID-360 盲区

`lio_error`、`mixed` 启用白噪声、固定偏置、时间偏差和 OU 相关测量抖动。
抖动相关时间 0.04 s，逐轴加速度 RMS 0.08 m/s²、角速度 RMS 0.008 rad/s。
这些是当前仿真扰动参数，不是 ICM40609 标定值。保存理想 IMU 与带噪测量，
用于对照。测量抖动不会偷偷改变真值轨迹或制造不一致的身体位移。
coverage、dynamic、edge_returns 保留理想 IMU，作为独立成因对照。

按 [Livox 官方参数](https://www.livoxtech.com/mid-360/specs) 约束近距盲区 0.1 m、
水平 360°、传感器系竖直 -7° 到 52°。对应 Faster-LIO blind 同步设为 0.1 m。
原方向表已经覆盖该竖直范围，不另外随机删除扇区。未发射方向不会伪造原始射线。
已有候选射线的 return_status 为 0 未命中/超距、1 有效、2 近距盲区、3 视场外。
Mesh 遮挡通过最近命中自然产生，不把看不见的后方表面写成回波。

水平安装、高 0.8 m 时，最低 -7° 射线对应平地近区约 6.52 m 不可见。
头顶也存在视场外区域。它们随安装姿态变化，不是额外人为删除的噪点。
当前改为 pitch=0.785 rad 后，前向截面的最低射线约为-51.98°，同一高度下
前方地面近区约0.63 m不可见。倾斜后的覆盖随方位角变化，不能把该数值当作
所有方向相同的盲区半径。
盲区不产生 keep_target=0 的点，只减少观测。完整 Mesh 仍可用于后续 RECON。
当前批量采集预算为10,000射线/帧，比此前24,000条减少58.3%。有效回波通常
少于候选射线数。历史config.actual.json和已采集数据不作覆盖修改。
方向表、均匀时序及简化回波不构成完整硬件数字孪生。
