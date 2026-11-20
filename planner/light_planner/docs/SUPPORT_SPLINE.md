# 轻量支撑面 B 样条

现有 Light 分层图一次搜索，按方向保留候选后沿用原规则排序。后端从世界系 XYZ 支撑路径拟合五次 B 样条，换轨时延续旧曲线的位置、速度、加速度和 jerk。结束处默认以零速度、零加速度和零 jerk 收尾。

先在水平弧长上按统一的水平速度、加速度和 jerk 约束确定时间并拟合平面样条。再沿同一时间轴，使用支撑高度和固定机身参考偏移独立拟合 Z，并从膨胀占用图收紧已观测净空。高度样条同时惩罚跟踪偏差和 jerk。高度求解失败时拒绝候选。最终按约 10 ms 复查水平运动学、Z 向 jerk、支撑高度范围和机身占用。

`scan_bspline_message()` 传递样条次数、节点向量和系数。flight type 4 的 `light` 使用独立 EGO 式轨迹服务器，跟踪器输出 yaw-only 的 `vx, vy, ωz`。运行时先验证支撑连接，再由轨迹服务器检查机身占用、限制水平指令并按旧轨迹进度切换。下面的独立试验仍可比较同一参考上的原生 SCAN 与轻量样条。

```bash
cd /home/cjh/Pure3D_ws
source Entrance/shell/_internal/sim_env.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m planner.light_planner.support_spline_compare  --input outputs/planner/local_candidates_20261117_v2  --output outputs/planner/support_spline_new --domain 187  --reference-mode rolling
```

再使用 `--reference-mode window` 和新输出目录，把参考端点限在共同的 1.5 m 窗口。两种实验都使用同一份保存的 PCD 生成参考，并把 PCD 的空间体素中心输入原生 SCAN v4。每组用独立参考话题。采集 SCAN 首条原生曲线与后续曲线，按旧轨迹推进给定的里程计，在换轨时比较位置、速度和加速度跳变。轻量后端使用相同换轨时刻及自身旧曲线状态。高度误差按 XY 投影到所选参考后计算。输出含参考及轨迹 CSV、样条系数、原生节点日志、汇总 JSON 和高度速度对照图。

该试验回放的是保存的聚合点云，传感器位姿和里程计由脚本生成。记录的曲线、控制消息和失败原因属于离线验证，不能作为机器人实际通过楼梯的证据。
