#!/usr/bin/env bash
# 整体构建：安装 A1、MID360、SCAN 与集中管理的启动文件。
# Usage（工作区根目录）：./Entrance/shell/build.sh
# 构建目录：.cache/gazebo_ws；启动入口与场景配置：Entrance/。
set -eo pipefail
# 只准备构建环境：不 source 尚未生成的 install 目录，否则首次构建会失败。
PURE3D_SIM_BUILD_ONLY=1 source "$(dirname -- "${BASH_SOURCE[0]}")/_internal/sim_env.bash"
cd "$PURE3D_SIM_WS"
# 默认限制 2 个并行编译任务，避免 Gazebo/SCAN 编译占满内存；调用者可覆盖 MAKEFLAGS。
export MAKEFLAGS="${MAKEFLAGS:--j2}"
# --base-paths 只包含本导航框架依赖的包；--packages-up-to 连同依赖一起构建，
# 输出放到工作区缓存的 build/install/log，使用软链接安装以便 launch 立即看到源码改动。
exec /usr/bin/python3 -m colcon --log-base "$PURE3D_SIM_ROOT/log" build \
  --build-base "$PURE3D_SIM_ROOT/build" --install-base "$PURE3D_SIM_ROOT/install" \
  --symlink-install --executor sequential \
  --base-paths Simulator/livox_ros_driver2 Simulator/Mid360_imu_sim \
    Simulator/quadruped_ros2_control Simulator/unitree_gazebo Simulator/SCAN-Planner/src/planner \
  --packages-up-to unitree_gazebo keyboard_input scan_planner \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DLIBTORCH_DIR="$LIBTORCH_ROOT" "$@"
