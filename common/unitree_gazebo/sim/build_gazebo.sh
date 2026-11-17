#!/usr/bin/env bash
set -eo pipefail
PURE3D_SIM_BUILD_ONLY=1 source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PURE3D_SIM_WS"
export MAKEFLAGS="${MAKEFLAGS:--j2}"
exec /usr/bin/python3 -m colcon --log-base "$PURE3D_SIM_ROOT/log" build \
  --build-base "$PURE3D_SIM_ROOT/build" --install-base "$PURE3D_SIM_ROOT/install" \
  --symlink-install --executor sequential \
  --base-paths common/livox_ros_driver2 common/Mid360_imu_sim \
    common/quadruped_ros2_control common/unitree_gazebo common/SCAN-Planner/src/planner \
  --packages-up-to unitree_gazebo keyboard_input scan_planner \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DLIBTORCH_DIR="$LIBTORCH_ROOT" "$@"
