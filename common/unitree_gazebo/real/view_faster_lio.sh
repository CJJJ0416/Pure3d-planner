#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec /opt/ros/humble/bin/ros2 run rviz2 rviz2 \
  -d "$PURE3D_UNITREE_ROOT/asset/scan_rviz.rviz" \
  --ros-args -r __node:=faster_lio_rviz -p use_sim_time:=false "$@"
