#!/usr/bin/env bash
# Source only: keep the simulator overlay separate from perception environments.
PURE3D_SIM_WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PURE3D_WS="$PURE3D_SIM_WS"
export PURE3D_UNITREE_ROOT="$PURE3D_WS/common/unitree_gazebo"
export PURE3D_LIO_MODE=sim
# Independently opened simulation terminals must use the same ROS transport.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PURE3D_SIM_ROOT="${PURE3D_SIM_ROOT:-$PURE3D_SIM_WS/.cache/gazebo_ws}"
export LIBTORCH_ROOT="${LIBTORCH_ROOT:-$HOME/third_party_lib/libtorch}"
# Discard other project overlays; use system ROS plus this workspace only.
for sim_path_name in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH GAZEBO_MODEL_PATH GAZEBO_PLUGIN_PATH GAZEBO_RESOURCE_PATH; do
  sim_path_value=""
  IFS=: read -r -a sim_path_parts <<< "${!sim_path_name:-}"
  for sim_path_part in "${sim_path_parts[@]}"; do
    case "$sim_path_part" in
      "$PURE3D_SIM_ROOT"/*|"$LIBTORCH_ROOT"/*|/opt/ros/humble|/opt/ros/humble/*|/usr/*|/lib/*|/lib64/*)
        sim_path_value="${sim_path_value:+$sim_path_value:}$sim_path_part" ;;
    esac
  done
  export "$sim_path_name=$sim_path_value"
done
unset sim_path_name sim_path_value sim_path_part sim_path_parts COLCON_CURRENT_PREFIX AMENT_CURRENT_PREFIX
# ROS setup scripts are not safe under nounset.
source /opt/ros/humble/setup.bash
if [[ "${PURE3D_SIM_BUILD_ONLY:-0}" != 1 && -f "$PURE3D_SIM_ROOT/install/local_setup.bash" ]]; then
  source "$PURE3D_SIM_ROOT/install/local_setup.bash"
fi
export ROS_LOG_DIR="$PURE3D_SIM_ROOT/ros_log"
export GAZEBO_MODEL_DATABASE_URI=""
mkdir -p "$ROS_LOG_DIR"
