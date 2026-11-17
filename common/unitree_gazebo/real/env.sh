#!/usr/bin/env bash
# Hardware environment; it does not load the Gazebo or training overlay.
PURE3D_WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PURE3D_WS
export PURE3D_UNITREE_ROOT="$PURE3D_WS/common/unitree_gazebo"
export PURE3D_LIO_MODE=real
export PURE3D_LIO_WS="${PURE3D_LIO_WS:-$HOME/livox_ws}"
for real_path_name in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH; do
  real_path_value=""
  IFS=: read -r -a real_path_parts <<< "${!real_path_name:-}"
  for real_path_part in "${real_path_parts[@]}"; do
    case "$real_path_part" in
      "$PURE3D_LIO_WS"/*|/opt/ros/humble|/opt/ros/humble/*|/usr/*|/lib/*|/lib64/*)
        real_path_value="${real_path_value:+$real_path_value:}$real_path_part" ;;
    esac
  done
  export "$real_path_name=$real_path_value"
done
unset real_path_name real_path_value real_path_part real_path_parts COLCON_CURRENT_PREFIX AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
if [[ -f "$PURE3D_LIO_WS/install/local_setup.bash" ]]; then
  source "$PURE3D_LIO_WS/install/local_setup.bash"
fi
# Optional hardware message/driver overlay; source configuration only.
if [[ -n "${PURE3D_REAL_UNDERLAY:-}" ]]; then
  source "$PURE3D_REAL_UNDERLAY/install/local_setup.bash"
fi
export ROS_LOG_DIR="$PURE3D_WS/outputs/unitree_real/ros_log"
mkdir -p "$ROS_LOG_DIR"
