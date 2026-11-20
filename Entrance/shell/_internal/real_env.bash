#!/usr/bin/env bash
# 真机内部环境库；加载 ROS、LIO 与可选硬件消息 overlay。
# Usage：由真机 launch 自动加载；调试可 source Entrance/shell/_internal/real_env.bash。
# LIO 与安装参数：Entrance/config/real/*.yaml。
# PURE3D_WS 由本文件位置上溯三级得到，即 Pure3D_ws 工作区根目录。
PURE3D_WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PURE3D_WS
export PURE3D_UNITREE_ROOT="$PURE3D_WS/Simulator/unitree_gazebo"
export PURE3D_LIO_MODE=real
# Faster-LIO 默认取自用户工作区 livox_ws，可用 PURE3D_LIO_WS 覆盖。
export PURE3D_LIO_WS="${PURE3D_LIO_WS:-$HOME/livox_ws}"
# 只保留系统 ROS、Faster-LIO 工作区与系统库路径，过滤其它工程遗留的 overlay。
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
# 清理循环用到的临时变量，避免污染子进程环境。
unset real_path_name real_path_value real_path_part real_path_parts COLCON_CURRENT_PREFIX AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
if [[ -f "$PURE3D_LIO_WS/install/local_setup.bash" ]]; then
  # 叠加 Faster-LIO 工作区；未构建时保留系统 ROS 并让 lio_runtime.bash 报出缺失。
  source "$PURE3D_LIO_WS/install/local_setup.bash"
fi
# Optional hardware message/driver overlay; source configuration only.
# 真机硬件消息/驱动 overlay 由调用者显式提供，不在本文件中写死路径。
if [[ -n "${PURE3D_REAL_UNDERLAY:-}" ]]; then
  source "$PURE3D_REAL_UNDERLAY/install/local_setup.bash"
fi
# 真机日志单独存放，便于与仿真日志区分。
export ROS_LOG_DIR="$PURE3D_WS/outputs/unitree_real/ros_log"
mkdir -p "$ROS_LOG_DIR"
