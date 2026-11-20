#!/usr/bin/env bash
# 仿真内部环境库；只加载本工作区 ROS overlay，与训练环境隔离。
# Usage：由 launch 自动加载；调试可 source Entrance/shell/_internal/sim_env.bash。
# LIO 与安装参数：Entrance/config/sim/*.yaml。
# PURE3D_SIM_WS 由本文件位置上溯三级得到，即 Pure3D_ws 工作区根目录。
PURE3D_SIM_WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PURE3D_WS="$PURE3D_SIM_WS"
export PURE3D_UNITREE_ROOT="$PURE3D_WS/Simulator/unitree_gazebo"
export PURE3D_LIO_MODE=sim
# Independently opened simulation terminals must use the same ROS transport.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
# 仿真构建/安装根目录与 libtorch 位置；都允许外部环境变量覆盖。
export PURE3D_SIM_ROOT="${PURE3D_SIM_ROOT:-$PURE3D_SIM_WS/.cache/gazebo_ws}"
export LIBTORCH_ROOT="${LIBTORCH_ROOT:-$HOME/third_party_lib/libtorch}"
# Discard other project overlays; use system ROS plus this workspace only.
# 逐个拆分 PATH 型变量，只保留系统 ROS、本仿真 overlay 与 libtorch 路径，过滤其它工程残留。
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
# 清理循环用到的临时变量，避免污染子进程环境。
unset sim_path_name sim_path_value sim_path_part sim_path_parts COLCON_CURRENT_PREFIX AMENT_CURRENT_PREFIX
# ROS setup scripts are not safe under nounset.
source /opt/ros/humble/setup.bash
if [[ "${PURE3D_SIM_BUILD_ONLY:-0}" != 1 && -f "$PURE3D_SIM_ROOT/install/local_setup.bash" ]]; then
  # 构建模式下跳过本工作区 install，正常运行则叠加到系统 ROS 之上。
  source "$PURE3D_SIM_ROOT/install/local_setup.bash"
fi
# 仿真日志固定写入缓存目录，禁止 Gazebo 访问在线模型库，保证离线可复现。
export ROS_LOG_DIR="$PURE3D_SIM_ROOT/ros_log"
export GAZEBO_MODEL_DATABASE_URI=""
mkdir -p "$ROS_LOG_DIR"
