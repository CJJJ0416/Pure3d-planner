#!/usr/bin/env bash
# LIO 内部运行实现，由 sim/real 的 launch 调用，不作为用户启动入口。
# Usage：ros2 launch Entrance/launch/sim/faster_lio.launch.py（真机使用 Entrance/launch/real/faster_lio_real.launch.py）。
# 配置 YAML 由所属 launch 的 lio_config 与 frames_config 参数指定。
set -eo pipefail
# 必须由 sim/real 环境库启动，保证 PURE3D_UNITREE_ROOT 与模式一致。
: "${PURE3D_UNITREE_ROOT:?Use the sim or real Faster-LIO entry point}"
: "${PURE3D_LIO_MODE:?Use the sim or real Faster-LIO entry point}"
lio_bag=''
# 离线 bag 流程：第一个参数只识别 --bag，其余参数原样留给 run_mapping_online。
if [[ "${1:-}" == '--bag' ]]; then
  if [[ $# -lt 2 ]]; then echo 'Usage: ros2 launch Entrance/launch/sim/faster_lio.launch.py bag:=<rosbag2目录>' >&2; exit 1; fi
  lio_bag="$(realpath "$2")"
  shift 2
  # bag 目录必须包含 rosbag2 元数据；该离线流程只在仿真模式、独立 ROS 域中使用。
  if [[ ! -f "$lio_bag/metadata.yaml" ]]; then echo "Bag metadata missing: $lio_bag" >&2; exit 1; fi
  if [[ "$PURE3D_LIO_MODE" != sim ]]; then echo '--bag is configured for the Pure3D simulation capture contract' >&2; exit 1; fi
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-184}"
  if [[ "$ROS_DOMAIN_ID" != 184 ]]; then echo 'Use ROS_DOMAIN_ID=184 for this offline bag workflow' >&2; exit 1; fi
  # 回放输出固定放在 bag 同级的 offline/lio，避免覆盖在线建图结果。
  export PURE3D_LIO_OUTPUT="${PURE3D_LIO_BAG_OUTPUT:-$(dirname -- "$lio_bag")/offline/lio}"
fi
# Faster-LIO 安装前缀与可执行文件检查；未构建时给出明确的设置提示。
lio_ws="${PURE3D_LIO_WS:-$HOME/livox_ws}"
lio_prefix="$lio_ws/install/faster_lio"
if [[ ! -x "$lio_prefix/lib/faster_lio/run_mapping_online" ]]; then
  echo "Faster-LIO is not built: $lio_prefix. Set PURE3D_LIO_WS to its workspace." >&2
  exit 1
fi
# 叠加 Faster-LIO 自身的运行环境；部分第三方依赖放在 sysroot 中，需要补充库搜索路径。
source "$lio_prefix/share/faster_lio/local_setup.bash"
lio_libs="$lio_ws/third_party/sysroot/usr/lib/x86_64-linux-gnu"
if [[ -d "$lio_libs" ]]; then
  export LD_LIBRARY_PATH="$lio_libs:${LD_LIBRARY_PATH:-}"
fi
# 输出目录、安装外参与算法参数统一转成绝对路径，后续生成的配置也放在输出目录。
lio_output="$(realpath -m "$PURE3D_LIO_OUTPUT")"
export PURE3D_LIO_OUTPUT="$lio_output"
lio_frames="$(realpath "$PURE3D_LIO_FRAMES_CONFIG")"
lio_config="$(realpath "$PURE3D_LIO_CONFIG")"
mkdir -p "$lio_output"
cd "$lio_output"
# 按 frames YAML 生成实际生效的 Faster-LIO 参数：包含外参、IMU 话题与 TF 帧名。
/usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/frame_config.py" \
  --config "$lio_frames" --lio-config "$lio_config" --profile "$PURE3D_LIO_MODE" \
  --output "$lio_output/faster_lio.actual.yaml"
# --check-config 只生成并报告配置，不订阅传感器、不启动建图。
if [[ "${1:-}" == '--check-config' ]]; then
  printf 'Profile: %s\nFrames: %s\nParameters: %s\n' "$PURE3D_LIO_MODE" "$lio_frames" "$lio_output/faster_lio.actual.yaml"
  exit 0
fi
if [[ -n "$lio_bag" ]]; then
  # 离线回放：先校验 bag 与 mission 契约，然后等待另一个进程回放 bag（不在此处启动播放器）。
  lio_bag_args=(--bag "$lio_bag" --report "$lio_output/bag_check.json")
  if [[ -f "$(dirname -- "$lio_bag")/mission.json" ]]; then
    lio_bag_args+=(--mission "$(dirname -- "$lio_bag")/mission.json")
  fi
  PYTHONPATH="$PURE3D_WS:${PYTHONPATH:-}" /usr/bin/python3 -m common.bag_tools.bag_mission validate "${lio_bag_args[@]}"
  printf 'Bag mode validated; waiting for a separate bag player in ROS_DOMAIN_ID=%s.\n' "$ROS_DOMAIN_ID"
else
  # 在线模式：从生成后的参数中读取 IMU 话题，等待其发布且重力模长合理后再启动建图。
  lio_imu_topic="$(/usr/bin/python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["/**"]["ros__parameters"]["common"]["imu_topic"])' "$lio_output/faster_lio.actual.yaml")"
  # 仿真默认不等待（传感器随 Gazebo 启动），真机默认最多等 120 s；可用环境变量覆盖。
  lio_wait_default=120
  if [[ "$PURE3D_LIO_MODE" == sim ]]; then lio_wait_default=0; fi
  /usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/wait_for_imu.py" \
    --topic "$lio_imu_topic" --gravity-norm "${PURE3D_IMU_GRAVITY_NORM:-9.81}" \
    --timeout "${PURE3D_LIO_WAIT_TIMEOUT:-$lio_wait_default}"
fi
# 启动完成后提示查看地图：sim/real 使用不同的 RViz 入口。
lio_rviz_launch=sim/rviz.launch.py
if [[ "$PURE3D_LIO_MODE" == real ]]; then lio_rviz_launch=real/rviz_real.launch.py; fi
printf '%s\n' \
  "MID360 Faster-LIO profile: $PURE3D_LIO_MODE" \
  'This Faster-LIO version names its Livox CustomMsg branch AVIA (lidar_type=1).' \
  "Public TF: see the configured frames in $lio_frames (simulation: world -> map -> camera_init -> body)." \
  "Mounting configuration: $lio_frames" \
  "View map: ros2 launch $PURE3D_WS/Entrance/launch/$lio_rviz_launch"
# 建图与锚定节点的 PID；退出时按顺序发送 SIGINT，再 wait 回收，避免孤儿进程。
lio_anchor_pid=''
lio_mapping_pid=''
lio_cleanup() {
  trap - INT TERM
  for lio_child in "$lio_mapping_pid" "$lio_anchor_pid"; do
    if [[ -n "$lio_child" ]]; then kill -INT "$lio_child" 2>/dev/null || true; fi
  done
  for lio_child in "$lio_mapping_pid" "$lio_anchor_pid"; do
    if [[ -n "$lio_child" ]]; then wait "$lio_child" 2>/dev/null || true; fi
  done
}
trap lio_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# map_anchor 负责 world -> map 锚定；Faster-LIO 在 /faster_lio 命名空间发布原始里程计与 TF，
# 公共 body TF 由 frame 桥接转换，因此这里把 /tf 重定向到 /faster_lio/raw_tf。
/usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/map_anchor.py" --config "$lio_frames" &
lio_anchor_pid=$!
"$lio_prefix/lib/faster_lio/run_mapping_online" --ros-args \
  -r __ns:=/faster_lio --params-file "$lio_output/faster_lio.actual.yaml" \
  -r Odometry:=raw_odometry -r /tf:=/faster_lio/raw_tf "$@" &
lio_mapping_pid=$!
# 以建图进程为主进程：它退出即触发 cleanup，一并关闭锚定节点。
wait "$lio_mapping_pid"
