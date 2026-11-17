#!/usr/bin/env bash
# Shared runtime; start through a sim/ or real/ entry point.
set -eo pipefail
: "${PURE3D_UNITREE_ROOT:?Use the sim or real Faster-LIO entry point}"
: "${PURE3D_LIO_MODE:?Use the sim or real Faster-LIO entry point}"
lio_bag=''
if [[ "${1:-}" == '--bag' ]]; then
  if [[ $# -lt 2 ]]; then echo 'Usage: run_faster_lio.sh --bag <rosbag2 directory>' >&2; exit 1; fi
  lio_bag="$(realpath "$2")"
  shift 2
  if [[ ! -f "$lio_bag/metadata.yaml" ]]; then echo "Bag metadata missing: $lio_bag" >&2; exit 1; fi
  if [[ "$PURE3D_LIO_MODE" != sim ]]; then echo '--bag is configured for the Pure3D simulation capture contract' >&2; exit 1; fi
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-184}"
  if [[ "$ROS_DOMAIN_ID" != 184 ]]; then echo 'Use ROS_DOMAIN_ID=184 for this offline bag workflow' >&2; exit 1; fi
  export PURE3D_LIO_OUTPUT="${PURE3D_LIO_BAG_OUTPUT:-$(dirname -- "$lio_bag")/offline/lio}"
fi
lio_ws="${PURE3D_LIO_WS:-$HOME/livox_ws}"
lio_prefix="$lio_ws/install/faster_lio"
if [[ ! -x "$lio_prefix/lib/faster_lio/run_mapping_online" ]]; then
  echo "Faster-LIO is not built: $lio_prefix. Set PURE3D_LIO_WS to its workspace." >&2
  exit 1
fi
source "$lio_prefix/share/faster_lio/local_setup.bash"
lio_libs="$lio_ws/third_party/sysroot/usr/lib/x86_64-linux-gnu"
if [[ -d "$lio_libs" ]]; then
  export LD_LIBRARY_PATH="$lio_libs:${LD_LIBRARY_PATH:-}"
fi
lio_output="$(realpath -m "$PURE3D_LIO_OUTPUT")"
export PURE3D_LIO_OUTPUT="$lio_output"
lio_frames="$(realpath "$PURE3D_LIO_FRAMES_CONFIG")"
lio_config="$(realpath "$PURE3D_LIO_CONFIG")"
mkdir -p "$lio_output"
cd "$lio_output"
/usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/frame_config.py" \
  --config "$lio_frames" --lio-config "$lio_config" --profile "$PURE3D_LIO_MODE" \
  --output "$lio_output/faster_lio.actual.yaml"
if [[ "${1:-}" == '--check-config' ]]; then
  printf 'Profile: %s\nFrames: %s\nParameters: %s\n' "$PURE3D_LIO_MODE" "$lio_frames" "$lio_output/faster_lio.actual.yaml"
  exit 0
fi
if [[ -n "$lio_bag" ]]; then
  lio_bag_args=(--bag "$lio_bag" --report "$lio_output/bag_check.json")
  if [[ -f "$(dirname -- "$lio_bag")/mission.json" ]]; then
    lio_bag_args+=(--mission "$(dirname -- "$lio_bag")/mission.json")
  fi
  PYTHONPATH="$PURE3D_WS:${PYTHONPATH:-}" /usr/bin/python3 -m Entry.mission.bag_mission validate "${lio_bag_args[@]}"
  printf 'Bag mode validated; waiting for a separate bag player in ROS_DOMAIN_ID=%s.\n' "$ROS_DOMAIN_ID"
else
  lio_imu_topic="$(/usr/bin/python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["/**"]["ros__parameters"]["common"]["imu_topic"])' "$lio_output/faster_lio.actual.yaml")"
  lio_wait_default=120
  if [[ "$PURE3D_LIO_MODE" == sim ]]; then lio_wait_default=0; fi
  /usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/wait_for_imu.py" \
    --topic "$lio_imu_topic" --gravity-norm "${PURE3D_IMU_GRAVITY_NORM:-9.81}" \
    --timeout "${PURE3D_LIO_WAIT_TIMEOUT:-$lio_wait_default}"
fi
printf '%s\n' \
  "MID360 Faster-LIO profile: $PURE3D_LIO_MODE" \
  'This Faster-LIO version names its Livox CustomMsg branch AVIA (lidar_type=1).' \
  "Public TF: see the configured frames in $lio_frames (simulation: world -> map -> camera_init -> body)." \
  "Mounting configuration: $lio_frames" \
  "View map: $PURE3D_UNITREE_ROOT/$PURE3D_LIO_MODE/view_faster_lio.sh"
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
/usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/map_anchor.py" --config "$lio_frames" &
lio_anchor_pid=$!
"$lio_prefix/lib/faster_lio/run_mapping_online" --ros-args \
  -r __ns:=/faster_lio --params-file "$lio_output/faster_lio.actual.yaml" \
  -r Odometry:=raw_odometry -r /tf:=/faster_lio/raw_tf "$@" &
lio_mapping_pid=$!
wait "$lio_mapping_pid"
