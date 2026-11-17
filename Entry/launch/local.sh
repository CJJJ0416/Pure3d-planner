#!/usr/bin/env bash
set -eo pipefail
entry_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)"
source /opt/ros/humble/setup.bash
export PYTHONPATH="$entry_root:${PYTHONPATH:-}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$entry_root/outputs/entry_local/ros_log}"
mkdir -p "$ROS_LOG_DIR"
cd "$entry_root"
exec /usr/bin/python3 -m Entry.runtime.local "$@"
