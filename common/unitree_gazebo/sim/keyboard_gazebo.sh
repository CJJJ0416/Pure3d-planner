#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec /opt/ros/humble/bin/ros2 run keyboard_input keyboard_input --ros-args -p publish_cmd_vel:=true "$@"
