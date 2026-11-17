#!/usr/bin/env bash
set -eo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/common/unitree_gazebo/sim/keyboard_gazebo.sh" "$@"
