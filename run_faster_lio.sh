#!/usr/bin/env bash
set -eo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/common/unitree_gazebo/sim/run_faster_lio.sh" "$@"
