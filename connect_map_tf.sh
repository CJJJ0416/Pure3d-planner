#!/usr/bin/env bash
set -eo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/common/unitree_gazebo/sim/connect_map_tf.sh" "$@"
