#!/usr/bin/env bash
# Independent local cost and timing only. Sensors, LIO and control stay separate.
set -eo pipefail
cost_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$cost_root/common/unitree_gazebo/sim/env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PYTHONPATH="$cost_root:${PYTHONPATH:-}"
cd "$cost_root"
exec /usr/bin/python3 -m Entry.runtime.cost_session "$@"
