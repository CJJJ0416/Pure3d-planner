#!/usr/bin/env bash
# Open independent run, keyboard, Faster-LIO and cost terminals in order.
set -eo pipefail
terminal_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$terminal_root/common/unitree_gazebo/sim/env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PYTHONPATH="$terminal_root:${PYTHONPATH:-}"
cd "$terminal_root"
exec /usr/bin/python3 -m common.unitree_gazebo.sim.total_terminal "$@"
