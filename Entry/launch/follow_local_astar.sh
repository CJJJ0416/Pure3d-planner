#!/usr/bin/env bash
set -eo pipefail
follow_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "$follow_root/common/unitree_gazebo/sim/env.sh"
export PYTHONPATH="$follow_root:${PYTHONPATH:-}"
cd "$follow_root"
exec /usr/bin/python3 -m Entry.planning.local_astar_follow "$@"
