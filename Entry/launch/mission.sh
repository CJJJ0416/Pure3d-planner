#!/usr/bin/env bash
set -eo pipefail
mission_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "$mission_root/common/unitree_gazebo/sim/env.sh"
export PYTHONPATH="$mission_root:${PYTHONPATH:-}"
cd "$mission_root"
exec /usr/bin/python3 -m Entry.mission.bag_mission "$@"
