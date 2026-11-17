#!/usr/bin/env bash
set -eo pipefail
navigation_root="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "$navigation_root/common/unitree_gazebo/sim/env.sh"
export PYTHONPATH="$navigation_root:${PYTHONPATH:-}"
cd "$navigation_root"
exec /usr/bin/python3 -m Entry.planning.navigation "$@"
