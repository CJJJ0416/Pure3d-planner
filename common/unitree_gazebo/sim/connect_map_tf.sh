#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export PURE3D_LIO_FRAMES_CONFIG="${PURE3D_LIO_FRAMES_CONFIG:-$PURE3D_UNITREE_ROOT/sim/config/frames.yaml}"
exec /usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/map_anchor.py" "$@"
