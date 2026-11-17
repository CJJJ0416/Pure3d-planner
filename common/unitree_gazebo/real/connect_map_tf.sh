#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export PURE3D_LIO_FRAMES_CONFIG="${PURE3D_LIO_FRAMES_CONFIG:-$PURE3D_UNITREE_ROOT/real/config/frames.yaml}"
export PURE3D_LIO_OUTPUT="${PURE3D_LIO_OUTPUT:-$PURE3D_WS/outputs/unitree_real/lio}"
exec /usr/bin/python3 "$PURE3D_UNITREE_ROOT/scripts/map_anchor.py" "$@"
