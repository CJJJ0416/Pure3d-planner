#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export PURE3D_LIO_FRAMES_CONFIG="${PURE3D_LIO_FRAMES_CONFIG:-$PURE3D_UNITREE_ROOT/real/config/frames.yaml}"
export PURE3D_LIO_CONFIG="${PURE3D_REAL_LIO_CONFIG:-$PURE3D_UNITREE_ROOT/real/config/faster_lio.yaml}"
export PURE3D_LIO_OUTPUT="${PURE3D_LIO_OUTPUT:-$PURE3D_WS/outputs/unitree_real/lio}"
exec "$PURE3D_UNITREE_ROOT/scripts/run_faster_lio.sh" "$@"
