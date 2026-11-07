#!/usr/bin/env bash
# 第一个参数为输出压缩包路径；默认位于 exports/，内容严格来自当前 HEAD。
# 工作区未提交修改不会进入包中，需要先提交后再运行此脚本。
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST="${1:-$ROOT_DIR/exports/glocal2-nav-source.tar.gz}"
mkdir -p "$(dirname -- "$DEST")"
DEST="$(realpath -m -- "$DEST")"
if ! git -C "$ROOT_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Create a local Git commit before packaging the source release." >&2
  exit 1
fi
# 仅打包已提交的源码快照，忽略本地环境、测试产物和未提交文件。
git -C "$ROOT_DIR" archive --format=tar.gz --output="$DEST" HEAD
printf 'Source archive: %s\n' "$DEST"
