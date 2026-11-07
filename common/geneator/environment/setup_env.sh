#!/usr/bin/env bash
# 用法：setup_env.sh [--full]。默认安装基础依赖，--full 加装 VST 所需依赖。
# PYTHON_BIN 决定项目虚拟环境解释器；PIP_PYTHON 只负责提供可用的 pip 安装工具。
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# 虚拟环境复用系统可见依赖，由具备 pip 的解释器向该环境安装项目依赖。
"$PYTHON_BIN" -m venv --without-pip --system-site-packages "$ROOT_DIR/.venv"
PIP_PYTHON="${PIP_PYTHON:-$PYTHON_BIN}"
if ! "$PIP_PYTHON" -m pip --version >/dev/null 2>&1; then
  if [[ -x /home/cjh/miniconda3/bin/python ]]; then PIP_PYTHON=/home/cjh/miniconda3/bin/python;
  else echo 'Set PIP_PYTHON to a Python with pip >= 22.3, or install python3-pip.' >&2; exit 1; fi
fi
if [[ "${1:-}" == --full ]]; then
  # 完整依赖模式不继续引用旧的本机依赖路径，避免混入来源不一致的运行库。
  "$ROOT_DIR/.venv/bin/python" - <<'PYCODE'
import pathlib, sysconfig
link=pathlib.Path(sysconfig.get_path('purelib'))/'vst_local_dependencies.pth'
if link.exists(): link.unlink()
PYCODE
fi
REQ="$ROOT_DIR/requirements.txt"
if [[ "${1:-}" == --full ]]; then REQ="$ROOT_DIR/requirements-vst.txt"; fi
"$PIP_PYTHON" -m pip --python "$ROOT_DIR/.venv/bin/python" install -r "$REQ"
"$ROOT_DIR/.venv/bin/python" - "$ROOT_DIR" "${1:-}" <<'PYCODE'
# 通过 .pth 引用源码，使后续修改直接生效，无需反复安装包。
import pathlib, sys, sysconfig
root=pathlib.Path(sys.argv[1]); site=pathlib.Path(sysconfig.get_path('purelib'))
(site/'stair_nav_source.pth').write_text(str(root/'src')+'\n')
link=site/'vst_local_dependencies.pth'
if sys.argv[2]=='--full' and link.exists(): link.unlink()
PYCODE
printf 'Environment ready: %s\n' "$ROOT_DIR/.venv"
