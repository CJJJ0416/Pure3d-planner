#!/usr/bin/env bash
# 在目标机器重新编译，避免跨机器复制带 CPU 指令集依赖的动态库。
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT_DIR/common/vst_global_planner/native"
BUILD_DIR="$ROOT_DIR/build/native"
DEP="$BUILD_DIR/dependencies"
JOBS="${BUILD_JOBS:-2}"
# 与 stair_nav.sh 共用解释器覆盖项，支持 Conda；默认仍使用原有 .venv。
RUNTIME_PYTHON="${STAIR_NAV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
# 构建顺序为 GTSAM、OSQP、原生规划扩展，全部放在忽略的 build/native 下。
# BUILD_JOBS 可覆盖并行数；GTSAM 使用本机指令集优化，因此产物不宜跨机器复用。
cmake -S "$SRC/3rdparty/gtsam-4.1.1" -B "$BUILD_DIR/gtsam" \
  -DCMAKE_INSTALL_PREFIX="$DEP" -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN'
cmake --build "$BUILD_DIR/gtsam" --parallel "$JOBS"
cmake --install "$BUILD_DIR/gtsam"
# OSQP 需要位置无关代码，供后续共享库链接；此入口不构建上游测试或示例。
cmake -S "$SRC/3rdparty/osqp" -B "$BUILD_DIR/osqp" \
  -DCMAKE_INSTALL_PREFIX="$DEP" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUNITTESTS=OFF -DENABLE_MKL_PARDISO=OFF
cmake --build "$BUILD_DIR/osqp" --parallel "$JOBS"
cmake --install "$BUILD_DIR/osqp"
# 最后把依赖前缀和项目 Python 传给原生模块；RPATH 相对动态库所在位置解析。
cmake -S "$SRC" -B "$BUILD_DIR/core" -DCMAKE_PREFIX_PATH="$DEP" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$RUNTIME_PYTHON" '-DCMAKE_INSTALL_RPATH=$ORIGIN'
cmake --build "$BUILD_DIR/core" --parallel "$JOBS"
cmake --install "$BUILD_DIR/core"
# 解析源符号链接后原子替换目标项，避免沿旧链接误写外部文件。
"$RUNTIME_PYTHON" - "$ROOT_DIR" <<'PYCODE'
import hashlib, json, os, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
dest = root / '.local/native'
dest.mkdir(parents=True, exist_ok=True)
files = list((root / 'build/native/install/lib/global_planner/native').glob('*.so*'))
for pattern in ('libgtsam.so*', 'libmetis-gtsam.so*'):
    files.extend((root / 'build/native/dependencies/lib').glob(pattern))
for source in files:
    temporary = dest / (source.name + '.new')
    shutil.copy2(source.resolve(), temporary)
    os.replace(temporary, dest / source.name)
provenance = json.loads((root / 'common/vst_global_planner/PROVENANCE.json').read_text())
info = {'source_commit': provenance['commit'],
        'native_sources': {k:v for k,v in provenance['files'].items() if k.startswith('native/')},
        'binary_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.so*')}}
(dest / 'BUILD_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
# 构建记录同时绑定上游源码清单与实际二进制指纹，便于核对部署来源。
PYCODE
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$RUNTIME_PYTHON" -c 'from stair_nav.planning.vst import load_native; load_native(); print("Native planner ready")'
