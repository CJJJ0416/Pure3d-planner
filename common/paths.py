"""Pure3D 组件共享的工作区、配置与原生代码目录。"""
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
CONFIG_DIR = ROOT / "Entrance/config"
NATIVE_DIR = PACKAGE_DIR / "native"
