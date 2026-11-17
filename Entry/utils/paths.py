"""Stable locations shared by classified Entry implementations."""
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "configs"
NATIVE_DIR = PACKAGE_DIR / "native"
