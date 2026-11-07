#!/usr/bin/env python3
"""Run pretraining directly, without application-imposed hardware quotas."""
import os
from pathlib import Path
import sys


def main():
    entry = Path(__file__).resolve().parent / "pretrain_perception.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    os.execve(sys.executable, [sys.executable, str(entry), *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()
