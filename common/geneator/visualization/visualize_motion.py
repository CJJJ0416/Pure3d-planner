#!/usr/bin/env python3
"""相机回放的命令入口，并兼容原脚本导入方式。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from stair_nav.visualization.motion import *  # 继续导出原脚本接口，兼容已有调用。

if __name__ == "__main__":
    main()
