#!/usr/bin/env bash
# 整体方案：按就绪状态依次打开 Gazebo、键盘、LIO、cost 四个独立终端。
# Usage（工作区根目录）：
#   ./Entrance/shell/total_terminal.sh --scene building
#   ./Entrance/shell/total_terminal.sh --scene terrain-prepared --model tiny
#   ./Entrance/shell/total_terminal.sh --dry-run
#   ./Entrance/shell/total_terminal.sh --help
# 默认场景：Entrance/config/startup.yaml；场景参数：Entrance/config/scenes/<场景>.yaml。
# cost 配置：Entrance/config/local_cost.yaml；单项方案使用 Entrance/launch/ 下的文件。
# 任一命令失败即停止；pipefail 保证管道中任一环节失败也能被发现。
set -eo pipefail
# terminal_root 为本文件所在 shell 目录上溯两级，即 Pure3D_ws 工作区根目录。
terminal_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# 帮助与预览不加载 ROS overlay、不清理现有进程。
for terminal_arg in "$@"; do
  case "$terminal_arg" in
    --help|-h|--dry-run)
      # 这三种模式只需 PYTHONPATH，直接进入 Python 实现，避免 source 环境造成副作用。
      export PYTHONPATH="$terminal_root:${PYTHONPATH:-}"
      exec /usr/bin/python3 "$terminal_root/Entrance/shell/_internal/total_terminal.py" "$@" ;;
  esac
done
# 正式启动前加载本工作区仿真 overlay；环境库会清理其它工程的路径残留。
source "$terminal_root/Entrance/shell/_internal/sim_env.bash"
# 四个终端必须共享同一 ROS 域并限制在本机，否则 Gazebo/LIO/cost 会互相看不到。
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PYTHONPATH="$terminal_root:${PYTHONPATH:-}"
cd "$terminal_root"
# exec 让 Python 接管当前进程，Ctrl+C/退出码能直接传递给终端托管逻辑。
exec /usr/bin/python3 "$terminal_root/Entrance/shell/_internal/total_terminal.py" "$@"
