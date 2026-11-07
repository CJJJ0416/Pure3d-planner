# Headless 服务器入口

本次从零训练的服务器配置、重建场景、采集、训练、分段测试和输出命令统一维护在 [SERVER_TRAINING.md](SERVER_TRAINING.md)。

服务器使用 `/opt/conda/envs/rl_env_glocal2/bin/python`。纯 CUDA Warp 采集、PyTorch 训练和 matplotlib Agg 绘图共用现有环境。

| 工作 | 入口 |
| --- | --- |
| 纯 Warp 多帧采集 | `scripts/data/collect_mid360_warp.py` |
| 原生 VST 标签 | `scripts/data/pair_mid360_lidar_reference.py` |
| 端到端调度 | `isaac_training/scripts/server/run_playground_campaign.py`，相对仓库根目录 |
| GT/预测/误差图 | `scripts/visualization/render_plan_a_comparison.py` |
| 训练曲线 | `scripts/visualization/plot_training_history.py` |

服务器输出 PNG 和指标文件，Isaac 原生查看器用于本地交互展示。
