# Traversability

建筑 PCD + 机器人能力参数 → 对应 VST trav_cost 的监督学习。

采用扁平文件结构，后续实现直接放在本目录：

| 文件 | 用途 |
|---|---|
| `dataset.py` | 点云、条件参数与标签读取 |
| `model.py` | 编码器和条件化 cost 头 |
| `prepare.py` | 数据准备 |
| `train.py` | 训练、损失和断点保存 |
| `predict.py` | 分块推理、融合与导出 |
| `visualize.py` | GT、预测与误差对比 |
| `robots.json` | 机器人参数配置 |
| `dataset.json` | 数据路径与划分 |
| `train_local.json`、`train_server.json` | 本地与服务器训练设置 |

目前只确定这些文件的职责，尚未添加正式实现。实际数据放 `../datasets/<building_id>/`，结果放 `../runs/cost_<experiment_name>/`。

VST 与场景生成工具从 `../common/` 调用。PTv3 和共用点云处理代码从 `../utils/` 调用，数据缓存统一放 `../datasets/cache/`。

此前草稿保存在 `../runs/conditioned_cost_local_20261107/drafts/Traversability/`，尚未验收。后续先确定数据契约和模型方案。

完整约定见 [文件架构](../docs/WORKSPACE_LAYOUT.md)。
