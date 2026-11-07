# Common

集中管理 VST 与场景生成工具。

| 目录 | 用途 |
|---|---|
| `vst_global_planner/` | VST 内核、原生规划器和已有组合场景工具 |
| `geneator/` | 已迁入的场景生成、修改、回放和环境脚本 |

预训练 PTv3、点云 I/O、坐标转换、分块和编码器封装放在 `../utils/`。

场景脚本从工程根目录通过 `common/geneator/...` 调用，VST 自带入口通过 `common/vst_global_planner/scripts/...` 调用。具体运行参数在后续数据契约和运行说明中确定。

