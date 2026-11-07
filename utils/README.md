# Utils

保存 PTv3 与任务间共用的点云处理工具。

- `PointTransformerV3/`：已有预训练编码器源码与模型加载接口。
- 点云 I/O、坐标转换、空间分块和编码器封装，在需要时直接增加对应 Python 文件，各维护一份。

VST 与场景生成工具统一位于 `../common/`。具体任务头和训练循环分别放在 `../Traversability/`、`../Reconstruction/`。

