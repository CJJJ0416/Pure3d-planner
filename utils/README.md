# Utils

保存 PTv3 与任务间共用的点云处理工具。

- `PointTransformerV3/`：已有预训练编码器源码与模型加载接口。
- `pointcloud.py`：工作空间配置、体素降采样、空间分块和旧 VST 查表工具。
- 坐标转换和编码器封装按需增加，各维护一份。

通用 PCD 读写和 ROS 点云转换统一位于 `../common/pointcloud_io/`。

VST 与场景生成工具统一位于 `../common/`。具体任务头和训练循环分别放在 `../Filter/`、`../Reconstruction/`、`../Traversability/`。Filter 与 Traversability 使用相同的分类目录，共用编码器通过坐标、inverse 和点索引传递特征。

