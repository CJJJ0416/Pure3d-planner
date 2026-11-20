# Pure3D 核心组件与共用工具

| 组件 | 职责 |
|---|---|
| [Filter](Filter/README.md) | 核心点云保留评分、训练、推理与显式过滤 |
| [Traversability](Traversability/README.md) | 核心条件化可通行代价、教师数据、训练与推理 |
| `PointTransformerV3/` | 固定版本的 PTv3 编码器源码 |
| 其余工具模块 | 核心分块、几何、点云处理与编码器封装 |

Usage（工作区根目录，使用 Pure3D Python 环境）：

```bash
python -m utils.Filter.inference.predict --help
python -m utils.Traversability.training.train --help
python -m utils.Traversability.inference.predict --help
```

数据、权重、训练记录和独立输出继续存放在工作区的 `datasets/`、`weights/`、`runs/` 与 `outputs/`。


保存 PTv3 与任务间共用的点云处理工具。

- `PointTransformerV3/`：已有预训练编码器源码与模型加载接口。
- `pointcloud.py`：工作空间配置、体素降采样、空间分块和旧 VST 查表工具。
- 坐标转换和编码器封装按需增加，各维护一份。

通用 PCD 读写和 ROS 点云转换统一位于 `../common/pointcloud_io/`。

VST 与场景生成工具统一位于 `../common/`。具体任务头和训练循环分别放在 `../Filter/`、`../Reconstruction/`、`../Traversability/`。Filter 与 Traversability 使用相同的分类目录，共用编码器通过坐标、inverse 和点索引传递特征。

