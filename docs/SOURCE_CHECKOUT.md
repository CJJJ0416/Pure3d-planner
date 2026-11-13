# 源码检出与本地产物

Pure3D 主仓库管理网络框架、数据生成、教师监督、点云 I/O、规划与可视化源码，以及运行必需的 MID-360 方向表和模型资产。datasets、runs、weights、缓存、编译产物和个人编辑器配置保留本地，不随源码提交。

## PTv3 / Utonia Tiny

`utils/PointTransformerV3` 保持现有的固定 Git 子模块，来源由根目录 .gitmodules 声明。检出主仓库后执行：

```bash
git submodule update --init --recursive utils/PointTransformerV3
```

模型环境使用 environment-ptv3.yml，其中可编辑安装该子模块。Filter 当前参考为 relative 模式，无六维 KNN／体素统计，权重身份由 Filter/configs/baseline.json 记录。checkpoint 必须单独准备，baseline.json 中路径为工作区相对路径，不代表权重随 Git 分发。

## Terrain-generator

common/geneator/terrain-generator 的运行源码按快照纳入主仓库，包含本机已有的 Python 3.11 dataclass default_factory 修订。来源、上游版本和修订说明位于 common/geneator/TERRAIN_GENERATOR_PROVENANCE.json。保留上游 LICENSE、源码和示例，不收录展示 GIF、测试目录或生成缓存。

本地该目录已有的 .git 元数据保持原样，不作为主仓库 Gitlink 提交。主仓库索引中的实际文件是可复现训练数据生成所需的源码快照。

## 运行环境与本地数据

- 模型训练和推理使用 Pure3D Python 环境。几何依赖见 Reconstruction/requirements.txt 和 common/geneator/requirements-wfc.txt。
- 逐点扫描使用 common/scanning/requirements.txt 中的 Warp。Faster-LIO 从 common/scanning/config.json 指定的 ROS 2 underlay 加载，不随本仓库分发。
- Isaac Sim 原生查看使用独立的 env_isaaclab 环境。ROS/RViz 使用加载 Humble 后的系统 Python。
- common/Mid360_imu_sim 提供方向表和模型资产。现有插件为瞬时扫描，新滚动扫描实现位于 common/scanning，不将瞬时 offset_time=0 的消息当作真实滚动测量。
- 本地 seen2scene 是独立参考 checkout，当前 Filter、Reconstruction 和 Traversability 不导入它。其权重、结果和本地改动保留原样，本次源码提交不收录该目录。

本地实验档案可保留 runs 下的路径作为证据定位。检出源码不等于已经拥有这些数据或权重，也不会自动启动训练、采集或下载模型。
