# 逐点恢复与独立可视化分支

2026-11-04。此分支只做推理和可视化，不改变PTv3训练入口、地图解码流程或checkpoint参数。

## 输出与恢复

```text
N个有效LiDAR点 [N,3]
  -> 0.1m体素采样，保留inverse [N]
  -> 官方完整PTv3编码器与自身解码器
  -> 体素特征F [M,64]
       ├─ 原地图头 -> cost/高度 [2,6,64,64]
       └─ 独立可视化 -> F[inverse] [N,64]，配回原始LiDAR XYZ
```

```python
point, inverse = prepare(points, feat=("coord",), grid_size=0.1,
                         center=False, pick="first")
tokens = model.backbone(point)
features_lidar = tokens.feat[inverse]
# features_lidar的行顺序与points相同，显示坐标仍使用原始points。
# 原地图头继续使用tokens.coord与tokens.feat。
```

同一体素内的原始点共享64维特征。恢复的是每个原始点的特征对应关系，不重新生成点位置。没有回波的射线不构成可恢复的点。

## 已完成检查

- 新楼梯场景40帧共11,234个有效返回全部恢复。第16帧166×64恢复为307×64，坐标和原始射线顺序不变。
- 原坡道场景130帧共37,164个有效返回全部恢复。第52帧184×64恢复为271×64。
- 每帧核验inverse对应的体素与原始点体素一致，恢复前后特征按inverse索引逐元素相同。原地图cost/高度与既有推理缓存的最大差均为0。
- 原checkpoint SHA-256为`416d597b68962be23511312efb34276b0831feb520e7535deae21f95ce8d72c2`，未变。46份官方PTv3文件指纹通过。没有新增可训练参数，没有训练或微调。

导出入口为`scripts/perception/export_point_restoration.py`。每帧NPZ独立保存原始点/世界坐标、有效射线索引、inverse、体素特征、恢复特征以及共享PCA颜色。PCA基于同一采集序列的体素特征拟合一次，序列回放复用同一颜色基准。PCA颜色用于看特征分布，不代表语义类别或cost。

## Isaac原生可视化

`scripts/visualization/view_point_restoration_isaac.py`复用采集时的USD网格。四个模式为原始点、体素特征、恢复点特征、在LiDAR点位采样的预测地图cost。提供40帧选择、回放、全景及楼梯近景。

第16帧四模式、两视角共8张原生截图已生成，界面回调、原网格顶点/面索引、显示点数通过核验。最后观测的窗口状态为closed，不表示窗口始终运行。

原生对照图为`outputs/terrain_generator_transfer_20261104/point_restoration/frame_00016_restoration_comparison.png`。其源截图与报告位于该目录的`native_v2/`。

cost视图在原始LiDAR点位查现有预测地图，不是新增逐点预测头。第16帧307个点均保留，289个在地图范围内着色，18个范围外显示灰色。cost采样未使用GT掩码。

## 第52帧RGB、GT与完整点云预测

用户指定的原相机、RGB及GT对照已按当前配置重新输出：

`outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt_prediction_point_restored.png`

上排为原RGB、58个GT cost监督格及63个GT高度监督格。下排为恢复后的271个LiDAR点的PCA特征、预测cost、预测地面高度。267个点在固定地图坐标及高度带内，4个范围外点用灰色保留。所有271个点均投影在原相机视野内，不用GT掩码或RGB深度遮挡掩码筛选。

预测颜色通过LiDAR端点所在的固定网格索引读取，点的XYZ保持真实LiDAR返回位置。图中的高度颜色表示预测地面高度，不表示该点被移动到了预测地面表面。为同时容纳全部返回点所处高度带，GT/预测高度共享色标扩展为[-1.5,0]m。GT仍显示原参考高度带[-0.5,0)m。

原GT格指标仍为cost MAE=7.9042/60、高度MAE=0.06149m。这些指标只对应原来的58/63个监督格，不是新增267个着色点的误差。原图未覆盖，新图与同名JSON保存完整计数、相机和数据指纹。

## 重现入口

在本项目目录，使用独立GPU环境导出。输出目录必须是新目录：

```bash
/home/cjh/glocal2-nav/.venv_ptv3_gpu/bin/python scripts/perception/export_point_restoration.py \
  --maps outputs/plan_a_maps_20261104 --output outputs/restoration_new --device cuda:0

/home/cjh/miniconda3/envs/env_isaaclab/bin/python scripts/visualization/render_rgb_point_restoration_comparison.py \
  --restoration outputs/restoration_new \
  --reference outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt.png \
  --output outputs/restoration_new/frame52_rgb_comparison.png
```

仅可视化任务已完成。下一步仍是补足多场景训练数据与独立验证，而不是改变恢复映射来增加预测能力。

## GT也映射到所有LiDAR点

按用户后续要求，新增`--gt-on-lidar`显示方式，新图为`outputs/plan_a_maps_20261104/plots/frame_00052_rgb_gt_prediction_all_points.png`。该图上下排均保留同一批271个原始点及相同相机投影位置。

GT直接从固定的全局VST查表。每个原始点先转到世界系，使用`round((world_xy-center)/resolution)+dimension//2`定位原生单元，再选择地面高度与该点处于同一重力高度带的图层。相同高度的重复图层沿用最低cost的确定性选择。不同高度的候选若相差超过1mm则标为歧义，不强行合并。无可用值或歧义点显示灰色，不用颜色值0代替。

第52帧全部271点均获得无歧义GT，267点可从预测的局部地图取值，4点超出局部地图范围以灰色保留。GT与预测共享原始XYZ和色标。图中GT地面高度是该世界XY/高度带对应的VST地面值，不要求等于LiDAR回波自身高度。225个回波距该地面不超过4cm，其余点也显示地面场的查表值。

这是一种全局教师场的可视化投影，没有修改局部GT标签文件、4cm高度匹配、连续性或观测覆盖训练掩码，没有新增训练或更新权重。

新图按267个共同可查点重新计算端点查表诊断，cost MAE=10.1714/60，高度MAE=0.08216m。该指标的点位和统计范围与旧58/63格指标不同。逐点GT值、源图层、原生网格索引、预测值和可用标记保存为同名前缀的`_point_values.npz`，并逐点验证原生VST数值一致。

重现时在原绘图命令中增加`--gt-on-lidar`，并使用新输出文件名。

## 最后一帧的平台边界

frame129是当前130帧坡道序列的最后一帧。200×64体素特征恢复为237×64，GT在全部237点上均可查且无高度歧义。预测局部地图覆盖202个点，其余35点范围外用灰色保留。

- 平台边界近景：`outputs/plan_a_maps_20261104/plots/frame_00129_rgb_gt_prediction_all_points.png`。近景相机显示210个点，其中202点有预测、8点范围外灰色。其余27点仅因相机近景裁切未显示。
- 完整点云全景：`outputs/plan_a_maps_20261104/plots/frame_00129_rgb_gt_prediction_all_points_full_view.png`。全部237个有效点都位于相机视野内，GT全部着色，预测202点着色、35点灰色。

完整视图由新增`render_plan_a_rgb_gt.py --view all-points`自动根据实际点云与参考GT范围拟合相机。两视图的GT/预测使用各自相同的原始点坐标和色标，没有调整场景或模型权重。

202个共同点的端点查表诊断为cost MAE=9.5231/60、高度MAE=2.3968cm。GT cost≥40的15个共同点，GT平均52.5083，预测平均12.2429。图中平台高代价边缘被明显低估，高度更接近GT。此处仅记录当前checkpoint与当前帧的查表诊断。
