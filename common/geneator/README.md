# WFC 粗糙多层环境

Filter 数据复用同一 WFC 模板，在 `filter_wfc.py` 中加入带门洞与顶板的走廊，并通过 `filter_components.py` 保留面片的语义、实例和目标角色。D0、D1 入口、连续拖影与 PCD 产物见 [Filter 数据准备](../../Filter/docs/DATA_PREPARATION.md)。

当前入口为 generate_wfc_multifloor.py，默认配置为 wfc_capability_dr.json。旧的固定三层样例配置 wfc_multifloor.json 保留。

## 生成方式

- 直接调用 terrain-generator 原生 WFCSolver、MeshTile、Perlin、坡道和台阶模块。
- 每个楼层按相邻模块的高度边界约束进行二维 WFC 求解。
- 再按配置叠加楼层，加入中庭、跨空桥、楼梯井、平整入口和出口平台。
- 默认启用域随机化，采样 3–4 层、2.6–3.8 m 层高和整体朝向。以原 16–18 行列、2.2–2.8 m 模块为每个 seed 的参考，平面占地缩小 60%，保留 40%。紧凑网格为 10–11 行列，模块保持正方形并作小幅尺寸修正，确保面积比例准确。楼梯宽度、步高、栏杆和机器人阈值均保留米制尺寸，不对完整场景做缩放。
- 每对相邻楼层生成 5 条连接：两部直梯、一部往复梯、一部旋转梯、一条斜坡。直梯含有栏杆和无栏杆两种，其余类型随 seed 和楼层交替栏杆配置。
- 每条连接从全图有上下层支撑的区域独立采样 XY，朝向为 0°、90°、180°、270°。楼梯井和上下落脚平台同步移动，不再固定在地图前后两排。
- 同步随机化步高、踏步深度、宽度、旋转圈数、栏杆高度、地面起伏和模块类型比例。
- 往复梯和旋转梯采用 2×2 单元井，直梯与斜坡采用 3×2 单元井。分别保留入口和出口的平整落脚单元，井体不重叠，落脚单元不被另一楼梯井覆盖。每个场景包含 30 个 below/near/above 能力阈值几何案例。
- 多层布局由本工作区的装配代码实现，不将二维 WFC 描述为自动完成三维全局可达性。

原生源码位于 terrain-generator/，源码快照来源与 Python 3.11 修订见
TERRAIN_GENERATOR_PROVENANCE.json。上游 commit 与本地修订来源记录在场景元数据中，不依赖嵌套 Git 元数据。
算法参考为 https://github.com/mxgmn/WaveFunctionCollapse 。

## 本机调用

在 Pure3D_ws 根目录执行：

    /home/cjh/glocal2-nav/.venv/bin/python common/geneator/generate_wfc_multifloor.py

单独生成另一个 seed：

    /home/cjh/glocal2-nav/.venv/bin/python common/geneator/generate_wfc_multifloor.py --seed 20261108

生成后自动保存 overview.png，包含全景、剖视图和各楼层视图。

### 多批次完整数据

    /home/cjh/glocal2-nav/.venv/bin/python -m Traversability.data.generate

`Traversability/configs/dataset.json` 配置三批、每批两个 seed。前两批训练，第三批分配验证与测试，按整个 seed 划分。每个 seed 顺序生成 Mesh、3 cm 完整 PCD、5 cm 降采样 PCD、五组完整 VST 教师和原生 Tomogram 代价点云，最后组织独立的建筑输入与查询核心块，不启动训练。单独替换一批可加 `--batch batch_01`。已有数据单独分块用 `python -m Traversability.data.blocks`，数据接口见 [独立分块](../../Traversability/docs/TRAVERSABILITY_BLOCKS.md)。

场景平铺在 output 下的 `scene_<seed>/`。`manifest.json` 保存面积、批次、划分、点数、原生监督文件、PCD 哈希与阶段耗时。`batch_00.json` 等为批次索引，`overview_all.png` 为实际场景合集，`layout_samples.png` 展示各实际场景的中间层平面布局。每个场景仍保留完整总览与细节图。五组教师均保留完整参数，距离曲线 p=2。

`landings.report.json` 记录实际 Mesh 上每个落脚平台的五个向下探针命中及高度误差。完整/降采样点云报告记录独立表面探针和栏杆覆盖。这些结果用于检查生成几何和采样质量，不代表已验证机器人导航或模型泛化。

## 产物

默认目录为 datasets/wfc_capability_dr_v2/scene_20261109/。v1 场景、点云和已有教师数据保留在原目录。

- terrain.obj：完整三层几何。
- floor_0.obj、floor_1.obj、floor_2.obj：独立楼层。
- F0_F1_straight_0.obj 等：每条独立连接，名称记录楼层和类型。
- floor_*_wave.npy、floor_*_mask.npy：WFC 解及楼板开口布局。
- scene.json、config.resolved.json：参数、seed、源码版本和坐标约定。
- overview.png：实际 Mesh 的 headless 总览。
- layout_overview.png：每层的楼梯井位置、类型、朝向和层间连接关系。
- connections_overview.png：每条直梯、往复梯、旋转梯、斜坡与栏杆近景。
- capability_overview.png：净空、坡度、步高、支撑和安全边距样本近景。
- capability_features.json：参考 VST 参数与各个几何样本的实际尺寸。
- wfc.log：原生求解日志。

截图中的剖视图隐藏前半部分上层几何，完整 terrain.obj 保留全部结构。场景生成和点云采样分开执行，VST 监督与训练仍属于后续步骤。

## 第二步：Mesh 表面转成点云

在工作区根目录执行：

    /home/cjh/glocal2-nav/.venv/bin/python common/geneator/mesh_to_pcd.py --scene datasets/wfc_capability_dr_v2/scene_20261109 --spacing 0.03

采样覆盖每个三角面，包括楼板上下表面、侧面、楼梯、栏杆和坑底。沿三角面的高度布置采样行，每行再等距取点，两方向间隔均不大于 spacing。保留三角面边界和顶点，防止小面积的栏杆面被纯随机采样漏掉。按微米小数精度去重，不做体素中心替换。

采样直接沿用 OBJ 中的世界坐标，单位为米，map 坐标系，Z 向上，保留负高度。terrain.obj 为拼接实体，采样也包括其接触面和内部面，不额外执行布尔并集。

同一场景目录新增：

- terrain.pcd：binary PCD，只包含 float32 XYZ，保留全部采样点。
- pointcloud.report.json：点数、边界、耗时、最近邻距离、独立表面探针到 PCD 的距离，以及每条连接和栏杆的覆盖统计。
- pointcloud_overview.png：源 Mesh、全场景 PCD、剖视与有栏杆楼梯的近景。全景只为绘图抽点，近景保留裁剪范围内的所有点，PCD 文件本身不抽点。

这一步不需要路线或 LiDAR，不生成 VST 标签，也不启动模型训练。后续场景用相同入口指定 --scene 即可。

### 点云降采样

对完整 terrain.pcd 做 5 cm 体素降采样，每个非空体素输出其点的质心：

    /home/cjh/glocal2-nav/.venv/bin/python common/geneator/downsample_pcd.py --scene datasets/wfc_capability_dr_v2/scene_20261109 --voxel 0.05

结果另存 terrain_downsampled.pcd，原始 terrain.pcd 保留。坐标系和单位不变。另输出与输出 PCD 同名的 .report.json 和 .png，记录点数减少比例、几何覆盖与楼梯、栏杆近景对比。质心会改变体素内的采样坐标，该文件为独立的降采样数据。首次 5 cm 产物的报告与图片另以 downsampling.report.json、pointcloud_downsampled.png 保存。

进一步生成 10 cm 版本，并与已有 5 cm PCD 对比：

    /home/cjh/glocal2-nav/.venv/bin/python common/geneator/downsample_pcd.py --voxel 0.10 --output terrain_downsampled_10cm.pcd --compare terrain_downsampled.pcd

实际降采样始终从完整 terrain.pcd 计算，避免对已降采样质心重复求平均。--compare 只选择对比图第一行的数据，不参与降采样。五组 VST 教师 pickle 共用各场景的 5 cm terrain_downsampled.pcd，点数见 manifest.json。来源文件名和哈希记录在教师元数据中。教师入口为 `python -m Traversability.data.teacher`，默认指向 v2，重复运行覆盖同名教师文件。

## 参数与场景覆盖

参考参数写在 wfc_capability_dr.json 的 reference_vst 中。每种情况按 below、near、above 三个区间采样，保留阈值附近与超过阈值的几何。

| VST 参数 | 对应生成内容 |
|---|---|
| interval_min、interval_free | 低顶、门洞、横梁下净空，参考高度的 0.55–1.5 倍 |
| slope_max | 精确角度的斜面案例，以及层间直坡和往复坡 |
| step_max | 阶跃案例与层间楼梯，步高覆盖参考能力两侧 |
| standable_ratio | 小支撑台、缺口、粗糙地面。仅记录面积代理，实际比例由 VST 的离散平坦网格计数决定 |
| safe_margin | 双侧墙通道，净宽等于参考单侧安全距离的两倍再乘采样因子 |
| cost_barrier | 固定为标签尺度，不把它作为几何尺寸随机化，未来导出标签时归一化 |

standable_ratio 的面积代理使用固定的 0.1 m 分辨率、7 格参考窗口，用于生成样本尺寸。当前教师已恢复 kernel_size=7、standable_ratio=0.1 和 inflation=0.1 等完整参数。教师按原生邻域平坦网格计数决定代价，生成时记录的面积代理不直接作为监督标签。

随机数由 seed 派生，所有采样结果写入 config.resolved.json，支持复现。现阶段的楼层、中庭和连接位置也会变化，不只更换 WFC 地面排列。scene.json 记录旋转前网格与世界坐标之间的 yaw。

connection_layout 配置控制连接的全图布局：bay_cells 为旋转前 3×2 格楼梯井，rotations_deg 为独立朝向。连接预留一格地图边界和一格间隔，落脚周边需要有楼板支撑，不能与同层另一处楼梯井重叠或放入无支撑的中庭。合法布局通过有限的空间排布求解得到，保留每对楼层的全部五条连接。scene.json 还记录每处连接的世界坐标中心和最终朝向。

## 环境

沿用现有 Python 3.10 几何环境，其已有 trimesh、Open3D、Matplotlib。补充依赖放 datasets/cache/wfc_runtime，入口自动将该目录加入搜索路径。已有 Conda 和虚拟环境没有修改。

本机补充依赖的复现方式：

    /home/cjh/miniconda3/envs/pure3d-ptv3/bin/python -m pip --python /home/cjh/glocal2-nav/.venv/bin/python install --target datasets/cache/wfc_runtime -r common/geneator/requirements-wfc.txt

这里使用 CPU PyTorch 满足上游工具导入需求，与 PTv3 的 CUDA 训练环境分开。原生模块缓存位于 datasets/cache/wfc_tiles。
