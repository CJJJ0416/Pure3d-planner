# Plan A：当前帧点云预测局部 cost 和高度图

2026-11-04已完成感知前端的地图接口与130帧推理。使用既有RTX4060完整PTv3训练1轮后的checkpoint，未重新训练或替换骨干。当前范围是PTv3与地图解码，不包含地图CNN、SRU或导航策略。

## 输出接口

`pack_plan_a_maps`只接收网络预测、当前雷达点云、重力up和固定地图配置，不读取教师标签或监督掩码。

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| maps | `[1,2,6,64,64]` float32 | 通道0为cost/60，通道1为沿重力相对雷达的地面高度，单位m |
| input_observed_hits | `[1,1,6,64,64]` bool | 仅由当前输入点的XY及重力高度分带得到的端点观测标记 |
| cost_raw | `[6,64,64]` | 网络cost乘固定尺度60 |

地图坐标轴为`[高度带, 雷达X, 雷达Y]`，分辨率0.1m，高度带边界`[-2,-1.5,-1,-0.5,0,0.5,1]`m。网格不是世界水平XY，也不把倾斜雷达Z当作物理高度。端点观测标记不是支撑或可通行标签。

3D预测表面使用固定雷达XY及**预测高度**按`z=(h-up_x*x-up_y*y)/up_z`重建，再使用保存的雷达位姿转换到世界坐标供可视化。未使用教师表面XYZ替代预测位置。

导出文件内的`gt_*`、`mask_*`和`world_from_sensor`属于离线诊断信息，不是地图网络或Actor输入。稠密预测不因缺少观测而自动填成安全或障碍。支撑仍unknown。

## 实际结果

产物目录：`outputs/plan_a_maps_20261104`。130帧均已推理并保存NPZ，各帧SHA256记录在manifest。checkpoint指纹为`416d597b68962be23511312efb34276b0831feb520e7535deae21f95ce8d72c2`。

- 在13,992个cost监督单元上，FP32推理的平均绝对误差为7.5453，尺度0..60。
- 在16,513个高度监督单元上，高度平均绝对误差为0.07807m。
- 输入端观测标记与独立NumPy计算逐帧一致，所有130帧的通道顺序、归一化、有限值及高度带范围通过。
- 重建3D点的重力高度最大误差为3.58e-7m。
- 全部结果来自训练过的同一建筑/路线，不是独立dev/test或导航结果。

验收摘要（历史验收入口已归档）。起点0、坡中52、平台129各保存6个高度带的标签/预测/绝对误差对比图，共18张。另提供3张全部6带的稠密预测概览。灰色为无有效监督，黑点表示实际输入端点，未观测区域的预测尚未核验。

## 运行

```bash
# 使用独立PTv3 CUDA环境。当前脚本不施加硬件配额。
python scripts/perception/export_plan_a.py \
  --dataset outputs/review/relocated_lidar_v2 \
  --checkpoint outputs/rtx4060_full_epoch_20261104/last.pt \
  --device cuda:0 --output outputs/plan_a_maps_new

# 使用含matplotlib的环境，生成供原生面板读取的对比纹理。
python scripts/visualization/render_plan_a_comparison.py \
  --maps outputs/plan_a_maps_new --frames 0 52 129
```

原生查看器入口为`scripts/visualization/view_plan_a_isaac.py`。它复用已有`paired_scene.usda`，在会话层添加预测覆盖，不覆盖原始场景。提供帧选择、高度带、cost/高度/误差切换和对比图面板。用户后续明确授权无硬件配额启动，原生GUI已实际运行并通过检查。原场景网格/拓扑、预测3D表面、界面回调和4张原生截图均通过，另已查看实际原生窗口。见`outputs/plan_a_maps_20261104/native/report.json`。窗口随后关闭，viewer_state为closed。

启动请求被自动审批拒绝。拒绝原因是此前有内存耗尽和主机重启记录，且审批未将“训练取消硬件配额”视为对“无配额Isaac原生可视化”的单独授权。随后用户先选择查看PNG，之后明确授权“直接启动isaac sim可视化，不管硬件限制”。启动已成功，先前审批阻碍已解决。


第52帧RGB/GT对照现已补充第二排PTv3预测，文件为`plots/frame_00052_rgb_gt_prediction.png`。预测值直接取自保存的网络张量，标记位置统一采用GT表面参照点，以比较同一地图单元的数值，不能将标记位置当作预测表面几何。

## 单独的逐点恢复可视化

新增独立诊断分支`F_lidar = F_voxel[inverse]`，恢复每个有效LiDAR点的64维特征，保留原始点位置与顺序。原地图解码头仍直接使用体素特征，训练代码和模型参数未变。新楼梯40帧和原坡道130帧均完成恢复，地图预测与既有缓存最大差为0。

第52帧新版对比图为`plots/frame_00052_rgb_gt_prediction_point_restored.png`。184×64体素特征恢复为271×64逐点特征，267个点可查预测cost/高度，4个范围外点保留为灰色。预测视图不再用GT监督掩码筛点。PCA特征颜色和cost颜色分开标注，详细说明与运行入口见[逐点恢复](POINTWISE_RESTORATION.md)。
