# SRU-SingleAscent-111824315

当前单条上楼训练所用的整栋两层建筑，seed `111824315`。
DAE 顶点、三角面、绕序和原点与源 OBJ 一致，单位米、Z 向上，显示与碰撞共用同一份网格。
保留所有楼梯、楼板、栏杆、孔洞和实体障碍，不补地面、不缩放建筑。
材质使用统一灰色，静态网格碰撞使用 ODE；接触摩擦系数为 0.8。

在工作空间编译安装后启动：

```bash
cd /home/cjh/glocal-nav2
source /opt/ros/humble/setup.bash
colcon build --packages-select unitree_gazebo --symlink-install
source install/setup.bash
ros2 launch unitree_gazebo sru_singleascent_111824315.launch.py
```

默认启动 A1、MID360 和 Gazebo，机器人放在本次上楼训练起点附近；`planner:=None`，RViz 关闭。
可追加 `planner:=scan rviz:=true`，但新建筑需要重新建立对应地图，旧 Building 地图不适用。
`x/y/height/yaw/gui/rviz/device/mid360/planner` 可以覆盖。
出生高度 `0.500 m` 是 A1 基座的初始离地高度，不能沿用 M0 的悬高。
Gazebo 继续使用工作空间现有 A1 腿部控制器；本次仅导出场景，没有部署 M0 导航策略。

场景：`worlds/sru_singleascent_111824315.world`；网格：`models/SRU-SingleAscent-111824315/meshes/building.dae`。
`export_manifest.json` 保存源网格指纹、坐标范围和格式转换校验。
