# 脚本目录

从仓库根目录执行；Python 脚本使用 `.venv/bin/python`。

| 目录 | 用途 |
| --- | --- |
| `environment/` | 安装依赖、编译原生规划器 |
| `scenes/` | 随机建筑、场景修改、添加障碍物 |
| `planning/` | 在已有 VST 上生成随机 GPMP 路线 |
| `visualization/` | 构建回放数据、启动网页服务、相机回放、Isaac 查看入口 |
| `training/` | 预览导航课程 |
| `release/` | 打包 Git 提交中的源码 |

```bash
.venv/bin/python common/geneator/visualization/build_scene_viewer.py outputs/demo
.venv/bin/python common/geneator/visualization/serve_scene_viewer.py outputs/demo --port 8769
```

脚本调用和文档已使用这些新路径。可复用的网页与渲染实现仍位于
`src/stair_nav/visualization/`；随机地图后台任务位于其中的 `random_maps.py`。

`scenes/simplify_navigation_ramps.py --source <saved-scene> --output <new-scene> --fixed-run <stopped-run>`
从已保存建筑派生连续坡面、无栏杆版本，并保留该单路线实验的起终点和初始朝向。
新几何必须重新运行 VST/原生规划后再注册；不覆盖源资产或恢复训练。
通用生成器也支持 `stair_structure.style="ramp"`、`stair_structure.railings=false`
及 `floor_features.railings=false`，现有默认浮空楼梯保持。

`scenes/export_gazebo_world.py` 将已有三角网格原样导出为 Gazebo world、DAE 和静态模型，
默认只生成文件，`--with-launch` 才额外输出可选的 A1 启动入口。

`scenes/generate_ramp_platform.py --config configs/ramp_platform.json --output outputs/ramp_platform`
复用现有平滑坡面和实体构件，生成独立的实心平台与连接斜坡。配置控制平台尺寸、
坡长、坡宽和过渡长度；输出 OBJ/STL/GLB、PCD、场景记录、几何校验及可离线旋转的网页。
此入口只建模，不启动 VST、规划或训练。
将配置改为 `configs/ramp_platform_side.json` 可生成沿平台右侧上升、坡顶转入主平台的 L 形坡道。

`visualization/manage_scene_viewer.py install <scene> --port 8769` 安装当前用户的常驻服务和每分钟健康检查；支持 `status`、`restart`、`stop`，详见[附录](../docs/APPENDIX.md#viewer-service)。
