# SRU-RampNoRails-111824315

seed `111824315`，包含连续平滑坡面、平台、孔洞和实体障碍；没有台阶、扶手或栏杆。
网格保持米制、Z向上和原世界坐标，显示与碰撞使用同一个DAE。
`model.sdf`/`model.config` 是静态模型描述；`worlds/sru_rampnorails_111824315.world` 是完整场景。
没有添加补地平面，也没有加载机器人、导航策略或路线。
Gazebo加载前应将包内的 `models` 目录加入 `GAZEBO_MODEL_PATH`。
此导出仅提供场景文件，不会自动启动仿真。来源与精度见 `export_manifest.json`。
