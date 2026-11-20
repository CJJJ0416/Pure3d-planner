# Faster-LIO ROS 2 / RViz2

Source: [KangGeon10/Faster-LIO-ROS2](https://github.com/KangGeon10/Faster-LIO-ROS2).
The existing local Faster-LIO port records upstream commit `c6306ed35b209fe39626591335c019a2630957b1` in `/home/cjh/livox_ws/README.md`.
The input configuration is `/home/cjh/livox_ws/src/faster-lio/rviz_cfg/loam_livox.rviz`.
Source configuration SHA256: `0dd6f80aca8d5bcb4382c29b4d6ffdfec2632d8b206dddde7bd86f1045b23ebe`.
License: GPL-2.0; see `LICENSE_faster_lio`.

The two Faster-LIO PointCloud2 displays are embedded directly in `Entrance/config/scan.rviz`:

- `Faster-LIO accumulated scans (120 s)`: accumulated registered observations.
- `Faster-LIO current scan`: the current registered frame, Decay Time 0.
- Both subscribe to `/faster_lio/cloud_registered` with ROS 2 PointCloud2 plugins.
- SCAN uses `world` as its Fixed Frame so Gazebo truth, manual markers and the raw MID360 scan remain visible before Faster-LIO starts. The enabled `MID360 raw scan` display subscribes to `/livox/lidar`.

The Gazebo SCAN launch loads this configuration automatically. Reload the same
configuration in an already-open RViz window to apply edits. The existing
`Entrance/launch/sim/rviz.launch.py` shortcuts also load the same `Entrance/config/scan.rviz`.

Accumulated registered scans are a visualization history, not a publication of the complete internal iVox map.
