# EGO / PCT RViz2 configuration

`ego_rviz.rviz` is migrated from
`/home/cjh/glocal-nav/src/planner/plan_manage/launch/default.rviz`, referenced by
that repository's `ego_planner/launch/rviz.launch`. The source repository does
not contain a file named `ego_rviz.launch`.

After building `unitree_gazebo` and sourcing the workspace, start the unified
A1 Building simulation and visualization with:

```bash
ros2 launch unitree_gazebo gazebo.launch.py
```

The simulation, A1 spawning/controller startup and RViz2 node are defined
directly in the single `launch/gazebo.launch.py`, with no launch-file includes.
`auto_ros2.sh` calls this file directly. The former `gazeboSim.launch.py` and
`ego_rviz.launch.py` entry files are backed up under `backups/launch-fusion-*`
and removed from the active package. `rviz:=false` disables RViz2; `gui:=false` disables
the Gazebo client. Pass both for a fully headless run. `rviz_config` overrides
the configuration.

To open only RViz2 beside an already running simulation, run:

```bash
rviz2 -d /home/cjh/glocal-nav2/src/unitree_gazebo/asset/ego_rviz.rviz --ros-args -p use_sim_time:=true
```

The configuration can also be opened directly:

```bash
rviz2 -d /home/cjh/glocal-nav2/src/unitree_gazebo/asset/ego_rviz.rviz --ros-args -p use_sim_time:=true
```

The EGO displays retain their original colors and topic names. PCT cloud displays
follow the original PCT RViz configuration: `PCT tomogram` and `PCT global_points`
use XYZ, Transient Local, Reliable, depth 1, and zero decay. The cached `/pct_path`
uses Transient Local too. The initial camera centers the Building map.
Fixed Frame is `map`; the goal tool and goal pose display both use
`/move_base_simple/goal`, while clicked points use `/clicked_point`.
Planning markers, `/pct_path`, `/global_points`, `/layer_G_2`, grid-map displays
and `/path_markers` remain available. PointCloud2 subscriptions use Best Effort
QoS so they can also receive sensor publishers. The saved ROS1 TF tree and Qt
dock state are cleared so RViz2 uses the current topology and layout.

The ROS2 interactive-marker display is configured with namespace
`/basic_controls`, replacing the ROS1 `/basic_controls/update` property; see
[the Humble implementation](https://github.com/ros2/rviz/blob/humble/rviz_default_plugins/src/rviz_default_plugins/displays/interactive_markers/interactive_marker_display.cpp).

The RViz configuration needs its corresponding ROS2 publishers. The main launch
starts simulation and EGO/tracking; PCT has its separate script. For the current A1 simulation, the PointCloud2 topic
`/livox/lidar` and a RobotModel on `/robot_description` can be added separately.
Faster-LIO's independent `lio_odom` frame requires a real localization transform
before its clouds can be displayed in this `map` configuration.

Current navigation startup is `auto_ros2.sh` for simulation/EGO and
`global_planner.sh` for one-shot tomography followed by persistent PCT plan.
In `flight_type=1`, EGO owns `/basic_controls`: drag the `end_pos` six-axis
marker; `start_pos` follows robot odometry transformed to `map` and is display-only.
The marker is shown 0.5 m above the planning goal. A 0.5 s timer submits a new
goal only after at least 1 cm of position change; EGO keeps its local replanning.
Goal Pose updates the same marker and passes yaw with the next position change.
In `flight_type=3`, PCT owns the marker server and publishes the global Path.
Run only the marker server for the chosen mode.

Launch arguments: `rviz_config:=/absolute/path/config.rviz` and
`use_sim_time:=false` for non-simulation data.

## Validation (2026-10-11)

`unitree_gazebo` builds and installs the configuration and launch file. The
installed `ros2 launch` entry loads RViz2 successfully with OpenGL 4.6, without
plugin-load errors. A short run in isolated ROS domain 73 confirmed the original
planning/PCT subscriptions, `/move_base_simple/goal` PoseStamped publisher,
`/clicked_point` PointStamped publisher, `/basic_controls/update` and feedback
connections, and Best Effort subscriptions for the enabled PointCloud2 displays.
The test published no goals or movement commands and RViz2 exited cleanly.
Evidence is in `/home/cjh/glocal-nav2/test_results/ego_rviz_validation.json`
and the adjacent build/launch logs. Rendering live planner data still requires
the corresponding ROS2 publishers; that was not tested by this configuration
migration.

## Single-file launch validation (2026-10-11)

The active launch directory contains only `gazebo.launch.py`. It directly
creates gzserver, gzclient, robot_state_publisher, robot spawning, controller
startup, state_from_gazebo and RViz2; it has no IncludeLaunchDescription or
references to other launch files. The unified file builds and launches
successfully. A short isolated run with CPU inference and MID360 downsample=4
confirmed both GUI processes, active controllers, RViz2 simulation time,
advancing /clock, 8 MID360 cloud frames and 744 IMU frames. All launched
processes exited cleanly. No movement or goal commands were sent. This is a
startup/data-flow check, not a locomotion or mapping-accuracy test.

The current main launch directly defines EGO or SCAN planning/tracking nodes alongside
the simulation and RViz2. Select `planner:=ego|scan|None` (default ego); `None` runs
simulation only. `flight_type:=1|2|3` selects manual/preset/PCT goals (default 3).
SCAN selects `scan_rviz.rviz` automatically and publishes maps on `/scan/*`.
In manual mode it provides the same `/basic_controls` six-axis `end_pos` marker,
including its Z coordinate and the 0.5 m display offset; the 2D Goal tool is removed.
Its marker server is disabled in path mode so PCT can own `/basic_controls`. PCT is managed separately:
`./global_planner.sh` activates tomography once, then automatically stays in
plan mode with simulation time until Ctrl+C.
The earlier validation above records the original simulation/RViz migration.

Current evidence: `/home/cjh/glocal-nav2/test_results/ego_gazebo_single_validation.json`
and the adjacent build/launch logs.
