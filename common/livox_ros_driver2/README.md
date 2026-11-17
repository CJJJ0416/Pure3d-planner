# Livox simulation message interfaces

This directory exports the ROS 2 package `livox_ros_driver2` with the official,
unmodified `CustomMsg` and `CustomPoint` definitions. It builds with ROS 2 Humble
without a physical Livox device or SDK.

Copied from `/home/cjh/livox_ws/src/livox_sim_interfaces`; both message hashes
match the glocal-nav2 MID360 manifest. Official definitions and licensing:
https://github.com/Livox-SDK/livox_ros_driver2/tree/master/msg and `LICENSE.txt`.

The Pure3D simulator builds this package locally via `./build_gazebo.sh`.
No `livox_ws` or other simulation overlay is needed to build the MID360 plugin.
Faster-LIO uses a separate, explicitly selected underlay at runtime.

This workspace does not include the hardware driver. Only one package named
`livox_ros_driver2` may be active; replace this interface-only package when
integrating the full hardware driver.
