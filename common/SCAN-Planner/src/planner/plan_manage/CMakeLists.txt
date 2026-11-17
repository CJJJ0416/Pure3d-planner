cmake_minimum_required(VERSION 3.8)
project(scan_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(pcl_conversions REQUIRED)
find_package(scan_bspline_opt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(scan_path_searching REQUIRED)
find_package(scan_plan_env REQUIRED)
find_package(rclcpp REQUIRED)
find_package(scan_planner_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(scan_traj_utils REQUIRED)
find_package(visualization_msgs REQUIRED)

set(scan_planner_dependencies
  ament_index_cpp scan_bspline_opt geometry_msgs nav_msgs scan_path_searching scan_plan_env
  rclcpp scan_planner_msgs sensor_msgs std_msgs tf2 tf2_geometry_msgs tf2_ros
  scan_traj_utils visualization_msgs)

add_executable(scan_planner_node
  src/scan_planner_node.cpp
  src/scan_replan_fsm.cpp
  src/planner_manager.cpp)
target_include_directories(scan_planner_node PRIVATE include)
ament_target_dependencies(scan_planner_node ${scan_planner_dependencies})
target_link_libraries(scan_planner_node Eigen3::Eigen)

foreach(target open_loop_controller closed_loop_controller go2_kinematic_sim go2_gait_publisher)
  add_executable(${target} src/${target}.cpp)
  target_include_directories(${target} PRIVATE include)
  ament_target_dependencies(${target}
    scan_bspline_opt geometry_msgs nav_msgs rclcpp scan_planner_msgs sensor_msgs
    std_msgs tf2 tf2_geometry_msgs tf2_ros)
  target_link_libraries(${target} Eigen3::Eigen)
endforeach()

add_executable(scan_workspace_bridge src/workspace_bridge.cpp)
target_include_directories(scan_workspace_bridge PRIVATE ${PCL_INCLUDE_DIRS})
ament_target_dependencies(scan_workspace_bridge interactive_markers rclcpp geometry_msgs nav_msgs sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(scan_workspace_bridge Eigen3::Eigen ${PCL_LIBRARIES})

install(TARGETS
  scan_workspace_bridge
  scan_planner_node open_loop_controller closed_loop_controller
  go2_kinematic_sim go2_gait_publisher
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME} OPTIONAL)
install(FILES launch/default.rviz DESTINATION share/${PROJECT_NAME}/rviz OPTIONAL)
install(PROGRAMS scripts/keypoint_recorder.py
  DESTINATION lib/${PROJECT_NAME} OPTIONAL)

ament_package()
