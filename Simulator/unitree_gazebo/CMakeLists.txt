cmake_minimum_required(VERSION 3.8)
project(unitree_gazebo)
find_package(ament_cmake REQUIRED)
set(dependencies rclcpp gazebo_msgs geometry_msgs nav_msgs tf2 tf2_ros)
foreach(dep IN LISTS dependencies)
  find_package(${dep} REQUIRED)
endforeach()
add_executable(state_from_gazebo src/state_from_gazebo.cpp)
ament_target_dependencies(state_from_gazebo ${dependencies})
install(TARGETS state_from_gazebo DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY worlds models asset sim real scripts DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
# 整个导航框架的启动入口唯一源目录：Pure3D_ws/Entrance，按 launch/config/shell 安装。
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../Entrance/" DESTINATION share/${PROJECT_NAME}/Entrance
  PATTERN "__pycache__" EXCLUDE
  PATTERN "compare_context.launch.py" EXCLUDE)
install(FILES README.md DESTINATION share/${PROJECT_NAME})
ament_package()
