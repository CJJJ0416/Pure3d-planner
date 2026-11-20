cmake_minimum_required(VERSION 3.8)
project(scan_bspline_opt)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(scan_path_searching REQUIRED)
find_package(scan_plan_env REQUIRED)
find_package(rclcpp REQUIRED)

add_library(scan_bspline_opt
  src/uniform_bspline.cpp
  src/bspline_optimizer.cpp
  src/gradient_descent_optimizer.cpp)
target_include_directories(scan_bspline_opt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(scan_bspline_opt scan_path_searching scan_plan_env rclcpp)
target_link_libraries(scan_bspline_opt Eigen3::Eigen)

install(TARGETS scan_bspline_opt EXPORT export_bspline_opt
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(export_bspline_opt HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 scan_path_searching scan_plan_env rclcpp)

ament_package()
