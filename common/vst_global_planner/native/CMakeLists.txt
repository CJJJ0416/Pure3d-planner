cmake_minimum_required(VERSION 3.13.0)
project(planning_core CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native") # have to enable because of gtsam

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
  src
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

add_subdirectory(src/common)
add_subdirectory(3rdparty/pybind11)
add_subdirectory(src/a_star)
add_subdirectory(src/trajectory_optimization)
add_subdirectory(src/map_manager)
add_subdirectory(src/ele_planner)

# 所有 Python 扩展和共享库同目录安装，$ORIGIN RPATH 由 ROS2 包构建传入。
install(TARGETS a_star traj_opt py_map_manager ele_planner
  a_star_search map_manager gpmp_optimizer ele_planner_lib common_smoothing
  LIBRARY DESTINATION lib/global_planner/native)
