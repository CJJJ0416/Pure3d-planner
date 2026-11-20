# Install headers
file(GLOB partition_headers "*.h")
install(FILES ${partition_headers} DESTINATION include/gtsam_unstable/partition)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
