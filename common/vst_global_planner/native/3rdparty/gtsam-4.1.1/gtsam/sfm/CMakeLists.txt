# Install headers
file(GLOB sfm_headers "*.h")
install(FILES ${sfm_headers} DESTINATION include/gtsam/sfm)

# Build tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
