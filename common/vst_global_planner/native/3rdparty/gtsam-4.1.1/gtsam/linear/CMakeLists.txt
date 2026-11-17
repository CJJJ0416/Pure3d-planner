# Install headers
file(GLOB linear_headers "*.h")
install(FILES ${linear_headers} DESTINATION include/gtsam/linear)

# Build tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
