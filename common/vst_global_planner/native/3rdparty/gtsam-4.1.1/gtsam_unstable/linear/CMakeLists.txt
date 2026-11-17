# Install headers
file(GLOB linear_headers "*.h")
install(FILES ${linear_headers} DESTINATION include/gtsam_unstable/linear)

# Add all tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
