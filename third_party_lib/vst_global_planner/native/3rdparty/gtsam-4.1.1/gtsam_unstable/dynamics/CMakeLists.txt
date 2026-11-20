# Install headers
file(GLOB dynamics_headers "*.h")
install(FILES ${dynamics_headers} DESTINATION include/gtsam_unstable/dynamics)

# Add all tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
