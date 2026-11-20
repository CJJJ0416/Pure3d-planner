# Install headers
file(GLOB discrete_headers "*.h")
install(FILES ${discrete_headers} DESTINATION include/gtsam_unstable/discrete)

# Add all tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

# Add examples
add_subdirectory(examples)
