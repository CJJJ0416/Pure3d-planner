# Install headers
set(subdir discrete)
file(GLOB discrete_headers "*.h")
# FIXME: exclude headers
install(FILES ${discrete_headers} DESTINATION include/gtsam/discrete)

# Add all tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
